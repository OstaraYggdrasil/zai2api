from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .account_pool import AccountPool
from .admin_page import render_admin_page
from .auth import AuthService
from .config import DEFAULT_LOG_RETENTION_DAYS, Settings, settings
from .db import AccountRecord, Database, LOG_RETENTION_DAYS_KEY, LogRecord
from .prompt_assembly import assemble_prompt
from .zai_client import UpstreamResult, ZAIClient, normalize_usage


class SupportsPromptPool(Protocol):
    async def collect_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> UpstreamResult: ...

    async def stream_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> AsyncIterator[Any]: ...


class SingleClientPool:
    def __init__(self, client: ZAIClient):
        self.client = client

    async def collect_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> UpstreamResult:
        return await self.client.collect_prompt(
            prompt=prompt,
            model=model,
            enable_thinking=enable_thinking,
            auto_web_search=auto_web_search,
        )

    async def stream_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> AsyncIterator[Any]:
        async for chunk in self.client.stream_prompt(
            prompt=prompt,
            model=model,
            enable_thinking=enable_thinking,
            auto_web_search=auto_web_search,
        ):
            yield chunk


@dataclass(slots=True)
class AppServices:
    settings: Settings
    db: Database
    auth: AuthService
    prompt_pool: SupportsPromptPool
    account_pool: AccountPool | None
    managed_upstream_client: ZAIClient | None


NOTHINKING_MODEL_SUFFIX = "-nothinking"

# Public model id -> upstream chat.z.ai model id.
# Synced with the chat.z.ai webpage model selector on 2026-10-01
# (verified in-page; DOM data-value attributes: GLM-5.3-Flash -> x-preview-l,
# GLM-5.3 -> glm-5.3, GLM-5.2 -> glm-5.2).
PUBLIC_MODEL_ALIASES: dict[str, str] = {
    "glm-5.3": "glm-5.3",
    "glm-5.3-flash": "x-preview-l",
    "glm-5.2": "glm-5.2",
}

# Retired public ids kept working for existing clients; mapped to successors.
LEGACY_MODEL_ALIASES: dict[str, str] = {
    "glm-5": "glm-5.3",
    "glm-5.1": "glm-5.2",
}

# Upstream ids whose capabilities report skip_think/free_think support, i.e. the
# only ones for which a "-nothinking" variant is advertised. (On the webpage the
# deep-thinking toggle is locked ON for GLM-5.3 / GLM-5.3-Flash.)
NOTHINKING_SUPPORTED_UPSTREAM: frozenset[str] = frozenset({"glm-5.2"})

UPSTREAM_MODEL_ALIASES: dict[str, str] = {
    upstream: public for public, upstream in PUBLIC_MODEL_ALIASES.items()
}
for _legacy_public, _legacy_upstream in LEGACY_MODEL_ALIASES.items():
    # Never let a retired id shadow the canonical public id of a live model.
    UPSTREAM_MODEL_ALIASES.setdefault(_legacy_upstream, _legacy_public)


def create_app(
    app_settings: Settings | None = None,
    upstream_client: ZAIClient | None = None,
    prompt_pool: SupportsPromptPool | None = None,
    account_pool: AccountPool | None = None,
) -> FastAPI:
    resolved_settings = app_settings or settings
    db = Database(resolved_settings.database_path)
    auth = AuthService(resolved_settings, db)

    managed_upstream_client = upstream_client
    managed_account_pool = account_pool

    if prompt_pool is None:
        if managed_account_pool is not None:
            resolved_prompt_pool: SupportsPromptPool = managed_account_pool
        elif managed_upstream_client is not None:
            resolved_prompt_pool = SingleClientPool(managed_upstream_client)
        else:
            managed_account_pool = AccountPool(resolved_settings, db)
            resolved_prompt_pool = managed_account_pool
    else:
        resolved_prompt_pool = prompt_pool
        if managed_account_pool is None and isinstance(prompt_pool, AccountPool):
            managed_account_pool = prompt_pool

    services = AppServices(
        settings=resolved_settings,
        db=db,
        auth=auth,
        prompt_pool=resolved_prompt_pool,
        account_pool=managed_account_pool,
        managed_upstream_client=managed_upstream_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services.db.initialize()
        services.db.delete_expired_admin_sessions()
        services.db.delete_logs_before(log_retention_cutoff(services))
        services.db.add_log(
            level="info",
            category="startup",
            message="Application started",
            details={
                "database_path": services.settings.database_path,
                "api_auth_enabled": services.auth.is_api_auth_enabled(),
                "panel_password_source": services.auth.panel_password_source(),
                "persisted_accounts": services.db.count_accounts(),
                "poll_interval_seconds": services.settings.account_poll_interval_seconds,
                "log_retention_days": current_log_retention_days(services),
            },
        )
        app.state.services = services

        poll_task: asyncio.Task[None] | None = None
        if services.account_pool is not None and services.settings.account_poll_interval_seconds > 0:
            poll_task = asyncio.create_task(run_account_health_monitor(services))

        try:
            yield
        finally:
            if poll_task is not None:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
            if services.managed_upstream_client is not None:
                await services.managed_upstream_client.aclose()

    app = FastAPI(title="zai2api", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def root(_: Request) -> HTMLResponse:
        return HTMLResponse(render_admin_page())

    @app.get("/api/admin/bootstrap")
    async def admin_bootstrap(request: Request) -> JSONResponse:
        current_services = get_services(request)
        logged_in = current_services.auth.verify_admin_session(
            request.cookies.get(current_services.settings.admin_cookie_name)
        )
        return JSONResponse(
            {
                "logged_in": logged_in,
                "panel_password": {
                    "source": current_services.auth.panel_password_source(),
                    "default_password_active": current_services.auth.panel_password_source()
                    == "default",
                },
                "api_password": {
                    "source": current_services.auth.api_password_source(),
                    "enabled": current_services.auth.is_api_auth_enabled(),
                },
                "accounts": account_summary(current_services),
                "frontend_ready": False,
            }
        )

    @app.post("/api/admin/login")
    async def admin_login(request: Request) -> JSONResponse:
        current_services = get_services(request)
        payload = await request.json()
        password = str(payload.get("password") or "")
        if not current_services.auth.verify_panel_password(password):
            current_services.db.add_log(
                level="warning",
                category="admin_auth",
                message="Admin login failed",
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")

        session_id, expires_at = current_services.auth.create_admin_session()
        current_services.db.add_log(
            level="info",
            category="admin_auth",
            message="Admin login succeeded",
            details={"expires_at": expires_at},
        )
        response = JSONResponse({"ok": True, "expires_at": expires_at})
        response.set_cookie(
            key=current_services.settings.admin_cookie_name,
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=current_services.settings.admin_cookie_secure,
            max_age=current_services.settings.admin_session_ttl_seconds,
            expires=current_services.settings.admin_session_ttl_seconds,
            path="/",
        )
        return response

    @app.post("/api/admin/logout")
    async def admin_logout(request: Request) -> JSONResponse:
        current_services = get_services(request)
        session_id = request.cookies.get(current_services.settings.admin_cookie_name)
        current_services.auth.delete_admin_session(session_id)
        current_services.db.add_log(
            level="info",
            category="admin_auth",
            message="Admin logged out",
        )
        response = JSONResponse({"ok": True})
        response.delete_cookie(current_services.settings.admin_cookie_name, path="/")
        return response

    @app.get("/api/admin/session")
    async def admin_session(request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        return JSONResponse(
            {
                "authenticated": True,
                "panel_password_source": current_services.auth.panel_password_source(),
                "api_password": {
                    "source": current_services.auth.api_password_source(),
                    "enabled": current_services.auth.is_api_auth_enabled(),
                },
                "accounts": account_summary(current_services),
            }
        )

    @app.get("/api/admin/accounts")
    async def admin_list_accounts(request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        pool = require_managed_account_pool(current_services)
        return JSONResponse({"accounts": [serialize_account(item) for item in pool.list_accounts()]})

    @app.post("/api/admin/accounts")
    async def admin_add_account(request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        pool = require_managed_account_pool(current_services)
        payload = await request.json()
        jwt = str(payload.get("jwt") or "").strip()
        if not jwt:
            raise HTTPException(status_code=400, detail="jwt is required")
        account = await pool.register_jwt(jwt)
        return JSONResponse({"account": serialize_account(account)})

    @app.post("/api/admin/accounts/{account_id}/enable")
    async def admin_enable_account(account_id: int, request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        pool = require_managed_account_pool(current_services)
        account = pool.set_account_enabled(account_id, True)
        return JSONResponse({"account": serialize_account(account)})

    @app.post("/api/admin/accounts/{account_id}/disable")
    async def admin_disable_account(account_id: int, request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        pool = require_managed_account_pool(current_services)
        account = pool.set_account_enabled(account_id, False)
        return JSONResponse({"account": serialize_account(account)})

    @app.post("/api/admin/accounts/{account_id}/check")
    async def admin_check_account(account_id: int, request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        pool = require_managed_account_pool(current_services)
        account = await pool.check_account(account_id)
        return JSONResponse({"account": serialize_account(account)})

    @app.get("/api/admin/logs")
    async def admin_logs(request: Request, limit: int = 100) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        current_services.db.delete_logs_before(log_retention_cutoff(current_services))
        safe_limit = max(1, min(limit, 500))
        return JSONResponse({"logs": [serialize_log(item) for item in current_services.db.list_logs(safe_limit)]})

    @app.get("/api/admin/settings/security")
    async def admin_security_settings(request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        return JSONResponse(serialize_security_settings(current_services))

    @app.post("/api/admin/settings/security")
    async def admin_update_security_settings(request: Request) -> JSONResponse:
        current_services = get_services(request)
        require_admin_session(request, current_services)
        payload = await request.json()
        changed: list[str] = []

        panel_password = str(payload.get("panel_password") or "").strip()
        if panel_password:
            current_services.auth.update_panel_password(panel_password)
            changed.append("panel_password")

        if bool(payload.get("disable_api_password")):
            current_services.auth.update_api_password(None)
            changed.append("api_password_disabled")
        else:
            api_password = str(payload.get("api_password") or "").strip()
            if api_password:
                current_services.auth.update_api_password(api_password)
                changed.append("api_password")

        log_retention_days = payload.get("log_retention_days")
        if log_retention_days is not None:
            try:
                retention_days = max(1, int(log_retention_days))
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="log_retention_days must be a positive integer") from exc
            current_services.db.set_setting(LOG_RETENTION_DAYS_KEY, str(retention_days))
            current_services.db.delete_logs_before(log_retention_cutoff(current_services, retention_days))
            changed.append("log_retention_days")

        if not changed:
            raise HTTPException(status_code=400, detail="No security changes requested")

        current_services.db.add_log(
            level="info",
            category="settings",
            message="Updated security settings",
            details={"changed": changed},
        )
        return JSONResponse(serialize_security_settings(current_services))

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        current_services = get_services(request)
        enforce_api_password(request, current_services)
        payload = {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "zai2api",
                }
                for model in available_models(current_services)
            ],
        }
        current_services.db.add_log(
            level="info",
            category="requests",
            message="Listed available models",
            details={
                "path": str(request.url.path),
                "model_count": len(payload["data"]),
            },
        )
        return JSONResponse(payload)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        current_services = get_services(request)
        enforce_api_password(request, current_services)
        payload = await request.json()
        requested_model = normalize_public_model_name(
            str(payload.get("model") or current_services.settings.default_model)
        )
        upstream_model, enable_thinking = resolve_model_request(requested_model)
        messages = payload.get("messages") or []
        stream = bool(payload.get("stream"))
        prompt = assemble_prompt(messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt could be assembled from messages")

        if stream:
            return StreamingResponse(
                stream_chat_completions(
                    db=current_services.db,
                    pool=current_services.prompt_pool,
                    model=requested_model,
                    upstream_model=upstream_model,
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                ),
                media_type="text/event-stream",
            )

        try:
            upstream_result = await current_services.prompt_pool.collect_prompt(
                prompt=prompt,
                model=upstream_model,
                enable_thinking=enable_thinking,
                auto_web_search=False,
            )
        except RuntimeError as exc:
            current_services.db.add_log(
                level="warning",
                category="requests",
                message="Chat completion request failed",
                details={
                    "path": str(request.url.path),
                    "model": requested_model,
                    "stream": False,
                    "enable_thinking": enable_thinking,
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        response = {
            "id": make_chat_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": upstream_result.answer_text,
                        "reasoning_content": upstream_result.reasoning_text,
                    },
                    "finish_reason": upstream_result.finish_reason,
                }
            ],
            "usage": upstream_result.usage,
        }
        current_services.db.add_log(
            level="info",
            category="requests",
            message="Completed chat completion request",
            details={
                "path": str(request.url.path),
                "model": requested_model,
                "stream": False,
                "enable_thinking": enable_thinking,
                "prompt_chars": len(prompt),
                "answer_chars": len(upstream_result.answer_text),
                "reasoning_chars": len(upstream_result.reasoning_text),
                "total_tokens": upstream_result.usage["total_tokens"],
            },
        )
        return JSONResponse(response)

    @app.post("/v1/responses")
    async def responses_api(request: Request):
        current_services = get_services(request)
        enforce_api_password(request, current_services)
        payload = await request.json()
        requested_model = normalize_public_model_name(
            str(payload.get("model") or current_services.settings.default_model)
        )
        upstream_model, enable_thinking = resolve_model_request(requested_model)
        stream = bool(payload.get("stream"))
        prompt = assemble_responses_prompt(payload)
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt could be assembled from input")

        if stream:
            return StreamingResponse(
                stream_responses(
                    db=current_services.db,
                    pool=current_services.prompt_pool,
                    model=requested_model,
                    upstream_model=upstream_model,
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                ),
                media_type="text/event-stream",
            )

        try:
            upstream_result = await current_services.prompt_pool.collect_prompt(
                prompt=prompt,
                model=upstream_model,
                enable_thinking=enable_thinking,
                auto_web_search=False,
            )
        except RuntimeError as exc:
            current_services.db.add_log(
                level="warning",
                category="requests",
                message="Responses request failed",
                details={
                    "path": str(request.url.path),
                    "model": requested_model,
                    "stream": False,
                    "enable_thinking": enable_thinking,
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

        response = {
            "id": make_response_id(),
            "object": "response",
            "status": "completed",
            "model": requested_model,
            "output": build_responses_output(
                answer_text=upstream_result.answer_text,
                reasoning_text=upstream_result.reasoning_text,
            ),
            "usage": {
                "input_tokens": upstream_result.usage["prompt_tokens"],
                "output_tokens": upstream_result.usage["completion_tokens"],
                "total_tokens": upstream_result.usage["total_tokens"],
            },
        }
        current_services.db.add_log(
            level="info",
            category="requests",
            message="Completed responses request",
            details={
                "path": str(request.url.path),
                "model": requested_model,
                "stream": False,
                "enable_thinking": enable_thinking,
                "prompt_chars": len(prompt),
                "answer_chars": len(upstream_result.answer_text),
                "reasoning_chars": len(upstream_result.reasoning_text),
                "total_tokens": upstream_result.usage["total_tokens"],
            },
        )
        return JSONResponse(response)

    return app


async def stream_chat_completions(
    *,
    db: Database,
    pool: SupportsPromptPool,
    model: str,
    upstream_model: str,
    prompt: str,
    enable_thinking: bool,
) -> AsyncIterator[bytes]:
    completion_id = make_chat_completion_id()
    created = int(time.time())
    final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    answer_chars = 0
    reasoning_chars = 0

    yield sse_json(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    try:
        async for chunk in pool.stream_prompt(
            prompt=prompt,
            model=upstream_model,
            enable_thinking=enable_thinking,
            auto_web_search=False,
        ):
            if chunk.error:
                db.add_log(
                    level="warning",
                    category="requests",
                    message="Streaming chat completion request failed",
                    details={
                        "path": "/v1/chat/completions",
                        "model": model,
                        "stream": True,
                        "enable_thinking": enable_thinking,
                        "answer_chars": answer_chars,
                        "reasoning_chars": reasoning_chars,
                        "error": chunk.error,
                    },
                )
                yield chat_stream_error_event(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    message=chunk.error,
                )
                yield b"data: [DONE]\n\n"
                return
            if chunk.usage:
                final_usage = normalize_usage(chunk.usage)
            if not chunk.text:
                continue

            delta = (
                {"reasoning_content": chunk.text}
                if chunk.phase == "thinking"
                else {"content": chunk.text}
            )
            if chunk.phase == "thinking":
                reasoning_chars += len(chunk.text)
            else:
                answer_chars += len(chunk.text)
            yield sse_json(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
    except RuntimeError as exc:
        db.add_log(
            level="warning",
            category="requests",
            message="Streaming chat completion request failed",
            details={
                "path": "/v1/chat/completions",
                "model": model,
                "stream": True,
                "enable_thinking": enable_thinking,
                "answer_chars": answer_chars,
                "reasoning_chars": reasoning_chars,
                "error": str(exc),
            },
        )
        yield chat_stream_error_event(
            completion_id=completion_id,
            created=created,
            model=model,
            message=str(exc),
        )
        yield b"data: [DONE]\n\n"
        return

    yield sse_json(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": final_usage,
        }
    )
    db.add_log(
        level="info",
        category="requests",
        message="Completed streaming chat completion request",
        details={
            "path": "/v1/chat/completions",
            "model": model,
            "stream": True,
            "enable_thinking": enable_thinking,
            "prompt_chars": len(prompt),
            "answer_chars": answer_chars,
            "reasoning_chars": reasoning_chars,
            "total_tokens": final_usage["total_tokens"],
        },
    )
    yield b"data: [DONE]\n\n"


async def stream_responses(
    *,
    db: Database,
    pool: SupportsPromptPool,
    model: str,
    upstream_model: str,
    prompt: str,
    enable_thinking: bool,
) -> AsyncIterator[bytes]:
    response_id = make_response_id()
    created = int(time.time())
    reasoning_item_id = f"rs_{uuid.uuid4().hex}"
    message_item_id = f"msg_{uuid.uuid4().hex}"
    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    reasoning_started = False
    message_started = False
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []

    yield sse_json(
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created,
                "model": model,
                "status": "in_progress",
            },
        }
    )
    yield sse_json(
        {
            "type": "response.in_progress",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created,
                "model": model,
                "status": "in_progress",
            },
        }
    )

    try:
        async for chunk in pool.stream_prompt(
            prompt=prompt,
            model=upstream_model,
            enable_thinking=enable_thinking,
            auto_web_search=False,
        ):
            if chunk.error:
                db.add_log(
                    level="warning",
                    category="requests",
                    message="Streaming responses request failed",
                    details={
                        "path": "/v1/responses",
                        "model": model,
                        "stream": True,
                        "enable_thinking": enable_thinking,
                        "answer_chars": len("".join(answer_parts)),
                        "reasoning_chars": len("".join(reasoning_parts)),
                        "error": chunk.error,
                    },
                )
                yield response_stream_failed_event(
                    response_id=response_id,
                    created=created,
                    model=model,
                    message=chunk.error,
                )
                yield b"data: [DONE]\n\n"
                return
            if chunk.usage:
                usage = normalize_usage(chunk.usage)
                final_usage = {
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                }
            if not chunk.text:
                continue

            if chunk.phase == "thinking":
                reasoning_parts.append(chunk.text)
                if not reasoning_started:
                    reasoning_started = True
                    yield sse_json(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {"id": reasoning_item_id, "type": "reasoning", "summary": []},
                        }
                    )
                continue

            answer_parts.append(chunk.text)
            output_index = 1 if reasoning_started else 0
            if not message_started:
                message_started = True
                yield sse_json(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": message_item_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    }
                )
                yield sse_json(
                    {
                        "type": "response.content_part.added",
                        "item_id": message_item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
            yield sse_json(
                {
                    "type": "response.output_text.delta",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": chunk.text,
                }
            )
    except RuntimeError as exc:
        db.add_log(
            level="warning",
            category="requests",
            message="Streaming responses request failed",
            details={
                "path": "/v1/responses",
                "model": model,
                "stream": True,
                "enable_thinking": enable_thinking,
                "answer_chars": len("".join(answer_parts)),
                "reasoning_chars": len("".join(reasoning_parts)),
                "error": str(exc),
            },
        )
        yield response_stream_failed_event(
            response_id=response_id,
            created=created,
            model=model,
            message=str(exc),
        )
        yield b"data: [DONE]\n\n"
        return

    if reasoning_started:
        yield sse_json(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": reasoning_item_id,
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "".join(reasoning_parts)}],
                },
            }
        )

    output_index = 1 if reasoning_started else 0
    if not message_started:
        message_started = True
        yield sse_json(
            {
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "id": message_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
            }
        )
        yield sse_json(
            {
                "type": "response.content_part.added",
                "item_id": message_item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }
        )

    final_answer = "".join(answer_parts)
    yield sse_json(
        {
            "type": "response.output_text.done",
            "item_id": message_item_id,
            "output_index": output_index,
            "content_index": 0,
            "text": final_answer,
        }
    )
    yield sse_json(
        {
            "type": "response.content_part.done",
            "item_id": message_item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": final_answer, "annotations": []},
        }
    )
    yield sse_json(
        {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "id": message_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": final_answer, "annotations": []}],
            },
        }
    )

    completed = {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": build_responses_output(final_answer, "".join(reasoning_parts)),
        "usage": final_usage,
    }
    db.add_log(
        level="info",
        category="requests",
        message="Completed streaming responses request",
        details={
            "path": "/v1/responses",
            "model": model,
            "stream": True,
            "enable_thinking": enable_thinking,
            "prompt_chars": len(prompt),
            "answer_chars": len(final_answer),
            "reasoning_chars": len("".join(reasoning_parts)),
            "total_tokens": final_usage["total_tokens"],
        },
    )
    yield sse_json({"type": "response.completed", "response": completed})
    yield b"data: [DONE]\n\n"


async def run_account_health_monitor(services: AppServices) -> None:
    assert services.account_pool is not None
    interval = services.settings.account_poll_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            await services.account_pool.check_all_accounts()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            services.db.add_log(
                level="warning",
                category="accounts",
                message="Background account health check failed",
                details={"error": str(exc)},
            )


def get_services(request: Request) -> AppServices:
    return request.app.state.services  # type: ignore[return-value]


def require_admin_session(request: Request, services: AppServices) -> None:
    session_id = request.cookies.get(services.settings.admin_cookie_name)
    if services.auth.verify_admin_session(session_id):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin authentication required")


def require_managed_account_pool(services: AppServices) -> AccountPool:
    if services.account_pool is not None:
        return services.account_pool
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Account management unavailable")


def enforce_api_password(request: Request, services: AppServices) -> None:
    if not services.auth.is_api_auth_enabled():
        return
    password = services.auth.extract_api_password(request)
    if password and services.auth.verify_api_password(password):
        return
    services.db.add_log(
        level="warning",
        category="api_auth",
        message="Rejected API request due to invalid API password",
        details={"path": str(request.url.path)},
    )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API password")


def account_summary(services: AppServices) -> dict[str, Any]:
    persisted_enabled = services.db.count_accounts(enabled_only=True)
    return {
        "persisted_total": services.db.count_accounts(),
        "persisted_enabled": persisted_enabled,
        "using_env_fallback": persisted_enabled == 0
        and bool(services.settings.zai_jwt or services.settings.zai_session_token),
    }


def serialize_account(account: AccountRecord) -> dict[str, Any]:
    return {
        "id": account.id,
        "user_id": account.user_id,
        "email": account.email,
        "name": account.name,
        "enabled": account.enabled,
        "status": account.status,
        "last_checked_at": account.last_checked_at,
        "last_error": account.last_error,
        "failure_count": account.failure_count,
        "masked_jwt": mask_secret(account.jwt),
        "masked_session_token": mask_secret(account.session_token),
        "created_at": account.created_at,
        "updated_at": account.updated_at,
    }


def serialize_log(record: LogRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "created_at": record.created_at,
        "level": record.level,
        "category": record.category,
        "message": record.message,
        "details": record.details,
    }


def serialize_security_settings(services: AppServices) -> dict[str, Any]:
    panel_source = services.auth.panel_password_source()
    api_source = services.auth.api_password_source()
    log_retention_source = log_retention_days_source(services)
    return {
        "panel_password": {
            "source": panel_source,
            "default_password_active": panel_source == "default",
            "overridden_by_env": panel_source == "env",
        },
        "api_password": {
            "source": api_source,
            "enabled": services.auth.is_api_auth_enabled(),
            "overridden_by_env": api_source == "env",
        },
        "log_retention": {
            "days": current_log_retention_days(services),
            "source": log_retention_source,
            "overridden_by_env": log_retention_source == "env",
            "default_active": log_retention_source == "default",
        },
        "poll_interval_seconds": services.settings.account_poll_interval_seconds,
    }


def available_models(services: AppServices) -> list[str]:
    models: list[str] = []
    default_model = normalize_public_model_name(services.settings.default_model)
    if default_model not in models:
        models.append(default_model)
    for model in PUBLIC_MODEL_ALIASES:
        if model not in models:
            models.append(model)
    result: list[str] = []
    for model in models:
        result.append(model)
        upstream = public_to_upstream_model(strip_nothinking_suffix(model))
        if upstream in NOTHINKING_SUPPORTED_UPSTREAM:
            result.append(f"{model}{NOTHINKING_MODEL_SUFFIX}")
    return result


def strip_nothinking_suffix(model: str) -> str:
    if model.lower().endswith(NOTHINKING_MODEL_SUFFIX):
        return model[: -len(NOTHINKING_MODEL_SUFFIX)] or model
    return model


def resolve_model_request(requested_model: str) -> tuple[str, bool]:
    normalized_model = normalize_public_model_name(requested_model)
    if normalized_model.lower().endswith(NOTHINKING_MODEL_SUFFIX):
        public_model = strip_nothinking_suffix(normalized_model)
        return public_to_upstream_model(public_model), False
    return public_to_upstream_model(normalized_model), True


def normalize_public_model_name(requested_model: str) -> str:
    normalized_model = requested_model.strip()
    if not normalized_model:
        return "glm-5.3"
    has_nothinking = normalized_model.lower().endswith(NOTHINKING_MODEL_SUFFIX)
    base_model = strip_nothinking_suffix(normalized_model) if has_nothinking else normalized_model
    public_model = canonical_public_model_name(base_model)
    if has_nothinking:
        return f"{public_model}{NOTHINKING_MODEL_SUFFIX}"
    return public_model


def public_to_upstream_model(public_model: str) -> str:
    if public_model in PUBLIC_MODEL_ALIASES:
        return PUBLIC_MODEL_ALIASES[public_model]
    return LEGACY_MODEL_ALIASES.get(public_model, public_model)


def canonical_public_model_name(requested_model: str) -> str:
    lower_model = requested_model.lower()
    if lower_model in PUBLIC_MODEL_ALIASES or lower_model in LEGACY_MODEL_ALIASES:
        return lower_model
    return UPSTREAM_MODEL_ALIASES.get(requested_model, requested_model)


def current_log_retention_days(services: AppServices) -> int:
    if services.settings.log_retention_days_env is not None:
        return max(1, int(services.settings.log_retention_days_env))
    stored_value = services.db.get_setting(LOG_RETENTION_DAYS_KEY)
    if stored_value is not None:
        try:
            return max(1, int(stored_value))
        except ValueError:
            pass
    return DEFAULT_LOG_RETENTION_DAYS


def log_retention_days_source(services: AppServices) -> str:
    if services.settings.log_retention_days_env is not None:
        return "env"
    if services.db.get_setting(LOG_RETENTION_DAYS_KEY) is not None:
        return "database"
    return "default"


def log_retention_cutoff(services: AppServices, retention_days: int | None = None) -> int:
    days = retention_days if retention_days is not None else current_log_retention_days(services)
    return int(time.time()) - (max(1, int(days)) * 86400)


def mask_secret(secret: str | None, prefix: int = 6, suffix: int = 4) -> str | None:
    if secret is None:
        return None
    if len(secret) <= prefix + suffix:
        return "*" * len(secret)
    return f"{secret[:prefix]}***{secret[-suffix:]}"


def assemble_responses_prompt(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("input"), str):
        return assemble_prompt([{"role": "user", "content": payload["input"]}])

    if isinstance(payload.get("input"), list):
        input_payload = payload["input"]
        if all(isinstance(item, dict) and "role" in item for item in input_payload):
            return assemble_prompt(input_payload)
        return assemble_prompt([{"role": "user", "content": normalize_input_blocks(input_payload)}])

    if isinstance(payload.get("messages"), list):
        return assemble_prompt(payload["messages"])

    return ""


def normalize_input_blocks(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        if item.get("type") in {"input_text", "text", "output_text"}:
            value = item.get("text") or item.get("content")
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def build_responses_output(answer_text: str, reasoning_text: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if reasoning_text:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex}",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
            }
        )
    output.append(
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": answer_text, "annotations": []}],
        }
    )
    return output


def chat_stream_error_event(
    *,
    completion_id: str,
    created: int,
    model: str,
    message: str,
) -> bytes:
    return sse_json(
        {
            "error": {
                "message": message,
                "type": "upstream_error",
                "code": "upstream_stream_error",
            },
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
    )


def response_stream_failed_event(
    *,
    response_id: str,
    created: int,
    model: str,
    message: str,
) -> bytes:
    return sse_json(
        {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created,
                "model": model,
                "status": "failed",
                "error": {
                    "message": message,
                    "type": "upstream_error",
                    "code": "upstream_stream_error",
                },
            },
        }
    )


def sse_json(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def make_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def make_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"
