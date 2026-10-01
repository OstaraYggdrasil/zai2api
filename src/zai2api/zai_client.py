from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx

from .config import Settings

FE_VERSION_FALLBACK = "prod-fe-1.1.98"
FE_VERSION_PATTERN = re.compile(rb"prod-fe-[0-9.]+")
SIGNING_SECRET = "key-@@@@)))()((9))-xxxx&&&%%%%%"
USER_AGENT = "Mozilla/5.0"


@dataclass(slots=True)
class SessionState:
    token: str
    user_id: str
    name: str
    email: str
    role: str


@dataclass(slots=True)
class UpstreamChunk:
    phase: str | None
    text: str
    usage: dict[str, int] | None = None
    done: bool = False
    error: str | None = None


@dataclass(slots=True)
class UpstreamResult:
    answer_text: str
    reasoning_text: str
    usage: dict[str, int]
    finish_reason: str


def parse_sse_line(line: str) -> tuple[bool, UpstreamChunk | None]:
    """Parse a single SSE ``data:`` line from the v2 completion stream.

    Returns ``(done, chunk)`` where ``done`` is True only for the ``[DONE]``
    terminator and ``chunk`` is None for blank lines, non-data lines and
    foreign events. Error payloads come back as a chunk with ``error`` set.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return False, None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return True, None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return False, None
    if event.get("type") != "chat:completion":
        return False, None

    data = event.get("data", {})
    error = data.get("error")
    if error:
        detail = error.get("detail") if isinstance(error, dict) else str(error)
        return False, UpstreamChunk(phase=None, text="", done=True, error=detail)

    usage = normalize_usage(data.get("usage")) if data.get("usage") else None
    text = data.get("delta_content") or data.get("content") or ""
    return False, UpstreamChunk(
        phase=data.get("phase") or "answer",
        text=text,
        usage=usage,
        done=bool(data.get("done")),
    )


class ZAIClient:
    def __init__(
        self,
        settings: Settings,
        *,
        zai_jwt: str | None = None,
        zai_session_token: str | None = None,
    ):
        self.settings = settings
        self.zai_jwt = settings.zai_jwt if zai_jwt is None else zai_jwt
        self.zai_session_token = (
            settings.zai_session_token if zai_session_token is None else zai_session_token
        )
        self._client = httpx.AsyncClient(
            base_url=settings.zai_base_url,
            timeout=settings.request_timeout,
            headers={
                "User-Agent": USER_AGENT,
                "X-FE-Version": FE_VERSION_FALLBACK,
                "Accept-Language": "en-US",
            },
        )
        self._lock = asyncio.Lock()
        self._fe_version: str | None = None
        self._session: SessionState | None = None
        if self.zai_session_token:
            self._session = self._session_from_token(self.zai_session_token)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def current_fe_version(self) -> str:
        """Best-effort fetch of the site's current frontend version.

        chat.z.ai rejects API calls whose X-FE-Version is stale ("Please refresh
        the page to update the app"), so refresh it from the homepage and cache
        the result. Falls back to the last known good version on any failure.
        """
        if self._fe_version:
            return self._fe_version
        try:
            response = await self._client.get("/")
            match = FE_VERSION_PATTERN.search(response.content)
            if match:
                self._fe_version = match.group(0).decode()
        except Exception:
            pass
        if not self._fe_version:
            self._fe_version = FE_VERSION_FALLBACK
        self._client.headers["X-FE-Version"] = self._fe_version
        return self._fe_version

    async def ensure_session(self, force_refresh: bool = False) -> SessionState:
        async with self._lock:
            if self._session and not force_refresh:
                return self._session

            await self.current_fe_version()

            if self.zai_jwt:
                response = await self._client.get(
                    "/api/v1/auths/",
                    headers={"Authorization": f"Bearer {self.zai_jwt}"},
                )
                response.raise_for_status()
                payload = response.json()
                self._session = SessionState(
                    token=payload["token"],
                    user_id=payload["id"],
                    name=payload.get("name", ""),
                    email=payload.get("email", ""),
                    role=payload.get("role", "user"),
                )
                return self._session

            if self._session:
                return self._session

            raise RuntimeError("Missing ZAI_JWT or ZAI_SESSION_TOKEN")

    async def verify_completion_version(self) -> int:
        session = await self.ensure_session()
        response = await self._client.get(
            "/api/config",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        response.raise_for_status()
        return int(response.json().get("completion_version", 1))

    async def create_chat(
        self,
        *,
        session: SessionState,
        model: str,
        prompt: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> tuple[str, str]:
        user_message_id = str(uuid.uuid4())
        chat = {
            "id": "",
            "title": "New Chat",
            "models": [model],
            "history": {
                "currentId": user_message_id,
                "messages": {
                    user_message_id: {
                        "id": user_message_id,
                        "parentId": None,
                        "childrenIds": [],
                        "role": "user",
                        "content": prompt,
                        "timestamp": int(time.time()),
                        "models": [model],
                    }
                },
            },
            "tags": [],
            "flags": [],
            "features": [],
            "mcp_servers": [],
            "enable_thinking": enable_thinking,
            "auto_web_search": auto_web_search,
            "message_version": 1,
            "timestamp": int(time.time() * 1000),
        }
        response = await self._client.post(
            "/api/v1/chats/new",
            headers={"Authorization": f"Bearer {session.token}"},
            json={"chat": chat},
        )
        response.raise_for_status()
        payload = response.json()
        return payload["id"], user_message_id

    async def stream_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> AsyncIterator[UpstreamChunk]:
        for attempt in range(2):
            session = await self.ensure_session(force_refresh=attempt > 0)
            try:
                record_id, user_message_id = await self.create_chat(
                    session=session,
                    model=model,
                    prompt=prompt,
                    enable_thinking=enable_thinking,
                    auto_web_search=auto_web_search,
                )
                assistant_message_id = str(uuid.uuid4())
                async for chunk in self._open_completion_stream(
                    session=session,
                    record_id=record_id,
                    user_message_id=user_message_id,
                    assistant_message_id=assistant_message_id,
                    prompt=prompt,
                    model=model,
                    enable_thinking=enable_thinking,
                    auto_web_search=auto_web_search,
                ):
                    yield chunk
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 401 and self.zai_jwt and attempt == 0:
                    continue
                raise

    async def collect_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> UpstreamResult:
        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, int] | None = None

        async for chunk in self.stream_prompt(
            prompt=prompt,
            model=model,
            enable_thinking=enable_thinking,
            auto_web_search=auto_web_search,
        ):
            if chunk.error:
                raise RuntimeError(chunk.error)
            if chunk.phase == "thinking":
                reasoning_parts.append(chunk.text)
            else:
                answer_parts.append(chunk.text)
            if chunk.usage:
                usage = chunk.usage

        return UpstreamResult(
            answer_text="".join(answer_parts),
            reasoning_text="".join(reasoning_parts),
            usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
        )

    async def _open_completion_stream(
        self,
        *,
        session: SessionState,
        record_id: str,
        user_message_id: str,
        assistant_message_id: str,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> AsyncIterator[UpstreamChunk]:
        timestamp_ms = str(int(time.time() * 1000))
        request_id = str(uuid.uuid4())
        signature = self._sign_prompt(
            request_id=request_id,
            timestamp_ms=timestamp_ms,
            user_id=session.user_id,
            prompt=prompt,
        )
        query = self._build_query(
            session_token=session.token,
            user_id=session.user_id,
            request_id=request_id,
            timestamp_ms=timestamp_ms,
        )
        body = {
            "stream": True,
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "signature_prompt": prompt,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": auto_web_search,
                "preview_mode": False,
                "flags": [],
                "enable_thinking": enable_thinking,
            },
            "variables": self._default_variables(session.name),
            "chat_id": record_id,
            "id": assistant_message_id,
            "current_user_message_id": user_message_id,
            "current_user_message_parent_id": None,
            "background_tasks": {
                "title_generation": True,
                "tags_generation": True,
            },
            "stream_options": {"include_usage": True},
        }
        path = f"/api/v2/chat/completions?{urlencode(query)}&signature_timestamp={timestamp_ms}"
        headers = {
            "Authorization": f"Bearer {session.token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Signature": signature,
        }
        async with self._client.stream("POST", path, headers=headers, json=body) as response:
            response.raise_for_status()
            async for chunk in self._iter_sse(response):
                yield chunk

    async def _iter_sse(self, response: httpx.Response) -> AsyncIterator[UpstreamChunk]:
        async for line in response.aiter_lines():
            done, chunk = parse_sse_line(line)
            if done:
                break
            if chunk is None:
                continue
            yield chunk

    def _sign_prompt(
        self,
        *,
        request_id: str,
        timestamp_ms: str,
        user_id: str,
        prompt: str,
    ) -> str:
        payload = {"requestId": request_id, "timestamp": timestamp_ms, "user_id": user_id}
        sorted_payload = ",".join(
            ",".join((key, value)) for key, value in sorted(payload.items())
        )
        prompt_b64 = base64.b64encode(prompt.encode()).decode()
        bucket = str(int(int(timestamp_ms) // (5 * 60 * 1000)))
        key1 = hmac.new(SIGNING_SECRET.encode(), bucket.encode(), hashlib.sha256).hexdigest()
        return hmac.new(
            key1.encode(),
            f"{sorted_payload}|{prompt_b64}|{timestamp_ms}".encode(),
            hashlib.sha256,
        ).hexdigest()

    def _build_query(
        self,
        *,
        session_token: str,
        user_id: str,
        request_id: str,
        timestamp_ms: str,
    ) -> dict[str, str]:
        return {
            "requestId": request_id,
            "timestamp": timestamp_ms,
            "user_id": user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": session_token,
            "user_agent": USER_AGENT,
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": "Asia/Taipei",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "1080",
            "viewport_width": "1920",
            "viewport_size": "1920x1080",
            "color_depth": "24",
            "pixel_ratio": "1",
            "current_url": "https://chat.z.ai/",
            "pathname": "/",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "https://chat.z.ai/",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5 & GLM-4.7",
            "timezone_offset": "-480",
            "local_time": time.strftime("%a %b %d %Y %H:%M:%S GMT%z"),
            "utc_time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Chrome",
            "os_name": "Linux",
        }

    def _default_variables(self, name: str) -> dict[str, str]:
        return {
            "{{USER_NAME}}": name,
            "{{USER_LOCATION}}": "Unknown",
            "{{CURRENT_DATETIME}}": time.strftime("%Y-%m-%d %I:%M:%S %p"),
            "{{CURRENT_DATE}}": time.strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": time.strftime("%I:%M:%S %p"),
            "{{CURRENT_WEEKDAY}}": time.strftime("%A"),
            "{{CURRENT_TIMEZONE}}": "UTC+8",
            "{{USER_LANGUAGE}}": "en-US",
        }

    def _session_from_token(self, token: str) -> SessionState:
        try:
            payload_b64 = token.split(".")[1]
            padded = payload_b64 + "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
            return SessionState(
                token=token,
                user_id=payload.get("id", "unknown"),
                name=payload.get("name", "unknown"),
                email=payload.get("email", ""),
                role=payload.get("role", "user"),
            )
        except Exception:
            return SessionState(token=token, user_id="unknown", name="unknown", email="", role="user")


def normalize_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
