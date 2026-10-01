from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from zai2api.config import Settings
from zai2api.server import create_app
from zai2api.zai_client import UpstreamResult


class FakeUpstreamClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def collect_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> UpstreamResult:
        self.calls.append(
            {
                "prompt": prompt,
                "model": model,
                "enable_thinking": enable_thinking,
                "auto_web_search": auto_web_search,
            }
        )
        return UpstreamResult(
            answer_text=f"echo:{prompt}",
            reasoning_text="fake reasoning",
            usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            finish_reason="stop",
        )

    async def stream_prompt(self, **_: object):
        if False:
            yield None

    async def aclose(self) -> None:
        return None


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = Settings(
        host="127.0.0.1",
        port=8000,
        log_level="info",
        zai_base_url="https://chat.z.ai",
        zai_jwt=None,
        zai_session_token=None,
        default_model="glm-5.3",
        request_timeout=120.0,
        database_path=str(tmp_path / "state.db"),
        panel_password_env=None,
        api_password_env=None,
        admin_cookie_name="zai2api_admin_session",
        admin_session_ttl_hours=24,
        admin_cookie_secure=False,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_default_panel_password_login_flow(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path), upstream_client=FakeUpstreamClient())

    with TestClient(app) as client:
        bootstrap = client.get("/api/admin/bootstrap")
        assert bootstrap.status_code == 200
        assert bootstrap.json()["panel_password"]["source"] == "default"
        assert bootstrap.json()["api_password"]["enabled"] is False

        login = client.post("/api/admin/login", json={"password": "123456"})
        assert login.status_code == 200
        assert "zai2api_admin_session" in client.cookies

        session = client.get("/api/admin/session")
        assert session.status_code == 200
        assert session.json()["authenticated"] is True


def test_api_auth_is_disabled_by_default(tmp_path: Path) -> None:
    upstream = FakeUpstreamClient()
    app = create_app(make_settings(tmp_path), upstream_client=upstream)

    with TestClient(app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        model_ids = [item["id"] for item in models.json()["data"]]
        assert model_ids == [
            "glm-5.3",
            "glm-5.3-flash",
            "glm-5.2",
            "glm-5.2-nothinking",
        ]

        response = client.post(
            "/v1/chat/completions",
            json={"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"].startswith("echo:")
        assert upstream.calls[-1]["model"] == "glm-5.3"
        assert upstream.calls[-1]["enable_thinking"] is True


def test_nothinking_model_variant_disables_thinking(tmp_path: Path) -> None:
    upstream = FakeUpstreamClient()
    app = create_app(make_settings(tmp_path), upstream_client=upstream)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "glm-5.2-nothinking", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "glm-5.2-nothinking"
        assert upstream.calls[-1]["model"] == "glm-5.2"
        assert upstream.calls[-1]["enable_thinking"] is False


def test_legacy_model_aliases_map_to_successors(tmp_path: Path) -> None:
    upstream = FakeUpstreamClient()
    app = create_app(make_settings(tmp_path), upstream_client=upstream)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={"model": "glm-5", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert first.status_code == 200
        assert upstream.calls[-1]["model"] == "glm-5.3"
        assert upstream.calls[-1]["enable_thinking"] is True

        second = client.post(
            "/v1/chat/completions",
            json={"model": "glm-5.1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert second.status_code == 200
        assert upstream.calls[-1]["model"] == "glm-5.2"
        assert upstream.calls[-1]["enable_thinking"] is True


def test_api_auth_rejects_missing_or_invalid_password(tmp_path: Path) -> None:
    app = create_app(
        make_settings(tmp_path, api_password_env="secret-key"),
        upstream_client=FakeUpstreamClient(),
    )

    with TestClient(app) as client:
        rejected_models = client.get("/v1/models")
        assert rejected_models.status_code == 401

        allowed_models = client.get(
            "/v1/models",
            headers={"authorization": "Bearer secret-key"},
        )
        assert allowed_models.status_code == 200
        assert allowed_models.json()["data"][0]["owned_by"] == "zai2api"

        rejected = client.post(
            "/v1/responses",
            json={"model": "glm-5", "input": "hello"},
        )
        assert rejected.status_code == 401

        allowed = client.post(
            "/v1/responses",
            headers={"authorization": "Bearer secret-key"},
            json={"model": "glm-5", "input": "hello"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["usage"]["total_tokens"] == 3
