from __future__ import annotations

from fastapi.testclient import TestClient

from zai2api.account_pool import AccountPool
from zai2api.browser_client import (
    UPSTREAM_MODEL_LABELS,
    make_browser_client_factory,
)
from zai2api.config import Settings
from zai2api.db import Database
from zai2api.server import create_app
from zai2api.zai_client import UpstreamChunk, parse_sse_line


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        host="127.0.0.1",
        port=8000,
        log_level="info",
        zai_base_url="https://chat.z.ai",
        zai_jwt="jwt",
        zai_session_token=None,
        default_model="glm-5.3",
        request_timeout=120.0,
        database_path=str(tmp_path / "state.db"),
        panel_password_env=None,
        api_password_env=None,
        admin_cookie_name="zai2api_admin_session",
        admin_session_ttl_hours=24,
        admin_cookie_secure=False,
        account_poll_interval_seconds=0,
    )
    base.update(overrides)
    return Settings(**base)


def test_parse_sse_line_answer_chunk():
    line = (
        'data: {"type":"chat:completion","data":'
        '{"phase":"answer","delta_content":"hi","done":false}}'
    )
    done, chunk = parse_sse_line(line)
    assert done is False
    assert isinstance(chunk, UpstreamChunk)
    assert chunk.phase == "answer"
    assert chunk.text == "hi"
    assert chunk.done is False


def test_parse_sse_line_thinking_and_usage():
    line = (
        'data: {"type":"chat:completion","data":'
        '{"phase":"thinking","delta_content":"hmm","usage":'
        '{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"done":true}}'
    )
    done, chunk = parse_sse_line(line)
    assert done is False
    assert chunk is not None
    assert chunk.phase == "thinking"
    assert chunk.usage == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert chunk.done is True


def test_parse_sse_line_done_and_ignorable():
    done, chunk = parse_sse_line("data: [DONE]")
    assert done is True
    assert chunk is None

    done, chunk = parse_sse_line("")
    assert done is False and chunk is None

    done, chunk = parse_sse_line('data: {"type":"other","data":{}}')
    assert done is False and chunk is None

    done, chunk = parse_sse_line("data: not-json{")
    assert done is False and chunk is None


def test_parse_sse_line_error():
    line = (
        'data: {"type":"chat:completion","data":'
        '{"error":{"detail":"FRONTEND_CAPTCHA_REQUIRED"}}}'
    )
    done, chunk = parse_sse_line(line)
    assert done is False
    assert chunk is not None
    assert chunk.error == "FRONTEND_CAPTCHA_REQUIRED"
    assert chunk.done is True


def test_upstream_model_labels_cover_webpage_models():
    assert UPSTREAM_MODEL_LABELS == {
        "glm-5.3": "GLM-5.3",
        "x-preview-l": "GLM-5.3-Flash",
        "glm-5.2": "GLM-5.2",
    }


def test_browser_transport_is_default_and_wires_pool(tmp_path):
    settings = make_settings(tmp_path, transport="browser")
    app = create_app(app_settings=settings)
    with TestClient(app):
        services = app.state.services
        assert services.managed_browser is not None
        assert isinstance(services.account_pool, AccountPool)
        assert services.prompt_pool is services.account_pool


def test_http_transport_wires_plain_pool(tmp_path):
    settings = make_settings(tmp_path, transport="http")
    app = create_app(app_settings=settings)
    with TestClient(app):
        services = app.state.services
        assert services.managed_browser is None
        assert isinstance(services.account_pool, AccountPool)


def test_browser_client_factory_builds_clients(tmp_path, monkeypatch):
    # This sandbox ships a malformed no_proxy list; sanitize it so the
    # httpx client construction under test does not depend on it.
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,::1")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
    from zai2api.browser_client import BrowserZAIClient, SharedBrowser

    settings = make_settings(tmp_path, transport="browser")
    shared = SharedBrowser(settings)
    factory = make_browser_client_factory(settings, shared)
    client = factory("jwt", None)
    assert isinstance(client, BrowserZAIClient)
    assert isinstance(client._shared, SharedBrowser)


def test_browser_shared_browser_not_started_by_wiring(tmp_path):
    # Creating the app must not launch Chromium; the browser starts lazily
    # on the first completion request.
    from zai2api.browser_client import SharedBrowser

    settings = make_settings(tmp_path, transport="browser")
    shared = SharedBrowser(settings)
    assert shared._started is False
    assert shared._page is None
