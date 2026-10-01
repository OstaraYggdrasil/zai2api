"""Browser-backed transport for chat.z.ai.

The raw HTTP completion endpoint rejects requests without a
``captcha_verify_param`` that only the site's own JavaScript can mint
(Alibaba Cloud CAPTCHA, ``FRONTEND_CAPTCHA_REQUIRED``). A headed Chromium
with a persistent profile passes that check transparently, so this module
drives the real chat UI and captures the page's own
``/api/v2/chat/completions`` SSE stream via an injected ``fetch`` hook.

It implements the same interface as :class:`ZAIClient` (see
``SupportsZAIClient`` in ``account_pool.py``), reusing the HTTP client for
auth/session work and the browser only for the completion stream.

Multi-account: :class:`BrowserPool` keeps one persistent headed Chromium
per chat.z.ai account (keyed by the account's user id, stable across
session-token refreshes), each with its own profile directory, so accounts
never share storage or tokens. Requests for the same account are
serialized through that account's browser lock; different accounts run in
parallel in their own browsers.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
from typing import Any, AsyncIterator, Callable

from .config import Settings
from .zai_client import (
    SessionState,
    UpstreamChunk,
    UpstreamResult,
    ZAIClient,
    parse_sse_line,
)

# Upstream model id -> label shown in the webpage model selector.
UPSTREAM_MODEL_LABELS = {
    "glm-5.3": "GLM-5.3",
    "x-preview-l": "GLM-5.3-Flash",
    "glm-5.2": "GLM-5.2",
}

# Installed via add_init_script so it runs before the page's own scripts on
# every navigation. Captures the raw SSE bytes of the page's completion
# request and optionally rewrites feature flags in the outgoing body (the
# HMAC signature only covers the signature prompt, so this is safe).
FETCH_HOOK_JS = r"""
(() => {
  if (window.__zaiHookInstalled) return;
  window.__zaiHookInstalled = true;
  window.__zaiChunks = [];
  window.__zaiReqConfig = {enable_thinking: true, auto_web_search: false};
  window.__zaiDone = false;
  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (typeof url === "string" && url.includes("/api/v2/chat/completions")) {
      try {
        const raw = init && init.body;
        if (typeof raw === "string") {
          const body = JSON.parse(raw);
          if (body && body.features) {
            body.features.enable_thinking = !!window.__zaiReqConfig.enable_thinking;
            body.features.auto_web_search = !!window.__zaiReqConfig.auto_web_search;
          }
          init = Object.assign({}, init, {body: JSON.stringify(body)});
        }
      } catch (e) { /* send the request untouched */ }
      window.__zaiChunks = [];
      window.__zaiDone = false;
      const resp = await origFetch(input, init);
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            for (;;) {
              const {done, value} = await reader.read();
              if (done) { window.__zaiDone = true; controller.close(); break; }
              window.__zaiChunks.push(decoder.decode(value, {stream: true}));
              controller.enqueue(value);
            }
          } catch (e) { window.__zaiDone = true; controller.error(e); }
        }
      });
      return new Response(stream, {status: resp.status, statusText: resp.statusText, headers: resp.headers});
    }
    return origFetch(input, init);
  };
})();
"""

# The login token is kept in a mutable window property (instead of being
# baked into the init script) so refreshed session tokens propagate to
# every later navigation.
TOKEN_INIT_JS = (
    "window.__zaiToken = {token};\n"
    "try { if (window.__zaiToken) localStorage.setItem('token', window.__zaiToken); } catch (e) {}\n"
)

_xvfb_proc: subprocess.Popen[bytes] | None = None
_xvfb_lock = asyncio.Lock()


async def _ensure_xvfb() -> None:
    """Start a process-wide Xvfb when no display is available (idempotent)."""
    global _xvfb_proc
    if os.environ.get("DISPLAY"):
        return
    async with _xvfb_lock:
        if os.environ.get("DISPLAY"):
            return
        if _xvfb_proc is not None and _xvfb_proc.poll() is None:
            os.environ["DISPLAY"] = ":99"
            return
        try:
            _xvfb_proc = subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1366x900x24"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The browser transport needs a display: install Xvfb "
                "or run under xvfb-run."
            ) from exc
        os.environ["DISPLAY"] = ":99"
        await asyncio.sleep(1.0)


def _teardown_xvfb() -> None:
    global _xvfb_proc
    if _xvfb_proc is not None:
        try:
            _xvfb_proc.terminate()
        except Exception:
            pass
        _xvfb_proc = None


class SharedBrowser:
    """One persistent headed Chromium logged into a single chat.z.ai account."""

    def __init__(self, settings: Settings, profile_dir: str):
        self._settings = settings
        self._profile_dir = profile_dir
        self._lock = asyncio.Lock()
        self._pw: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._started = False

    def lock(self) -> asyncio.Lock:
        return self._lock

    async def ensure_page(self, token: str) -> Any:
        """Start the browser on first use (or relaunch if it died) and
        return the page, with the current login token applied."""
        if (
            self._started
            and self._page is not None
            and not self._page.is_closed()
        ):
            await self._apply_token(self._page, token)
            return self._page

        await self._reset()
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "The browser transport needs the 'playwright' package "
                "(pip install playwright) and a Chromium build."
            ) from exc

        await _ensure_xvfb()
        self._pw = await async_playwright().start()
        proxy = self._proxy_config()
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": self._profile_dir,
            "headless": False,
            "ignore_https_errors": True,
            "user_agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "viewport": {"width": 1366, "height": 900},
            "locale": "zh-CN",
            "timezone_id": "Asia/Shanghai",
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        }
        if proxy:
            launch_kwargs["proxy"] = proxy
        self._ctx = await self._pw.chromium.launch_persistent_context(**launch_kwargs)
        await self._ctx.add_init_script(
            TOKEN_INIT_JS.format(token=json.dumps(token)) + FETCH_HOOK_JS
        )
        self._page = await self._ctx.new_page()
        await self._page.goto(
            self._settings.zai_base_url + "/", wait_until="domcontentloaded"
        )
        await self._page.wait_for_timeout(4000)
        await self._page.keyboard.press("Escape")
        self._started = True
        return self._page

    @staticmethod
    async def _apply_token(page: Any, token: str) -> None:
        try:
            await page.evaluate(TOKEN_INIT_JS.format(token=json.dumps(token)))
        except Exception:
            pass

    def _proxy_config(self) -> dict[str, str] | None:
        proxy_url = self._settings.browser_proxy
        if not proxy_url:
            # This sandbox routes outbound traffic through a local forward
            # proxy; use it when present, otherwise go direct.
            try:
                sock = socket.create_connection(("127.0.0.1", 8899), timeout=1.5)
                sock.close()
                proxy_url = "http://127.0.0.1:8899"
            except OSError:
                proxy_url = None
        return {"server": proxy_url} if proxy_url else None

    async def _reset(self) -> None:
        if self._ctx is not None:
            try:
                await self._ctx.close()
            except Exception:
                pass
            self._ctx = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._page = None
        self._started = False

    async def aclose(self) -> None:
        await self._reset()


class BrowserPool:
    """Owns one :class:`SharedBrowser` per chat.z.ai account.

    Keyed by the account's user id (stable across session-token refreshes);
    each account gets its own persistent profile directory under
    ``settings.browser_profile_dir``. Browsers start lazily on first use.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._browsers: dict[str, SharedBrowser] = {}
        self._lock = asyncio.Lock()

    def profile_dir_for(self, user_id: str) -> str:
        safe = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return os.path.join(self._settings.browser_profile_dir, f"profile-{safe}")

    async def get(self, user_id: str) -> SharedBrowser:
        async with self._lock:
            browser = self._browsers.get(user_id)
            if browser is None:
                browser = SharedBrowser(self._settings, self.profile_dir_for(user_id))
                self._browsers[user_id] = browser
            return browser

    def __len__(self) -> int:
        return len(self._browsers)

    async def aclose(self) -> None:
        async with self._lock:
            browsers = list(self._browsers.values())
            self._browsers.clear()
        for browser in browsers:
            await browser.aclose()
        _teardown_xvfb()


class BrowserZAIClient:
    """``SupportsZAIClient`` that streams completions through the webpage."""

    def __init__(
        self,
        settings: Settings,
        pool: BrowserPool,
        *,
        zai_jwt: str | None = None,
        zai_session_token: str | None = None,
    ):
        self.settings = settings
        self._pool = pool
        self._http = ZAIClient(
            settings, zai_jwt=zai_jwt, zai_session_token=zai_session_token
        )

    async def ensure_session(self, force_refresh: bool = False) -> SessionState:
        return await self._http.ensure_session(force_refresh=force_refresh)

    async def verify_completion_version(self) -> int:
        return await self._http.verify_completion_version()

    async def aclose(self) -> None:
        # Account browsers outlive per-request clients; only the HTTP
        # session helper is closed here.
        await self._http.aclose()

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

    async def stream_prompt(
        self,
        *,
        prompt: str,
        model: str,
        enable_thinking: bool,
        auto_web_search: bool,
    ) -> AsyncIterator[UpstreamChunk]:
        label = UPSTREAM_MODEL_LABELS.get(model)
        if label is None:
            raise RuntimeError(f"Browser transport has no UI label for model {model!r}")
        session = await self.ensure_session()
        browser = await self._pool.get(session.user_id)

        async with browser.lock():
            page = await browser.ensure_page(session.token)
            await self._new_chat(page)
            await self._select_model(page, label)
            await page.evaluate(
                "window.__zaiReqConfig = {enable_thinking: %s, auto_web_search: %s};"
                "window.__zaiChunks = []; window.__zaiDone = false;"
                % (json.dumps(bool(enable_thinking)), json.dumps(bool(auto_web_search)))
            )
            composer = page.locator("textarea").first
            await composer.click()
            await composer.press_sequentially(prompt, delay=10)
            await page.wait_for_timeout(500)
            await page.locator("#send-message-button").click()

            buf = ""
            processed = 0
            start = time.monotonic()
            idle_since = time.monotonic()
            timeout = self.settings.request_timeout
            while True:
                await asyncio.sleep(0.4)
                state = await page.evaluate(
                    "() => ({chunks: window.__zaiChunks.join(''), done: window.__zaiDone})"
                )
                text = state["chunks"]
                if len(text) > processed:
                    buf += text[processed:]
                    processed = len(text)
                    idle_since = time.monotonic()
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:
                    done, chunk = parse_sse_line(line)
                    if done:
                        return
                    if chunk is None:
                        continue
                    yield chunk
                    if chunk.error:
                        return
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    raise RuntimeError("Browser completion timed out")
                if state["done"] and not buf.strip() and time.monotonic() - idle_since > 3:
                    return
                if not state["chunks"] and elapsed > 30:
                    raise RuntimeError(
                        "No completion stream captured; the page may have shown a CAPTCHA"
                    )

    async def _new_chat(self, page: Any) -> None:
        await page.goto(
            self.settings.zai_base_url + "/", wait_until="domcontentloaded"
        )
        await page.wait_for_timeout(2500)
        await page.keyboard.press("Escape")
        await page.evaluate(
            """() => {
                const el = [...document.querySelectorAll('[role="button"]')]
                    .find(e => (e.innerText || '').trim() === '新聊天');
                if (el) el.click();
            }"""
        )
        await page.wait_for_timeout(1500)

    async def _select_model(self, page: Any, label: str) -> None:
        state = await page.evaluate(
            """(label) => {
                const cur = document.querySelector('[id^="model-selector-"]');
                if (cur && (cur.innerText || '').includes(label)) return 'already';
                if (cur) cur.click();
                return 'opened';
            }""",
            label,
        )
        if state == "already":
            return
        # NOTE: label is bound via the evaluate argument below; keep the
        # two-step flow so we can wait for the dropdown to render.
        await page.wait_for_timeout(800)
        picked = await page.evaluate(
            """(label) => {
                const els = [...document.querySelectorAll(
                    'button, [role="option"], [role="menuitem"], li')];
                const el = els.find(e => (e.innerText || '').trim() === label
                    || (e.innerText || '').trim().startsWith(label));
                if (el) { el.click(); return true; }
                return false;
            }""",
            label,
        )
        if not picked:
            # Fallback: close the dropdown; the default model stays selected.
            await page.keyboard.press("Escape")
        await page.wait_for_timeout(800)


def make_browser_client_factory(
    settings: Settings, pool: BrowserPool
) -> Callable[[str | None, str | None], BrowserZAIClient]:
    def factory(
        zai_jwt: str | None, zai_session_token: str | None
    ) -> BrowserZAIClient:
        return BrowserZAIClient(
            settings, pool, zai_jwt=zai_jwt, zai_session_token=zai_session_token
        )

    return factory
