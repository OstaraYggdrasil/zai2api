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


class SharedBrowser:
    """Owns one persistent headed Chromium used by all browser clients."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = asyncio.Lock()
        self._pw: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._xvfb_proc: subprocess.Popen[bytes] | None = None
        self._started = False

    def lock(self) -> asyncio.Lock:
        return self._lock

    async def ensure_page(self, token: str) -> Any:
        """Start the browser on first use and return the shared page."""
        if self._started and self._page is not None:
            try:
                await self._page.evaluate(
                    f"localStorage.setItem('token', {json.dumps(token)})"
                )
            except Exception:
                pass
            return self._page

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise RuntimeError(
                "The browser transport needs the 'playwright' package "
                "(pip install playwright) and a Chromium build."
            ) from exc

        await self._ensure_xvfb()
        self._pw = await async_playwright().start()
        proxy = self._proxy_config()
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": self._settings.browser_profile_dir,
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
            f"localStorage.setItem('token', {json.dumps(token)});\n" + FETCH_HOOK_JS
        )
        self._page = await self._ctx.new_page()
        await self._page.goto(
            self._settings.zai_base_url + "/", wait_until="domcontentloaded"
        )
        await self._page.wait_for_timeout(4000)
        await self._page.keyboard.press("Escape")
        self._started = True
        return self._page

    async def _ensure_xvfb(self) -> None:
        if os.environ.get("DISPLAY"):
            return
        try:
            self._xvfb_proc = subprocess.Popen(
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

    async def aclose(self) -> None:
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
        if self._xvfb_proc is not None:
            try:
                self._xvfb_proc.terminate()
            except Exception:
                pass
            self._xvfb_proc = None
        self._started = False


class BrowserZAIClient:
    """``SupportsZAIClient`` that streams completions through the webpage."""

    def __init__(
        self,
        settings: Settings,
        shared: SharedBrowser,
        *,
        zai_jwt: str | None = None,
        zai_session_token: str | None = None,
    ):
        self.settings = settings
        self._shared = shared
        self._http = ZAIClient(
            settings, zai_jwt=zai_jwt, zai_session_token=zai_session_token
        )

    async def ensure_session(self, force_refresh: bool = False) -> SessionState:
        return await self._http.ensure_session(force_refresh=force_refresh)

    async def verify_completion_version(self) -> int:
        return await self._http.verify_completion_version()

    async def aclose(self) -> None:
        # The shared browser outlives per-request clients.
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

        async with self._shared.lock():
            page = await self._shared.ensure_page(session.token)
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
    settings: Settings, shared: SharedBrowser
) -> Callable[[str | None, str | None], BrowserZAIClient]:
    def factory(
        zai_jwt: str | None, zai_session_token: str | None
    ) -> BrowserZAIClient:
        return BrowserZAIClient(
            settings, shared, zai_jwt=zai_jwt, zai_session_token=zai_session_token
        )

    return factory
