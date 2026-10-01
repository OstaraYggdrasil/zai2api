# zai2api

OpenAI-compatible chat/completion proxy backed by `https://chat.z.ai/`.

> **How the CAPTCHA is handled (2026-10-01):** chat.z.ai requires an
> Alibaba Cloud slider CAPTCHA (`captcha_verify_param`) before **every**
> chat completion (`enable_captcha: true` in `/api/config`; the server
> rejects requests without it: `FRONTEND_CAPTCHA_REQUIRED` /
> `missing_param`). Raw HTTP cannot obtain this token, so the default
> transport (`ZAI_TRANSPORT=browser`) drives the real webpage through a
> headed Chromium with a persistent profile: the site's own JavaScript
> mints the token transparently, exactly like a normal browser session.
> The page's `/api/v2/chat/completions` SSE stream is captured via an
> injected `fetch` hook, so streaming, reasoning output and usage all work
> end to end. Set `ZAI_TRANSPORT=http` to use the legacy raw-HTTP
> transport (currently rejected upstream).
>
> **Multiple accounts:** register extra accounts in the admin panel (or via
> the accounts API) and requests round-robin across them. With the browser
> transport each account gets its **own** headed Chromium and its **own**
> persistent profile directory (`<BROWSER_PROFILE_DIR>/profile-<user-id>`),
> so accounts never share storage, cookies or login tokens. Requests for
> the same account are serialized through its browser; different accounts
> run in parallel.

## Features

- Supports `POST /v1/chat/completions`
- Supports `POST /v1/responses`
- Creates a fresh upstream chat for every request
- Preserves reasoning output separately from final answer text
- Reuses `ZAI_SESSION_TOKEN` directly or refreshes it from `ZAI_JWT`
- Browser transport passes the completion CAPTCHA with zero interaction

## Requirements

- Python 3.12+
- `uv`
- Playwright + a Chromium build (`pip install playwright && playwright install chromium`;
  the browser transport launches it headed, using Xvfb automatically when no
  display is present)
- One of:
  - `ZAI_JWT`
  - `ZAI_SESSION_TOKEN`

## Run

```bash
export ZAI_JWT='your-jwt'
uv run python -m zai2api
```

Or with the installed script:

```bash
export ZAI_JWT='your-jwt'
uv run zai2api
```

Default bind address is `0.0.0.0:8000`.

## Environment variables

- `ZAI_JWT`: preferred auth source; used to fetch a fresh session token
- `ZAI_SESSION_TOKEN`: optional direct session token reuse
- `DEFAULT_MODEL`: defaults to `glm-5.3`
- Available public model ids (synced with the chat.z.ai webpage model selector, 2026-10-01): `glm-5.3`, `glm-5.3-flash`, `glm-5.2`
- A `-nothinking` variant is offered for `glm-5.2` (the only webpage model whose upstream capabilities allow disabling thinking; the webpage locks deep thinking ON for GLM-5.3 / GLM-5.3-Flash)
- Legacy ids `glm-5` and `glm-5.1` still work and map to `glm-5.3` / `glm-5.2`
- `HOST`: defaults to `0.0.0.0`
- `PORT`: defaults to `8000`
- `LOG_LEVEL`: defaults to `info`
- `REQUEST_TIMEOUT`: defaults to `120`
- `ZAI_TRANSPORT`: `browser` (default) or `http` — see the note at the top
- `BROWSER_PROFILE_DIR`: base directory for persistent Chromium profiles, defaults to `data/browser-profile`. With the browser transport each account gets its own subdirectory `profile-<user-id>` underneath it — keep this on persistent disk, not `/tmp`.
- `BROWSER_PROXY`: proxy URL for the browser (e.g. `http://127.0.0.1:8899`); when unset, a local forward proxy on `127.0.0.1:8899` is auto-detected, otherwise the browser goes direct

## Example requests

### Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "glm-5.3",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "Say hello."}
    ]
  }'
```

### Responses API

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model": "glm-5",
    "input": "Say hello."
  }'
```
