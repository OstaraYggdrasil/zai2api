# zai2api

OpenAI-compatible chat/completion proxy backed by `https://chat.z.ai/`.

> **Known limitation (2026-10-01):** chat.z.ai now requires an interactive
> Alibaba Cloud slider CAPTCHA (`captcha_verify_param`) before **every**
> chat completion (`enable_captcha: true` in `/api/config`; the server
> rejects requests without it: `FRONTEND_CAPTCHA_REQUIRED` /
> `missing_param`). A headless proxy cannot obtain this token without human
> interaction per message, so `/v1/chat/completions` and `/v1/responses`
> currently fail upstream even with a valid account. Everything else
> (auth, session refresh, chat creation, `/v1/models`, admin panel) works.

## Features

- Supports `POST /v1/chat/completions`
- Supports `POST /v1/responses`
- Creates a fresh upstream chat for every request
- Preserves reasoning output separately from final answer text
- Reuses `ZAI_SESSION_TOKEN` directly or refreshes it from `ZAI_JWT`

## Requirements

- Python 3.12+
- `uv`
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
