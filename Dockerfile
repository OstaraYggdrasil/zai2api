FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /uvx /bin/

WORKDIR /app

COPY --from=builder /app /app

# Browser transport needs a real Chromium plus its OS dependencies and a
# virtual display (the app auto-starts Xvfb when no DISPLAY is present).
RUN mkdir -p /app/data \
    && apt-get update \
    && /app/.venv/bin/playwright install --with-deps chromium \
    && apt-get install -y --no-install-recommends xvfb \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
VOLUME ["/app/data"]

CMD ["uv", "run", "zai2api"]
