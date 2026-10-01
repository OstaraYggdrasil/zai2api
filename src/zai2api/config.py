from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ENV_FILE = ".env"
DEFAULT_PANEL_PASSWORD = "123456"
DEFAULT_ADMIN_COOKIE_NAME = "zai2api_admin_session"
DEFAULT_DATABASE_PATH = "data/zai2api.db"
DEFAULT_LOG_RETENTION_DAYS = 7


def _load_dotenv_file(path: str = ENV_FILE) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv_file()


@dataclass(slots=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    zai_base_url: str = os.getenv("ZAI_BASE_URL", "https://chat.z.ai")
    zai_jwt: str | None = os.getenv("ZAI_JWT")
    zai_session_token: str | None = os.getenv("ZAI_SESSION_TOKEN")
    default_model: str = os.getenv("DEFAULT_MODEL", "glm-5.3")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "120"))
    # Transport selection: "browser" (default) drives the real webpage through
    # a headed Chromium, which passes the completion CAPTCHA transparently;
    # "http" uses raw HTTP (currently rejected upstream:
    # FRONTEND_CAPTCHA_REQUIRED / missing captcha_verify_param).
    transport: str = os.getenv("ZAI_TRANSPORT", "browser")
    browser_profile_dir: str = os.getenv(
        "BROWSER_PROFILE_DIR", os.path.join("data", "browser-profile")
    )
    browser_proxy: str | None = os.getenv("BROWSER_PROXY")
    database_path: str = os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH)
    panel_password_env: str | None = os.getenv("PANEL_PASSWORD") or os.getenv("ADMIN_PASSWORD")
    api_password_env: str | None = os.getenv("API_PASSWORD")
    admin_cookie_name: str = os.getenv("ADMIN_COOKIE_NAME", DEFAULT_ADMIN_COOKIE_NAME)
    admin_session_ttl_hours: int = int(os.getenv("ADMIN_SESSION_TTL_HOURS", "168"))
    admin_cookie_secure: bool = os.getenv("ADMIN_COOKIE_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    account_poll_interval_seconds: int = int(os.getenv("ACCOUNT_POLL_INTERVAL_SECONDS", "300"))
    log_retention_days_env: int | None = (
        int(value) if (value := os.getenv("LOG_RETENTION_DAYS")) is not None else None
    )

    @property
    def admin_session_ttl_seconds(self) -> int:
        return self.admin_session_ttl_hours * 3600


settings = Settings()
