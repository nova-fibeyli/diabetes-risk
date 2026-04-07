from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Diabetes Risk Support API"
    app_version: str = "2.0.0"
    environment: str = "development"

    backend_base_url: str = "http://127.0.0.1:8005"
    frontend_url: str = "http://127.0.0.1:5173"
    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )
    admin_emails: str = ""

    database_url: str = "sqlite+aiosqlite:///./diabetes_app.db"
    auth_cookie_name: str = "diabetes_risk_session"
    jwt_secret_key: str = "change-me-jwt-secret"
    session_secret_key: str = "change-me-session-secret"
    auth_token_expiry_days: int = 7
    auth_cookie_secure: bool = False

    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_path: str = "/auth/google/callback"

    model_data_url: str = (
        "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
    )
    model_target_column: str = "Outcome"
    model_name: str = "LogisticRegression"

    data_dir: Path = Path("./data")
    model_dir: Path = Path("./models")
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def google_redirect_uri(self) -> str:
        return f"{self.backend_base_url.rstrip('/')}{self.google_redirect_path}"

    @property
    def parsed_cors_origins(self) -> list[str]:
        if not self.cors_origins:
            return []
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def parsed_admin_emails(self) -> set[str]:
        if not self.admin_emails:
            return set()
        return {item.strip().lower() for item in self.admin_emails.split(",") if item.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
