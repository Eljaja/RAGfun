from __future__ import annotations

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    # Service
    service_name: str = "document-storage"
    environment: str = "dev"
    log_level: str = "INFO"

    # Storage Backend
    storage_backend: str = "local"  # local|s3
    storage_path: str = "/data/documents"  # for local storage

    # S3-compatible storage (MinIO)
    s3_endpoint: AnyHttpUrl | None = None
    s3_bucket: str = "documents"
    s3_access_key: SecretStr | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str = "us-east-1"

    # Database (PostgreSQL for metadata)
    db_url: str = "postgresql://postgres:postgres@localhost:5432/document_storage"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # Limits
    max_file_size_mb: int = 100
    allowed_content_types: str = "text/plain,text/markdown,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    # ODS admin + tenant API keys (optional; used for \"full ODS\" tenant enforcement in Gate)
    # When admin_secret is set, the admin endpoints (/v1/tenants, /v1/tenants/{id}/api-keys) require it.
    ods_admin_secret: SecretStr | None = None
    # Salt is used to hash API keys before storing them in Postgres (sha256(salt + key)).
    # If empty/unset, hashing still works but key rotation becomes riskier; strongly prefer setting.
    ods_api_key_salt: SecretStr | None = None

    def safe_summary(self) -> dict:
        """Configuration summary without secrets."""
        return {
            "service": {
                "name": self.service_name,
                "environment": self.environment,
            },
            "storage": {
                "backend": self.storage_backend,
                "path": self.storage_path if self.storage_backend == "local" else None,
                "s3_endpoint": str(self.s3_endpoint) if self.s3_endpoint else None,
                "s3_bucket": self.s3_bucket if self.storage_backend == "s3" else None,
                "s3_region": self.s3_region if self.storage_backend == "s3" else None,
            },
            "database": {
                "url_set": bool(self.db_url),
                "pool_min": self.db_pool_min,
                "pool_max": self.db_pool_max,
            },
            "limits": {
                "max_file_size_mb": self.max_file_size_mb,
                "allowed_content_types": self.allowed_content_types.split(","),
            },
            "ods": {
                "admin_secret_set": self.ods_admin_secret is not None,
                "api_key_salt_set": self.ods_api_key_salt is not None,
            },
        }


    @property
    def allowed_content_types_list(self) -> list[str]:
        return [ct.strip() for ct in self.allowed_content_types.split(",")]


def load_settings() -> Settings:
    return Settings()
