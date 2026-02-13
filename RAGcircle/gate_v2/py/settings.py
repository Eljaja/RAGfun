# settings.py
# from RAGfun.RAGcircle.presign.py.presign_main import opensearch, qdrant
from pydantic_settings import BaseSettings, SettingsConfigDict
import os





class Constants(BaseSettings):
    model_config = SettingsConfigDict(
        # Load from environment-specific file
        env_file=f"../.env.{os.getenv('ENV', 'local')}",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    max_file_size: int = 100 * 1024
    max_link_ttl: int = 3600
    allowed_content_types: list[str] = [
        # Documents
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/rtf",
        "application/epub+zip",
    ]
    allowed_text_pattern: str = r"^(text\/|application\/(json|xml|javascript|typescript|python|x-python))"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load from environment-specific file
        env_file=f"../.env.{os.getenv('ENV', 'local')}",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Auth check 
    auth_server: str
    admin_secret_token: str

    aws_region: str = "us-east-1"
    aws_access_key_id: str = "rustfs"
    aws_secret_access_key: str = "password"
    aws_session_token: str | None = None
    s3_endpoint_url: str = "http://localhost:9004"
    # TODO: based on the .env for RUSTFS get the queue arn instead of passing 
    # the actual values
    queue_arn: str = "arn:rustfs:sqs:us-east-1:primary:mqtt"
    bucket_name: str = "ragfun"

    database_url: str = "postgresql://user:pass@localhost:5438/db"
    max_projects_per_user: int = 5

    qdrant_url: str = "http://localhost:8903"
    opensearch_url: str = "http://localhost:8905"


# can we automate this?
# 2026 and I am still not good at metaprogramming
def load_settings() -> Settings:
    return Settings()

def load_constants() -> Constants:
    return Constants()
