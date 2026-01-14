# settings.py
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load from environment-specific file
        env_file=f"../.env.{os.getenv('ENV', 'local')}",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    aws_region: str = "eu-central-1"
    aws_access_key_id: str = "rustfs"
    aws_secret_access_key: str = "password"
    aws_session_token: str | None = None
    s3_endpoint_url: str = "http://localhost:9004"
    queue_arn: str = "arn:rustfs:sqs:us-east-1:webhook:webhook"