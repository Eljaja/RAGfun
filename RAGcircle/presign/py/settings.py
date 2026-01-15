# settings.py
from pydantic import Field
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

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load from environment-specific file
        env_file=f"../.env.{os.getenv('ENV', 'local')}",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    aws_region: str = "us-east-1"
    aws_access_key_id: str = "rustfs"
    aws_secret_access_key: str = "password"
    aws_session_token: str | None = None
    s3_endpoint_url: str = "http://localhost:9004"
    # TODO: based on the .env for RUSTFS get the queue arn instead of passing 
    # the actual values
    queue_arn: str = "arn:rustfs:sqs:us-east-1:primary:mqtt"
    