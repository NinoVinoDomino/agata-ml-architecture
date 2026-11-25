from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGATA_", env_file=".env", extra="ignore")

    timezone: str = "Europe/Moscow"
    release_id: str = "template-v1"
    edition_dir: Path = Path("data/daily")
    s3_bucket: str | None = None

    generator: Literal["template", "grpc", "openai_compatible"] = "template"
    generation_concurrency: int = Field(default=4, ge=1, le=48)
    model_timeout: float = Field(default=10, gt=0)
    model_grpc_target: str | None = None
    model_grpc_tls: bool = True
    model_endpoint: AnyHttpUrl | None = None
    model_id: str | None = None
    model_api_key: SecretStr | None = None

    telegram_bot_token: SecretStr | None = None
    telegram_webhook_secret: SecretStr | None = None
    telegram_secret_arn: str | None = None

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be an IANA timezone name") from exc
        return value

    @model_validator(mode="after")
    def model_settings_required(self) -> "Settings":
        if self.generator == "openai_compatible" and not (self.model_endpoint and self.model_id):
            raise ValueError("the OpenAI-compatible generator requires MODEL_ENDPOINT and MODEL_ID")
        if self.generator == "grpc" and not (self.model_grpc_target and self.model_id):
            raise ValueError("the gRPC generator requires MODEL_GRPC_TARGET and MODEL_ID")
        if self.generator == "grpc" and self.model_api_key and not self.model_grpc_tls:
            raise ValueError("the gRPC generator requires TLS when using an API key")
        return self
