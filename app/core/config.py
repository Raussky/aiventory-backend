from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    DATABASE_URL: Optional[str] = None

    POSTGRES_SERVER: Optional[str] = None
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    REDIS_HOST: str
    REDIS_PORT: str = "6379"
    REDIS_PASSWORD: Optional[str] = None

    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[str] = None
    EMAILS_FROM_NAME: Optional[str] = None

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_DEFAULT_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.DATABASE_URL:
            self.SQLALCHEMY_DATABASE_URI = self.DATABASE_URL
        else:
            self.SQLALCHEMY_DATABASE_URI = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = "AIventory System"


settings = Settings()