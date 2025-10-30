from pydantic_settings import BaseSettings
from typing import List, Optional
from decouple import config

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = config("PROJECT_NAME", default="HeartTells, FastAPI AI-assisted Decision Maker")
    API_V1_STR: str = config("API_V1_STR", default="/api/v1")
    SECRET_KEY: str = config("SECRET_KEY", default="change_to_secret_key")
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30, cast=int)
    REFRESH_TOKEN_EXPIRE_MINUTES: int = config("REFRESH_TOKEN_EXPIRE_MINUTES", default=60*24*7, cast=int)
    ENVIRONMENT: str = config("ENVIRONMENT", default="dev")

    # Database
    DATABASE_NAME: str = config("DATABASE_NAME", default="hearttells")
    DATABASE_USERNAME: str = config("DATABASE_USERNAME", default="postgres")
    DATABASE_PASSWORD: str = config("DATABASE_PASSWORD", default="")
    DATABASE_HOST: str = config("DATABASE_HOST", default="localhost")
    DATABASE_PORT: str = config("DATABASE_PORT", default="5432")
    DATABASE_URI: str = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000",
                               "http://localhost:3002",
                               "http://127.0.0.1:3000",
                               "http://127.0.0.1:3002",
                               "http://localhost:8080"]

    # Development
    DEBUG: bool = config("DEBUG", default=True, cast=bool)

    # ChatGPT API
    CHATGPT_API_KEY: str = config("CHATGPT_API_KEY", default="")

    # Google OAuth
    GOOGLE_CLIENT_ID: str = config("GOOGLE_CLIENT_ID", default="")
    GOOGLE_CLIENT_SECRET: str = config("GOOGLE_CLIENT_SECRET", default="")
    GOOGLE_REDIRECT_URI: str = config("GOOGLE_REDIRECT_URI", default="http://localhost:8000/auth/google/callback")

    # Apple Watch HealthKit
    # Apple Watch HealthKitElectrocardiogram

    # Local storage ECG?
    LOCAL_STORAGE_PATH: str = config("LOCAL_STORAGE_PATH", default="app/storage")

    # Embedding storage ?
    EMBEDDING_STORAGE_PATH: str = config("EMBEDDING_STORAGE_PATH", default="app/storage/embeddings")

    class Config:
        case_sensitive = True

# Create global settings instance
settings = Settings()