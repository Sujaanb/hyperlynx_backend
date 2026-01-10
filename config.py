from typing import List
import json
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/HL/content/v1"

    # Default allowed origins for CORS - add your production domain(s) here
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
        "http://localhost:3001",
        "http://localhost:5173",
        "https://localhost:5173",
        "https://hyperlynx-frontend.vercel.app",
        "https://hyperlynx-frontend-git-main-sujaanb.vercel.app",
        "https://hyperlynx.ai",
        "https://www.hyperlynx.ai",
    ]

    PROJECT_NAME: str = "Hyperlynx Platform APIs"

    class Config:
        case_sensitive = True


settings = Settings()

# If an operator set BACKEND_CORS_ORIGINS in the environment, pydantic may
# provide it as a string. Support common formats: a JSON list or a
# comma-separated string of origins.
raw = settings.BACKEND_CORS_ORIGINS
if isinstance(raw, str):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            settings.BACKEND_CORS_ORIGINS = parsed
        else:
            # Fallback to splitting on commas
            settings.BACKEND_CORS_ORIGINS = [s.strip() for s in raw.split(",") if s.strip()]
    except Exception:
        settings.BACKEND_CORS_ORIGINS = [s.strip() for s in raw.split(",") if s.strip()]