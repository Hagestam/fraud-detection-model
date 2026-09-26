from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", "artifacts/current"))
    redis_url: str | None = os.getenv("REDIS_URL")
    redis_ttl_seconds: int = int(os.getenv("REDIS_TTL_SECONDS", "60"))
    redis_timeout_seconds: float = float(os.getenv("REDIS_TIMEOUT_SECONDS", "0.05"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
