from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from fraud_detection.artifacts import ModelBundle
from fraud_detection.settings import settings

logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud_detection.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = ModelBundle(settings.model_dir)
    app.state.redis = None
    if settings.redis_url:
        try:
            import redis.asyncio as redis

            client = redis.from_url(
                settings.redis_url,
                socket_connect_timeout=settings.redis_timeout_seconds,
                socket_timeout=settings.redis_timeout_seconds,
                decode_responses=True,
            )
            await asyncio.wait_for(client.ping(), timeout=settings.redis_timeout_seconds)
            app.state.redis = client
            logger.info("Redis cache enabled")
        except Exception as exc:  # Redis is an optimization; inference remains available without it.
            logger.warning("Redis unavailable; serving without cache (%s)", type(exc).__name__)
    yield
    if app.state.redis is not None:
        await app.state.redis.aclose()


app = FastAPI(title="Financial Anomaly Detection API", version="0.1.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction: dict[str, Any] = Field(min_length=1)

    @field_validator("transaction")
    @classmethod
    def transaction_values_must_be_scalars(cls, values: dict[str, Any]) -> dict[str, Any]:
        for name, value in values.items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"Feature {name!r} must be a scalar string, number, boolean, or null")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Feature {name!r} must be finite")
        return values


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    bundle: ModelBundle | None = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready", "model_version": bundle.model_version}


@app.post("/v1/predict")
async def predict(request: PredictionRequest) -> dict[str, Any]:
    bundle: ModelBundle = app.state.bundle
    row = request.transaction
    cache = getattr(app.state, "redis", None)
    cache_key = "fraud:v1:" + hashlib.sha256(
        (bundle.model_version + json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)).encode()
    ).hexdigest()
    if cache is not None:
        try:
            cached = await asyncio.wait_for(cache.get(cache_key), timeout=settings.redis_timeout_seconds)
            if cached is not None:
                result = json.loads(cached)
                result["cache_hit"] = True
                result["inference_ms"] = 0.0
                return result
        except Exception as exc:
            logger.debug("Redis read failed (%s)", type(exc).__name__)

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(bundle.predict, row)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    result["inference_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["cache_hit"] = False
    if cache is not None and settings.redis_ttl_seconds > 0:
        try:
            await asyncio.wait_for(
                cache.set(cache_key, json.dumps(result, allow_nan=False), ex=settings.redis_ttl_seconds),
                timeout=settings.redis_timeout_seconds,
            )
        except Exception as exc:
            logger.debug("Redis write failed (%s)", type(exc).__name__)
    return result
