# app/db/redis.py
from redis.asyncio import Redis  # Используем правильный импорт для нового Redis
from app.core.config import settings


async def get_redis() -> Redis:
    """Возвращает клиент Redis как зависимость FastAPI"""
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    if settings.REDIS_PASSWORD:
        redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"

    redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        yield redis
    finally:
        await redis.close()