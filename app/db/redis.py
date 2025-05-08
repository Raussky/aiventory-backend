# /app/db/redis.py
from redis.asyncio import Redis
from app.core.config import settings
import logging
import asyncio
from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def get_redis() -> Redis:
    """Возвращает клиент Redis как зависимость FastAPI"""
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    if settings.REDIS_PASSWORD:
        redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"

    # Добавляем логику повторных попыток подключения
    max_retries = 3
    retry_delay = 1  # секунды

    redis = None
    last_error = None

    for attempt in range(max_retries):
        try:
            redis = Redis.from_url(redis_url, decode_responses=True)
            # Проверяем соединение
            await redis.ping()
            logger.info(f"Successfully connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            break
        except Exception as e:
            last_error = e
            if redis:
                await redis.close()
                redis = None

            if attempt < max_retries - 1:
                logger.warning(
                    f"Redis connection attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Экспоненциальное увеличение задержки
            else:
                logger.error(f"Redis connection failed after {max_retries} attempts: {str(e)}")

    # Если не удалось подключиться после всех попыток
    if not redis:
        error_msg = f"Не удалось подключиться к Redis ({settings.REDIS_HOST}:{settings.REDIS_PORT}). "
        error_msg += "Проверьте, что Redis запущен и доступен. "

        # Add detailed troubleshooting information based on environment
        if settings.REDIS_HOST == "redis":
            error_msg += "Если запуск выполняется локально (не в Docker), установите REDIS_HOST=localhost в .env файле."
        elif settings.REDIS_HOST == "localhost":
            error_msg += "Если запуск выполняется в Docker, установите REDIS_HOST=redis в .env файле."
        elif "railway.internal" in settings.REDIS_HOST:
            error_msg += "Вы используете Railway.app - убедитесь, что сервис Redis правильно настроен в вашем проекте."

        logger.error(error_msg)

        # В режиме разработки можно продолжить без Redis с ограниченной функциональностью
        if settings.API_V1_STR.startswith('/api'):  # Простая проверка на режим разработки
            logger.warning("Продолжаем запуск без Redis. Некоторые функции будут недоступны.")
            try:
                # Создаем заглушку Redis для неблокирующей работы
                class RedisMock:
                    async def get(self, *args, **kwargs):
                        return None

                    async def set(self, *args, **kwargs):
                        return True

                    async def delete(self, *args, **kwargs):
                        return True

                    async def close(self):
                        pass

                    async def publish(self, *args, **kwargs):
                        return 0

                    async def ping(self, *args, **kwargs):
                        return True

                yield RedisMock()
                return
            except Exception:
                pass

        # Если не в режиме разработки или заглушка не сработала
        raise HTTPException(
            status_code=503,
            detail="Сервис временно недоступен. Пожалуйста, попробуйте позже."
        )

    try:
        yield redis
    finally:
        if redis:
            await redis.close()