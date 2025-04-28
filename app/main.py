# app/main.py
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from loguru import logger
import uuid
import aioredis
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import Response

from app.api.v1 import auth, warehouse, store, prediction
from app.core.config import settings
from app.db.redis import get_redis

# Метрики Prometheus
REQUEST_COUNT = Counter(
    "app_request_count",
    "Application Request Count",
    ["app_name", "method", "endpoint", "http_status"]
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Application Request Latency",
    ["app_name", "method", "endpoint"]
)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Inventory Management System API",
    description="API для управления запасами малого бизнеса",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production изменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] {request.method} {request.url.path}")

    start_time = time.time()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Обновляем метрики Prometheus
        REQUEST_LATENCY.labels(
            "inventory-system",
            request.method,
            request.url.path
        ).observe(process_time)

        REQUEST_COUNT.labels(
            "inventory-system",
            request.method,
            request.url.path,
            response.status_code
        ).inc()

        logger.info(f"[{request_id}] Completed {response.status_code} in {process_time:.4f}s")

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.exception(f"[{request_id}] Failed in {process_time:.4f}s: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"}
        )


# Подключение роутеров API
app.include_router(
    auth.router,
    prefix=f"{settings.API_V1_STR}/auth",
    tags=["authentication"]
)

app.include_router(
    warehouse.router,
    prefix=f"{settings.API_V1_STR}/warehouse",
    tags=["warehouse"]
)

app.include_router(
    store.router,
    prefix=f"{settings.API_V1_STR}/store",
    tags=["store"]
)

app.include_router(
    prediction.router,
    prefix=f"{settings.API_V1_STR}/prediction",
    tags=["prediction"]
)


# Эндпоинт для проверки состояния приложения
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


# Эндпоинт для метрик Prometheus
@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.on_event("startup")
async def startup_db_client():
    logger.info(f"Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    if settings.REDIS_PASSWORD:
        redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"

    try:
        app.state.redis = await aioredis.from_url(
            redis_url,
            decode_responses=True
        )
        # Проверяем соединение
        await app.state.redis.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        logger.error("Make sure Redis is running and accessible:")
        logger.error(f"- Host: {settings.REDIS_HOST}")
        logger.error(f"- Port: {settings.REDIS_PORT}")
        logger.error("If running locally (not in Docker), set REDIS_HOST=localhost in .env file")

        # Продолжаем запуск без Redis в режиме разработки
        if app.debug:
            logger.warning("Continuing startup without Redis in development mode")
            app.state.redis = None
        else:
            # В продакшене лучше завершить запуск с ошибкой
            raise e


@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Closing Redis connection...")
    await app.state.redis.close()
    logger.info("Redis connection closed")