# app/tasks/celery_app.py
from celery import Celery
from app.core.config import settings

redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
if settings.REDIS_PASSWORD:
    redis_url = f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url,
)

celery_app.conf.task_routes = {
    "app.tasks.notifications.*": {"queue": "notifications"},
    "app.tasks.predictions.*": {"queue": "predictions"},
}

celery_app.conf.timezone = 'UTC'