# Multi-stage build для оптимизации размера образа
FROM continuumio/miniconda3:latest AS builder

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libzbar0 \
    libzbar-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создание conda окружения
RUN conda create -n inventory python=3.10 -y

# Активация окружения и установка Prophet
SHELL ["/bin/bash", "-c"]
RUN source activate inventory && \
    conda install -c conda-forge prophet -y

# Копирование requirements
WORKDIR /app
COPY requirements.txt .

# Установка Python зависимостей
RUN source activate inventory && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install prophet

# Production образ
FROM continuumio/miniconda3:latest

# Установка системных зависимостей для runtime
RUN apt-get update && apt-get install -y \
    libpq5 \
    libzbar0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    netcat-openbsd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копирование conda окружения из builder
COPY --from=builder /opt/conda/envs/inventory /opt/conda/envs/inventory

# Создание рабочей директории
WORKDIR /app

# Копирование кода приложения
COPY . .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Установка переменных окружения
ENV PATH="/opt/conda/envs/inventory/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Создание entrypoint скрипта
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Ожидание доступности PostgreSQL\n\
echo "Waiting for PostgreSQL..."\n\
while ! nc -z $POSTGRES_SERVER $POSTGRES_PORT; do\n\
  sleep 1\n\
done\n\
echo "PostgreSQL is ready!"\n\
\n\
# Ожидание доступности Redis\n\
echo "Waiting for Redis..."\n\
while ! nc -z $REDIS_HOST $REDIS_PORT; do\n\
  sleep 1\n\
done\n\
echo "Redis is ready!"\n\
\n\
# Запуск миграций\n\
echo "Running database migrations..."\n\
alembic upgrade head\n\
\n\
# Запуск приложения\n\
echo "Starting application..."\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Expose порт
EXPOSE 8000

# Entrypoint и команда по умолчанию
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]