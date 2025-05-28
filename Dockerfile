FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей включая netcat
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libzbar0 \
    libzbar-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    netcat-traditional \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Устанавливаем prophet альтернативным способом (без conda)
RUN pip install --no-cache-dir prophet==1.1.5 || \
    echo "Prophet installation failed, using fallback"

# Копируем код приложения
COPY . .

# Создаем улучшенный скрипт entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Функция для проверки доступности сервиса\n\
wait_for_service() {\n\
    local host=$1\n\
    local port=$2\n\
    local service_name=$3\n\
    local max_attempts=30\n\
    local attempt=1\n\
    \n\
    echo "Waiting for $service_name at $host:$port..."\n\
    \n\
    while [ $attempt -le $max_attempts ]; do\n\
        if nc -z "$host" "$port" 2>/dev/null; then\n\
            echo "$service_name is available!"\n\
            return 0\n\
        fi\n\
        echo "Attempt $attempt/$max_attempts: $service_name is not ready yet..."\n\
        sleep 2\n\
        attempt=$((attempt + 1))\n\
    done\n\
    \n\
    echo "ERROR: $service_name failed to become available after $max_attempts attempts"\n\
    return 1\n\
}\n\
\n\
# Ждем инициализации приватной сети Railway\n\
if [ ! -z "$RAILWAY_ENVIRONMENT" ]; then\n\
    echo "Running on Railway, waiting for private network initialization..."\n\
    sleep 5\n\
fi\n\
\n\
# Определяем хосты и порты\n\
DB_HOST="${POSTGRES_SERVER:-localhost}"\n\
DB_PORT="${POSTGRES_PORT:-5432}"\n\
REDIS_HOST_VAR="${REDIS_HOST:-localhost}"\n\
REDIS_PORT_VAR="${REDIS_PORT:-6379}"\n\
\n\
# Проверяем PostgreSQL\n\
if ! wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL"; then\n\
    echo "WARNING: Could not connect to PostgreSQL, but continuing..."\n\
fi\n\
\n\
# Проверяем Redis\n\
if ! wait_for_service "$REDIS_HOST_VAR" "$REDIS_PORT_VAR" "Redis"; then\n\
    echo "WARNING: Could not connect to Redis, but continuing..."\n\
fi\n\
\n\
# Запускаем миграции\n\
echo "Running database migrations..."\n\
if [ ! -z "$DATABASE_URL" ]; then\n\
    echo "Using DATABASE_URL for migrations"\n\
    alembic upgrade head || echo "WARNING: Migration failed, but continuing..."\n\
else\n\
    echo "DATABASE_URL not set, attempting migration with individual variables"\n\
    alembic upgrade head || echo "WARNING: Migration failed, but continuing..."\n\
fi\n\
\n\
# Запускаем приложение\n\
echo "Starting application..."\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Создаем healthcheck скрипт
RUN echo '#!/bin/bash\n\
curl -f http://localhost:${PORT:-8000}/api/health || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Настройка healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/healthcheck.sh"]

# Используем переменную PORT от Railway
ENV PORT=8000

EXPOSE $PORT

ENTRYPOINT ["/entrypoint.sh"]

# Команда запуска с поддержкой переменной PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]