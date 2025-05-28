FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
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

# Создаем скрипт entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Waiting for PostgreSQL..."\n\
while ! nc -z $POSTGRES_SERVER $POSTGRES_PORT; do\n\
  sleep 0.1\n\
done\n\
echo "PostgreSQL started"\n\
\n\
echo "Waiting for Redis..."\n\
while ! nc -z $REDIS_HOST $REDIS_PORT; do\n\
  sleep 0.1\n\
done\n\
echo "Redis started"\n\
\n\
echo "Running migrations..."\n\
alembic upgrade head\n\
\n\
echo "Starting application..."\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Устанавливаем netcat для проверки доступности сервисов
RUN apt-get update && apt-get install -y netcat-traditional && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]