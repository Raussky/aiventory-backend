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

# Создаем простой entrypoint для публичных URL
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting on Railway..."\n\
\n\
# Проверяем наличие DATABASE_URL\n\
if [ ! -z "$DATABASE_URL" ]; then\n\
    echo "DATABASE_URL is set, running migrations..."\n\
    alembic upgrade head || echo "WARNING: Migration failed, but continuing..."\n\
else\n\
    echo "WARNING: DATABASE_URL not set"\n\
fi\n\
\n\
# Запускаем приложение\n\
echo "Starting application on port ${PORT:-8000}..."\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Используем переменную PORT от Railway
ENV PORT=8000

EXPOSE $PORT

ENTRYPOINT ["/entrypoint.sh"]

# Команда запуска с поддержкой переменной PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]