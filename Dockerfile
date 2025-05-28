# .dockerignore - Обновленный файл для исключения ненужных файлов из контекста сборки

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/
.env*
!.env.example

# Тесты - не нужны в production
tests/
pytest.ini
.pytest_cache/
.coverage
htmlcov/
*.cover

# Разработка и документация
*.md
!README.md
docs/
*.sh
docker-compose*.yml
Dockerfile*
.dockerignore
.gitignore
.git/
.github/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Данные и примеры
*.csv
*.xlsx
*.xls
*.db
*.sqlite3
data/
example*

# Логи
logs/
*.log

# Redis
dump.rdb

# Frontend (если есть)
frontend/
node_modules/

# Временные файлы
tmp/
temp/

# CI/CD
railway.json
runtime.txt
Procfile

# Nginx конфигурация (будет монтироваться как volume)
nginx/

# Conda
.conda/

---

# Стадия 1: Сборка зависимостей
FROM python:3.10-slim as builder

WORKDIR /build

# Установка только необходимых системных пакетов для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем только файл requirements
COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем wheel для более быстрой установки
RUN pip install --upgrade pip wheel

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем Prophet отдельно (он требует особого внимания)
RUN pip install --no-cache-dir prophet==1.1.5

# Стадия 2: Финальный образ
FROM python:3.10-slim

WORKDIR /app

# Установка только runtime зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

# Копируем виртуальное окружение из builder стадии
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем только необходимые файлы приложения
COPY app app/
COPY alembic alembic/
COPY alembic.ini .

# Создаем скрипт entrypoint
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
while ! nc -z $POSTGRES_SERVER $POSTGRES_PORT; do
  sleep 0.1
done
echo "PostgreSQL started"

echo "Waiting for Redis..."
while ! nc -z $REDIS_HOST $REDIS_PORT; do
  sleep 0.1
done
echo "Redis started"

echo "Running migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"
EOF

RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]