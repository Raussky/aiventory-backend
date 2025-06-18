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
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости Python (добавляем plotly)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir plotly==5.17.0

# Устанавливаем prophet
RUN pip install --no-cache-dir prophet==1.1.5 || \
    echo "Prophet installation failed, using fallback"

# Копируем код приложения
COPY . .

# Создаем улучшенный entrypoint для Railway
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Starting on Railway..."\n\
\n\
# Функция для безопасного применения миграций\n\
apply_migrations() {\n\
    echo "Checking database connection..."\n\
    if [ ! -z "$DATABASE_URL" ]; then\n\
        echo "DATABASE_URL is set, checking migrations..."\n\
        \n\
        # Проверяем, существует ли таблица alembic_version\n\
        if psql "$DATABASE_URL" -tc "SELECT 1 FROM information_schema.tables WHERE table_name = '"'"'alembic_version'"'"';" | grep -q 1; then\n\
            echo "Alembic version table exists, checking current revision..."\n\
            CURRENT_REV=$(alembic current 2>/dev/null | grep -oE '"'"'[a-f0-9]{12}'"'"' || echo "none")\n\
            echo "Current revision: $CURRENT_REV"\n\
            \n\
            # Проверяем, есть ли неприменённые миграции\n\
            if alembic history -r$CURRENT_REV:head 2>/dev/null | grep -q "Rev:"; then\n\
                echo "Found pending migrations, applying..."\n\
                alembic upgrade head || {\n\
                    echo "Migration failed, trying to fix..."\n\
                    # Если миграция не удалась, попробуем пометить текущее состояние\n\
                    alembic stamp head\n\
                    echo "Marked current state as head"\n\
                }\n\
            else\n\
                echo "No pending migrations"\n\
            fi\n\
        else\n\
            echo "Alembic version table does not exist, initializing..."\n\
            # Проверяем, есть ли уже таблицы в БД\n\
            TABLE_COUNT=$(psql "$DATABASE_URL" -tc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '"'"'public'"'"';" | tr -d '"'"' '"'"')\n\
            \n\
            if [ "$TABLE_COUNT" -gt 0 ]; then\n\
                echo "Database has existing tables, stamping current state..."\n\
                alembic stamp head\n\
            else\n\
                echo "Empty database, running all migrations..."\n\
                alembic upgrade head\n\
            fi\n\
        fi\n\
    else\n\
        echo "WARNING: DATABASE_URL not set"\n\
    fi\n\
}\n\
\n\
# Применяем миграции\n\
apply_migrations\n\
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