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

# Устанавливаем зависимости Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir plotly==5.17.0

# Устанавливаем prophet с обработкой ошибок
RUN pip install --no-cache-dir prophet==1.1.5 || \
    echo "Prophet installation failed, will use fallback methods"

# Копируем код приложения
COPY . .

# Создаем директорию для скриптов
RUN mkdir -p /scripts

# Создаем скрипт для ожидания базы данных
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
host="$1"\n\
port="$2"\n\
shift 2\n\
\n\
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$host" -p "$port" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\\q"; do\n\
  >&2 echo "Postgres is unavailable - sleeping"\n\
  sleep 1\n\
done\n\
\n\
>&2 echo "Postgres is up - executing command"\n\
exec "$@"' > /scripts/wait-for-postgres.sh && chmod +x /scripts/wait-for-postgres.sh

# Создаем скрипт для применения миграций
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "Checking database connection..."\n\
\n\
# Ждем доступности базы данных\n\
/scripts/wait-for-postgres.sh "$POSTGRES_SERVER" "$POSTGRES_PORT"\n\
\n\
echo "Database is ready, checking migrations..."\n\
\n\
# Проверяем текущее состояние миграций\n\
CURRENT_REV=$(alembic current 2>/dev/null | grep -oE "[a-f0-9]{12}" || echo "none")\n\
echo "Current revision: $CURRENT_REV"\n\
\n\
# Проверяем наличие колонки created_at в prediction\n\
echo "Checking for missing columns..."\n\
PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_SERVER" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" << EOF || true\n\
-- Добавляем недостающие колонки если их нет\n\
ALTER TABLE prediction ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();\n\
ALTER TABLE prediction ADD COLUMN IF NOT EXISTS forecast_qty_lower FLOAT;\n\
ALTER TABLE prediction ADD COLUMN IF NOT EXISTS forecast_qty_upper FLOAT;\n\
ALTER TABLE prediction ADD COLUMN IF NOT EXISTS meta_info TEXT;\n\
\n\
-- Обновляем created_at для существующих записей\n\
UPDATE prediction SET created_at = COALESCE(generated_at, NOW()) WHERE created_at IS NULL;\n\
\n\
-- Проверяем и добавляем user_sid если нужно\n\
DO $$\n\
BEGIN\n\
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns \n\
                   WHERE table_name = '"'"'prediction'"'"' \n\
                   AND column_name = '"'"'user_sid'"'"') THEN\n\
        ALTER TABLE prediction ADD COLUMN user_sid VARCHAR(22);\n\
        \n\
        -- Заполняем user_sid из связанных данных\n\
        UPDATE prediction p\n\
        SET user_sid = (\n\
            SELECT DISTINCT u.user_sid\n\
            FROM sale s\n\
            JOIN storeitem si ON s.store_item_sid = si.sid\n\
            JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid\n\
            JOIN upload u ON wi.upload_sid = u.sid\n\
            WHERE wi.product_sid = p.product_sid\n\
            LIMIT 1\n\
        )\n\
        WHERE p.user_sid IS NULL;\n\
        \n\
        -- Удаляем записи без user_sid\n\
        DELETE FROM prediction WHERE user_sid IS NULL;\n\
        \n\
        -- Делаем колонку обязательной если есть данные\n\
        IF EXISTS (SELECT 1 FROM prediction LIMIT 1) THEN\n\
            ALTER TABLE prediction ALTER COLUMN user_sid SET NOT NULL;\n\
        END IF;\n\
        \n\
        -- Добавляем внешний ключ если его нет\n\
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints \n\
                       WHERE constraint_name = '"'"'fk_prediction_user'"'"') THEN\n\
            ALTER TABLE prediction ADD CONSTRAINT fk_prediction_user \n\
            FOREIGN KEY (user_sid) REFERENCES "user"(sid) ON DELETE CASCADE;\n\
        END IF;\n\
        \n\
        -- Создаем индексы\n\
        CREATE INDEX IF NOT EXISTS ix_prediction_user_sid ON prediction(user_sid);\n\
        CREATE INDEX IF NOT EXISTS ix_prediction_product_user ON prediction(product_sid, user_sid);\n\
    END IF;\n\
END $$;\n\
EOF\n\
\n\
echo "Database fixes applied, running Alembic migrations..."\n\
\n\
# Применяем миграции\n\
alembic upgrade head || {\n\
    echo "Alembic upgrade failed, trying to fix..."\n\
    \n\
    # Если alembic_version не существует, создаем её\n\
    PGPASSWORD=$POSTGRES_PASSWORD psql -h "$POSTGRES_SERVER" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\n\
        CREATE TABLE IF NOT EXISTS alembic_version (\n\
            version_num VARCHAR(32) NOT NULL,\n\
            CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)\n\
        );\n\
    "\n\
    \n\
    # Помечаем текущее состояние\n\
    alembic stamp head\n\
    echo "Alembic state fixed"\n\
}\n\
\n\
echo "All migrations completed successfully!"' > /scripts/migrate.sh && chmod +x /scripts/migrate.sh

# Создаем простой entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Переменная порта для Railway
ENV PORT=8000

EXPOSE $PORT

ENTRYPOINT ["/entrypoint.sh"]

# Команда по умолчанию
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]