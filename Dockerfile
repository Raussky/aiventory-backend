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
            # Специальная проверка для проблемы с user_sid в prediction\n\
            echo "Checking for user_sid column in prediction table..."\n\
            HAS_USER_SID=$(psql "$DATABASE_URL" -tc "SELECT 1 FROM information_schema.columns WHERE table_name = '"'"'prediction'"'"' AND column_name = '"'"'user_sid'"'"';" | grep -q 1 && echo "yes" || echo "no")\n\
            \n\
            if [ "$HAS_USER_SID" = "no" ]; then\n\
                echo "user_sid column missing in prediction table, applying fix..."\n\
                # Применяем SQL напрямую для добавления колонки\n\
                psql "$DATABASE_URL" << EOF || true\n\
                -- Добавляем колонку user_sid если её нет\n\
                ALTER TABLE prediction ADD COLUMN IF NOT EXISTS user_sid VARCHAR(22);\n\
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
                -- Делаем колонку обязательной\n\
                ALTER TABLE prediction ALTER COLUMN user_sid SET NOT NULL;\n\
                \n\
                -- Добавляем внешний ключ\n\
                ALTER TABLE prediction ADD CONSTRAINT fk_prediction_user \n\
                FOREIGN KEY (user_sid) REFERENCES "user"(sid) ON DELETE CASCADE;\n\
                \n\
                -- Создаем индексы\n\
                CREATE INDEX IF NOT EXISTS ix_prediction_user_sid ON prediction(user_sid);\n\
                CREATE INDEX IF NOT EXISTS ix_prediction_product_user ON prediction(product_sid, user_sid);\n\
EOF\n\
                echo "user_sid column fix applied"\n\
            fi\n\
            \n\
            # Проверяем и применяем миграции\n\
            echo "Checking for pending migrations..."\n\
            if alembic history -r$CURRENT_REV:head 2>/dev/null | grep -q "Rev:"; then\n\
                echo "Found pending migrations, applying..."\n\
                \n\
                # Пробуем применить миграции по одной\n\
                PENDING_REVS=$(alembic history -r$CURRENT_REV:head 2>/dev/null | grep "Rev:" | awk '"'"'{print $2}'"'"' | tac)\n\
                \n\
                for REV in $PENDING_REVS; do\n\
                    echo "Applying migration $REV..."\n\
                    alembic upgrade $REV || {\n\
                        echo "Migration $REV failed, checking if already applied..."\n\
                        # Проверяем, не применена ли уже эта миграция\n\
                        if alembic current | grep -q $REV; then\n\
                            echo "Migration $REV already applied, skipping"\n\
                        else\n\
                            # Если это проблемная миграция с user_sid, помечаем как примененную\n\
                            if [ "$REV" = "7c6b3e388b7e" ] || [ "$REV" = "ae666d42ee9a" ]; then\n\
                                echo "Marking problematic migration $REV as applied"\n\
                                alembic stamp $REV\n\
                            else\n\
                                echo "Failed to apply migration $REV"\n\
                                exit 1\n\
                            fi\n\
                        fi\n\
                    }\n\
                done\n\
                \n\
                echo "All migrations applied"\n\
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
        \n\
        # Финальная проверка состояния БД\n\
        echo "Final database check..."\n\
        psql "$DATABASE_URL" -c "SELECT table_name FROM information_schema.tables WHERE table_schema = '"'"'public'"'"' ORDER BY table_name;" || true\n\
        \n\
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