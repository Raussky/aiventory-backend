FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей по группам для уменьшения использования памяти
# Группа 1: Основные инструменты сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Группа 2: PostgreSQL и Redis клиенты
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Группа 3: Библиотеки для обработки изображений и штрих-кодов
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*

# Группа 4: Дополнительные библиотеки
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем зависимости Python по группам
# Основные зависимости
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.103.1 \
    uvicorn==0.23.2 \
    sqlalchemy==2.0.21 \
    alembic==1.12.0 \
    asyncpg==0.28.0 \
    redis==4.6.0 \
    pydantic==2.3.0 \
    pydantic-settings==2.0.3

# Остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Prophet и plotly отдельно
RUN pip install --no-cache-dir plotly==5.17.0 && \
    pip install --no-cache-dir prophet==1.1.5 || echo "Prophet installation failed, will use fallback methods"

# Копируем entrypoint скрипт
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Копируем код приложения
COPY . .

# Переменная порта
ENV PORT=8000

EXPOSE $PORT

ENTRYPOINT ["/docker-entrypoint.sh"]

# Команда по умолчанию
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]