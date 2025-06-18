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
    redis-tools \
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

# Копируем entrypoint скрипт
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Копируем код приложения
COPY . .

# Переменная порта для Railway
ENV PORT=8000

EXPOSE $PORT

ENTRYPOINT ["/docker-entrypoint.sh"]

# Команда по умолчанию
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]