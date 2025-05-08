# Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Устанавливаем системные зависимости, включая libgl1 для OpenCV
# Заменен netcat на netcat-openbsd
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libzbar0 \
    libzbar-dev \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    netcat-openbsd \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .

# Создание conda-окружения с установкой prophet и других зависимостей
RUN conda create -n inventory python=3.10 -y && \
    echo "source activate inventory" > ~/.bashrc && \
    /bin/bash -c "source activate inventory && \
    conda install -c conda-forge prophet -y && \
    pip install --no-cache-dir -r requirements.txt"

# Устанавливаем переменную окружения PATH
ENV PATH /opt/conda/envs/inventory/bin:$PATH

# Копирование кода
COPY . .

# Создаем и делаем исполняемым скрипт для запуска
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Используем entrypoint скрипт
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Запуск приложения через conda-окружение
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]