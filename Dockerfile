# Dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libzbar0 \
    libzbar-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .

# Создание conda-окружения с установкой prophet и других зависимостей
RUN conda create -n inventory python=3.10 -y && \
    echo "source activate inventory" > ~/.bashrc && \
    /bin/bash -c "source activate inventory && \
    conda install -c conda-forge prophet -y && \
    pip install --no-cache-dir -r requirements.txt"

# Устанавливаем переменную окружения PATH, чтобы все команды использовали python из conda
ENV PATH /opt/conda/envs/inventory/bin:$PATH

# Копирование кода
COPY . .

# Запуск приложения через conda-окружение
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]