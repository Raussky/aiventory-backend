#!/bin/bash

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция вывода заголовка
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Функция печати подсказки
print_step() {
    echo -e "${GREEN}$1${NC}"
}

# Функция печати предупреждения
print_warning() {
    echo -e "${YELLOW}ВНИМАНИЕ: $1${NC}"
}

# Функция печати ошибки
print_error() {
    echo -e "${RED}ОШИБКА: $1${NC}"
}

# Проверка существования .env файла
check_env_file() {
    if [ ! -f .env ]; then
        print_header "Создание файла .env"
        cat > .env << EOF
# API settings
SECRET_KEY=$(openssl rand -hex 32)
API_V1_STR=/api/v1
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# PostgreSQL settings
POSTGRES_SERVER=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=aiventory

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis-password

# Email settings
SMTP_TLS=True
SMTP_PORT=587
SMTP_HOST=smtp.example.com
SMTP_USER=user@example.com
SMTP_PASSWORD=your-smtp-password
EMAILS_FROM_EMAIL=noreply@example.com
EMAILS_FROM_NAME=Inventory System
EOF
        print_step "Создан файл .env с настройками по умолчанию. Отредактируйте его при необходимости."
    else
        print_step "Файл .env уже существует."
    fi
}

# Функция для создания nginx директорий и конфигов
setup_nginx() {
    print_header "Настройка Nginx"

    mkdir -p nginx/conf
    mkdir -p nginx/ssl

    # Создание конфигурации Nginx
    cat > nginx/conf/default.conf << EOF
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    location /api {
        proxy_pass http://api:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

    print_step "Nginx директории и конфигурации созданы."
}

# Запуск в Docker
start_with_docker() {
    print_header "Запуск проекта с Docker"

    # Проверка наличия Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker не установлен. Установите Docker и повторите попытку."
        exit 1
    fi

    # Проверка наличия Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose не установлен. Установите Docker Compose и повторите попытку."
        exit 1
    fi

    check_env_file
    setup_nginx

    print_step "Запуск контейнеров..."
    docker-compose up -d

    print_step "Ожидание готовности PostgreSQL..."
    sleep 10

    print_step "Применение миграций..."
    docker-compose exec api alembic upgrade head

    print_header "Проект успешно запущен!"
    echo "API доступен по адресу: http://localhost:8000/api/docs"
    echo "Celery worker и beat запущены"
    echo "PostgreSQL запущен на порту 5432"
    echo "Redis запущен на порту 6379"
    echo "Nginx запущен на порту 80"
}

# Запуск локально с Conda
start_with_conda() {
    print_header "Запуск проекта локально с Conda"

    # Проверка наличия Conda
    if ! command -v conda &> /dev/null; then
        print_error "Conda не установлена. Установите Miniconda или Anaconda и повторите попытку."
        exit 1
    fi

    check_env_file

    # Создание conda окружения
    print_step "Создание conda окружения..."
    conda create -n inventory python=3.10 -y

    # Активация окружения
    print_step "Активация окружения..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate inventory

    # Установка prophet через conda
    print_step "Установка prophet через conda..."
    conda install -c conda-forge prophet -y

    # Установка зависимостей
    print_step "Установка остальных зависимостей..."
    pip install -r requirements.txt

    # Установка дополнительных нужных пакетов
    pip install greenlet==2.0.2

    # Для macOS может понадобиться zbar
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_step "Установка zbar для macOS..."
        if command -v brew &> /dev/null; then
            brew install zbar
            ln -sf "$(brew --prefix)/lib/libzbar.dylib" "$(conda info --base)/envs/inventory/lib/libzbar.dylib"
        else
            print_warning "Homebrew не установлен. Возможно, потребуется вручную установить zbar."
        fi
    fi

    # Запуск PostgreSQL (локально или через Docker)
    print_step "Запуск PostgreSQL..."
    if command -v docker &> /dev/null; then
        docker run --name inventory-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=aiventory -p 5432:5432 -d postgres:14
    else
        print_warning "Docker не установлен. Убедитесь, что PostgreSQL запущен локально."
    fi

    # Запуск Redis (локально или через Docker)
    print_step "Запуск Redis..."
    if command -v docker &> /dev/null; then
        docker run --name inventory-redis -p 6379:6379 -d redis:alpine redis-server --requirepass "redis-password"
    else
        print_warning "Docker не установлен. Убедитесь, что Redis запущен локально."
    fi

    # Ожидание готовности базы данных
    print_step "Ожидание готовности PostgreSQL..."
    sleep 5

    # Создание и применение миграций
    print_step "Инициализация и применение миграций..."
    export PYTHONPATH=$PWD

    # Проверка существования alembic.ini
    if [ ! -f alembic.ini ]; then
        alembic init alembic

        # Обновление alembic/env.py для работы с моделями
        cat > alembic/env.py << EOF
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Переопределение URL-адреса подключения к базе данных из .env
postgres_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_SERVER')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
config.set_main_option("sqlalchemy.url", postgres_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# импорт моделей
from app.models.base import Base
from app.models.users import User, VerificationToken
from app.models.inventory import *

# привязка к моделям
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF
    fi

    alembic revision --autogenerate -m "Initial migration"
    alembic upgrade head

    # Запуск сервисов в отдельных терминалах
    print_header "Запуск сервисов"
    print_step "Запуск FastAPI сервера..."

    # Для macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"'

        print_step "Запуск Celery worker..."
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app worker --loglevel=info"'

        print_step "Запуск Celery beat..."
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app beat --loglevel=info"'
    # Для Linux
    else
        gnome-terminal -- bash -c "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload; exec bash" || xterm -e "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload" &

        gnome-terminal -- bash -c "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app worker --loglevel=info; exec bash" || xterm -e "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app worker --loglevel=info" &

        gnome-terminal -- bash -c "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app beat --loglevel=info; exec bash" || xterm -e "cd $(pwd) && source $(conda info --base)/etc/profile.d/conda.sh && conda activate inventory && celery -A app.tasks.celery_app beat --loglevel=info" &
    fi

    print_header "Проект успешно запущен!"
    echo "API доступен по адресу: http://localhost:8000/api/docs"
    echo "Celery worker и beat запущены в отдельных терминалах"
    echo "PostgreSQL запущен на порту 5432"
    echo "Redis запущен на порту 6379"
}

# Главное меню
print_header "Скрипт запуска проекта инвентаризации"
echo "Выберите способ запуска:"
echo "1) Запуск с Docker (рекомендуется)"
echo "2) Запуск локально с Conda"
echo "3) Выход"

read -p "Введите номер (1-3): " choice

case $choice in
    1)
        start_with_docker
        ;;
    2)
        start_with_conda
        ;;
    3)
        echo "Выход из скрипта."
        exit 0
        ;;
    *)
        print_error "Неверный выбор. Выход из скрипта."
        exit 1
        ;;
esac