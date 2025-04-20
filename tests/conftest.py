import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from jose import jwt
from datetime import datetime, timedelta
import nanoid
import uuid

from app.db.session import get_db
from app.main import app
from app.models.base import Base
from app.models.users import User, UserRole
from app.models.inventory import (
    Category, Product, WarehouseItem, WarehouseItemStatus,
    Upload, StoreItem, StoreItemStatus, TimeFrame, Prediction
)
from app.core.config import settings
from app.core.security import get_password_hash

# Создаем тестовую базу данных
TEST_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_inventory"

# Переопределяем настройки для тестов
settings.SQLALCHEMY_DATABASE_URI = TEST_DB_URL
settings.SECRET_KEY = "test_secret_key"

engine = create_async_engine(TEST_DB_URL)
TestingSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


@pytest.fixture(scope="session")
def event_loop():
    """Переопределение event_loop для корректной работы с pytest-asyncio"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def test_db():
    """Создает тестовую базу и таблицы перед тестами, удаляет после"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Создаем тестовые данные
    async with TestingSessionLocal() as db:
        # Создаем тестового пользователя
        user = User(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            email="test@example.com",
            password_hash=get_password_hash("password"),
            is_verified=True,
            role=UserRole.OWNER
        )
        db.add(user)

        # Создаем тестовые категории
        category = Category(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            name="Test Category"
        )
        db.add(category)

        # Создаем тестовые продукты
        product = Product(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            category_sid=category.sid,
            name="Test Product",
            barcode="1234567890123",
            default_unit="шт",
            default_price=100.0
        )
        db.add(product)

        # Создаем тестовую загрузку
        upload = Upload(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            user_sid=user.sid,
            file_name="test_file.csv",
            uploaded_at=datetime.utcnow(),
            rows_imported=1
        )
        db.add(upload)

        # Создаем тестовый товар на складе
        warehouse_item = WarehouseItem(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            upload_sid=upload.sid,
            product_sid=product.sid,
            batch_code="TEST-001",
            quantity=10,
            expire_date=datetime.utcnow().date() + timedelta(days=30),
            received_at=datetime.utcnow().date(),
            status=WarehouseItemStatus.IN_STOCK
        )
        db.add(warehouse_item)

        # Создаем тестовый товар в магазине
        store_item = StoreItem(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            warehouse_item_sid=warehouse_item.sid,
            quantity=5,
            price=120.0,
            moved_at=datetime.utcnow(),
            status=StoreItemStatus.ACTIVE
        )
        db.add(store_item)

        # Создаем тестовый прогноз
        prediction = Prediction(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            product_sid=product.sid,
            timeframe=TimeFrame.MONTH,
            period_start=datetime.utcnow().date(),
            period_end=datetime.utcnow().date() + timedelta(days=30),
            forecast_qty=100.0,
            generated_at=datetime.utcnow(),
            model_version="test_model_v1"
        )
        db.add(prediction)

        await db.commit()

        yield db

    # Удаляем тестовую базу после тестов
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session():
    """Создает новую сессию базы данных для теста"""
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture
async def client(test_db):
    """Создает тестовый клиент FastAPI"""

    # Переопределяем зависимость базы данных
    async def override_get_db():
        async with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    app.dependency_overrides = {}


@pytest.fixture
def test_user():
    """Возвращает данные тестового пользователя"""
    return {
        "email": "test@example.com",
        "password": "password"
    }


@pytest.fixture
async def test_token(client, test_user):
    """Получает токен авторизации для тестового пользователя"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )
    return response.json()["access_token"]


@pytest.fixture
def authorized_client(client, test_token):
    """Создает авторизованный клиент с заголовком Authorization"""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_token}"
    }
    return client