import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from jose import jwt
from datetime import datetime, timedelta
import nanoid
import uuid
import os
from unittest.mock import AsyncMock, patch, MagicMock

from app.db.session import get_db
from app.db.redis import get_redis
from app.main import app
from app.models.base import Base
from app.models.users import User, UserRole, VerificationToken
from app.models.inventory import (
    Category, Product, WarehouseItem, WarehouseItemStatus,
    Upload, StoreItem, StoreItemStatus, TimeFrame, Prediction, Sale, Discount
)
from app.core.config import settings
from app.core.security import get_password_hash

# Test database URL - use environment variable if available for CI/CD support
TEST_DB_URL = os.getenv("TEST_DB_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/test_inventory")

# Override settings for tests
settings.SQLALCHEMY_DATABASE_URI = TEST_DB_URL
settings.SECRET_KEY = "test_secret_key"
settings.EMAILS_FROM_EMAIL = "test@example.com"

# Create async engine and session
engine = create_async_engine(TEST_DB_URL)
TestingSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


@pytest.fixture(scope="session")
def event_loop():
    """Override event loop for pytest-asyncio"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db():
    """Create test database tables before tests and drop them after"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield

    # Clean up after tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(test_db):
    """Create a clean database session for each test"""
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture
async def test_user(db_session):
    """Create a test user and return credentials"""
    user = User(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        email="test@example.com",
        password_hash=get_password_hash("password123"),
        is_verified=True,
        role=UserRole.OWNER
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    return {
        "user": user,
        "email": "test@example.com",
        "password": "password123",
        "id": user.id,
        "sid": user.sid
    }


@pytest.fixture
async def test_category(db_session):
    """Create a test category"""
    category = Category(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        name="Test Category"
    )
    db_session.add(category)
    await db_session.commit()
    await db_session.refresh(category)
    return category


@pytest.fixture
async def test_product(db_session, test_category):
    """Create a test product"""
    product = Product(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        category_sid=test_category.sid,
        name="Test Product",
        barcode="1234567890123",
        default_unit="шт",
        default_price=100.0
    )
    db_session.add(product)
    await db_session.commit()
    await db_session.refresh(product)
    return product


@pytest.fixture
async def test_upload(db_session, test_user):
    """Create a test upload record"""
    upload = Upload(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        user_sid=test_user["sid"],
        file_name="test_file.csv",
        uploaded_at=datetime.utcnow(),
        rows_imported=1
    )
    db_session.add(upload)
    await db_session.commit()
    await db_session.refresh(upload)
    return upload


@pytest.fixture
async def test_warehouse_item(db_session, test_product, test_upload):
    """Create a test warehouse item"""
    warehouse_item = WarehouseItem(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        upload_sid=test_upload.sid,
        product_sid=test_product.sid,
        batch_code="TEST-001",
        quantity=10,
        expire_date=datetime.utcnow().date() + timedelta(days=30),
        received_at=datetime.utcnow().date(),
        status=WarehouseItemStatus.IN_STOCK
    )
    db_session.add(warehouse_item)
    await db_session.commit()
    await db_session.refresh(warehouse_item)
    return warehouse_item


@pytest.fixture
async def test_store_item(db_session, test_warehouse_item):
    """Create a test store item"""
    store_item = StoreItem(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        warehouse_item_sid=test_warehouse_item.sid,
        quantity=5,
        price=120.0,
        moved_at=datetime.utcnow(),
        status=StoreItemStatus.ACTIVE
    )
    db_session.add(store_item)
    await db_session.commit()
    await db_session.refresh(store_item)
    return store_item


@pytest.fixture
async def test_prediction(db_session, test_product):
    """Create a test prediction"""
    prediction = Prediction(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        product_sid=test_product.sid,
        timeframe=TimeFrame.MONTH,
        period_start=datetime.utcnow().date(),
        period_end=datetime.utcnow().date() + timedelta(days=30),
        forecast_qty=100.0,
        generated_at=datetime.utcnow(),
        model_version="test_model_v1"
    )
    db_session.add(prediction)
    await db_session.commit()
    await db_session.refresh(prediction)
    return prediction


@pytest.fixture
async def test_discount(db_session, test_store_item, test_user):
    """Create a test discount"""
    now = datetime.utcnow()
    discount = Discount(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        store_item_sid=test_store_item.sid,
        percentage=10.0,
        starts_at=now,
        ends_at=now + timedelta(days=7),
        created_by_sid=test_user["sid"]
    )
    db_session.add(discount)
    await db_session.commit()
    await db_session.refresh(discount)
    return discount


@pytest.fixture
async def test_sale(db_session, test_store_item, test_user):
    """Create a test sale"""
    sale = Sale(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        store_item_sid=test_store_item.sid,
        sold_qty=2,
        sold_price=120.0,
        sold_at=datetime.utcnow(),
        cashier_sid=test_user["sid"]
    )
    db_session.add(sale)

    # Update store item quantity
    store_item = await db_session.get(StoreItem, test_store_item.id)
    store_item.quantity -= 2

    await db_session.commit()
    await db_session.refresh(sale)
    return sale


@pytest.fixture
async def verification_token(db_session, test_user):
    """Create an email verification token"""
    token = VerificationToken(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        user_id=test_user["id"],
        token="123456",
        expires_at=datetime.utcnow() + timedelta(days=1)
    )
    db_session.add(token)
    await db_session.commit()
    await db_session.refresh(token)
    return token


@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    redis_mock.publish.return_value = 1
    return redis_mock


@pytest.fixture
async def client(db_session, mock_redis):
    """Create test client with mocked dependencies"""

    # Override db dependency
    async def override_get_db():
        yield db_session

    # Override redis dependency
    async def override_get_redis():
        yield mock_redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    # Clear overrides after test
    app.dependency_overrides = {}


@pytest.fixture
async def test_token(test_user):
    """Generate a test token"""
    payload = {
        "sub": test_user["sid"],
        "exp": datetime.utcnow() + timedelta(days=1),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


@pytest.fixture
def authorized_client(client, test_token):
    """Create authorized client with authentication token"""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_token}"
    }
    return client


# Email mocking
@pytest.fixture
def mock_email_service():
    """Mock email service"""
    with patch("app.services.email.send_email") as mock:
        mock.return_value = True
        yield mock


# File parsing mocking
@pytest.fixture
def mock_file_parser():
    """Mock file parser service"""
    with patch("app.services.file_parser.detect_and_parse_file") as mock:
        mock.return_value = [
            {
                "name": "Test Product",
                "category": "Test Category",
                "barcode": "9876543210123",
                "quantity": 15,
                "expire_date": (datetime.now() + timedelta(days=90)).date(),
                "received_at": datetime.now().date(),
                "batch_code": "TEST-002",
                "unit": "шт",
                "price": 150.0
            }
        ]
        yield mock


# Barcode service mocking
@pytest.fixture
def mock_barcode_service():
    """Mock barcode service"""
    with patch("app.services.barcode.decode_barcode_from_base64") as mock:
        mock.return_value = "1234567890123"
        yield mock


# Prediction service mocking
@pytest.fixture
def mock_prediction_service():
    """Mock prediction service"""
    with patch("app.services.prediction.PredictionService") as MockPredictionService:
        service_instance = MockPredictionService.return_value

        # Mock the generate_forecast method
        service_instance.generate_forecast.return_value = [
            {
                "product_sid": "test_product_sid",
                "timeframe": TimeFrame.MONTH,
                "period_start": datetime.now().date(),
                "period_end": (datetime.now() + timedelta(days=30)).date(),
                "forecast_qty": 120.0,
                "generated_at": datetime.now(),
                "model_version": "test_v1.0"
            }
        ]

        # Mock the save_forecast method
        service_instance.save_forecast.return_value = [
            Prediction(
                sid=nanoid.generate(size=22),
                product_sid="test_product_sid",
                timeframe=TimeFrame.MONTH,
                period_start=datetime.now().date(),
                period_end=(datetime.now() + timedelta(days=30)).date(),
                forecast_qty=120.0,
                generated_at=datetime.now(),
                model_version="test_v1.0"
            )
        ]

        yield service_instance