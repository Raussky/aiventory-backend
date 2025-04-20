import pytest
from pydantic import ValidationError
from datetime import datetime, date, timedelta

from app.schemas.user import UserCreate, UserVerify, UserResponse
from app.schemas.inventory import (
    CategoryCreate, ProductCreate, WarehouseItemCreate,
    WarehouseItemResponse, WarehouseItemFilter
)
from app.schemas.store import (
    StoreItemCreate, DiscountCreate, SaleCreate,
    StoreItemStatus, StoreItemFilter
)
from app.schemas.prediction import (
    PredictionCreate, PredictionRequest, TimeFrame
)


def test_user_create_schema():
    """Test UserCreate schema validation"""
    # Valid user
    valid_user = UserCreate(email="test@example.com", password="securepass123")
    assert valid_user.email == "test@example.com"
    assert valid_user.password == "securepass123"

    # Test email validation
    with pytest.raises(ValidationError):
        UserCreate(email="not-an-email", password="password123")

    # Test password validation (assuming min_length is set)
    with pytest.raises(ValidationError):
        UserCreate(email="test@example.com", password="short")


def test_user_verify_schema():
    """Test UserVerify schema validation"""
    valid_verify = UserVerify(email="test@example.com", code="123456")
    assert valid_verify.email == "test@example.com"
    assert valid_verify.code == "123456"

    # Test email validation
    with pytest.raises(ValidationError):
        UserVerify(email="not-an-email", code="123456")


def test_warehouse_item_create_schema():
    """Test WarehouseItemCreate schema validation"""
    valid_item = WarehouseItemCreate(
        product_sid="product123",
        batch_code="BATCH001",
        quantity=10,
        expire_date=date.today() + timedelta(days=30),
        received_at=date.today()
    )
    assert valid_item.product_sid == "product123"
    assert valid_item.quantity == 10

    # Test quantity validation
    with pytest.raises(ValidationError):
        WarehouseItemCreate(
            product_sid="product123",
            batch_code="BATCH001",
            quantity=0,  # Invalid: must be positive
            expire_date=date.today() + timedelta(days=30),
            received_at=date.today()
        )

    with pytest.raises(ValidationError):
        WarehouseItemCreate(
            product_sid="product123",
            batch_code="BATCH001",
            quantity=-5,  # Invalid: must be positive
            expire_date=date.today() + timedelta(days=30),
            received_at=date.today()
        )


def test_discount_create_schema():
    """Test DiscountCreate schema validation"""
    now = datetime.utcnow()

    valid_discount = DiscountCreate(
        store_item_sid="storeitem123",
        percentage=15.0,
        starts_at=now,
        ends_at=now + timedelta(days=7)
    )
    assert valid_discount.store_item_sid == "storeitem123"
    assert valid_discount.percentage == 15.0

    # Test percentage validation (must be between 0 and 100)
    with pytest.raises(ValidationError):
        DiscountCreate(
            store_item_sid="storeitem123",
            percentage=-5.0,  # Invalid: must be >= 0
            starts_at=now,
            ends_at=now + timedelta(days=7)
        )

    with pytest.raises(ValidationError):
        DiscountCreate(
            store_item_sid="storeitem123",
            percentage=110.0,  # Invalid: must be <= 100
            starts_at=now,
            ends_at=now + timedelta(days=7)
        )

    # Test date validation (end_date must be after start_date)
    with pytest.raises(ValidationError):
        DiscountCreate(
            store_item_sid="storeitem123",
            percentage=15.0,
            starts_at=now,
            ends_at=now - timedelta(days=1)  # Invalid: before start_date
        )


def test_sale_create_schema():
    """Test SaleCreate schema validation"""
    valid_sale = SaleCreate(
        store_item_sid="storeitem123",
        sold_qty=3,
        sold_price=150.0
    )
    assert valid_sale.store_item_sid == "storeitem123"
    assert valid_sale.sold_qty == 3
    assert valid_sale.sold_price == 150.0

    # Test quantity validation
    with pytest.raises(ValidationError):
        SaleCreate(
            store_item_sid="storeitem123",
            sold_qty=0,  # Invalid: must be > 0
            sold_price=150.0
        )

    # Test price validation
    with pytest.raises(ValidationError):
        SaleCreate(
            store_item_sid="storeitem123",
            sold_qty=3,
            sold_price=-10.0  # Invalid: must be >= 0
        )


def test_prediction_request_schema():
    """Test PredictionRequest schema validation"""
    valid_request = PredictionRequest(
        product_sid="product123",
        timeframe=TimeFrame.MONTH,
        periods=6,
        refresh=False
    )
    assert valid_request.product_sid == "product123"
    assert valid_request.timeframe == TimeFrame.MONTH
    assert valid_request.periods == 6
    assert valid_request.refresh is False

    # Test periods validation (must be between 1 and 12)
    with pytest.raises(ValidationError):
        PredictionRequest(
            product_sid="product123",
            timeframe=TimeFrame.MONTH,
            periods=0  # Invalid: must be >= 1
        )

    with pytest.raises(ValidationError):
        PredictionRequest(
            product_sid="product123",
            timeframe=TimeFrame.MONTH,
            periods=15  # Invalid: must be <= 12
        )


def test_filter_schemas():
    """Test filter schemas"""
    # Test WarehouseItemFilter
    valid_wh_filter = WarehouseItemFilter(
        upload_sid="upload123",
        expire_soon=True,
        product_sid="product123",
        status="in_stock",
        search="test"
    )
    assert valid_wh_filter.upload_sid == "upload123"
    assert valid_wh_filter.expire_soon is True
    assert valid_wh_filter.product_sid == "product123"
    assert valid_wh_filter.status == "in_stock"
    assert valid_wh_filter.search == "test"

    # Test StoreItemFilter
    valid_store_filter = StoreItemFilter(
        status=StoreItemStatus.ACTIVE,
        expired_only=False,
        search="test",
        category_sid="category123"
    )
    assert valid_store_filter.status == StoreItemStatus.ACTIVE
    assert valid_store_filter.expired_only is False
    assert valid_store_filter.search == "test"
    assert valid_store_filter.category_sid == "category123"


def test_schema_config_from_attributes():
    """Test that schemas correctly use from_attributes=True"""
    # These schemas should have Config with from_attributes=True
    # We're checking the class attribute exists with expected value
    assert hasattr(UserResponse.Config, "from_attributes")
    assert UserResponse.Config.from_attributes is True

    assert hasattr(WarehouseItemResponse.Config, "from_attributes")
    assert WarehouseItemResponse.Config.from_attributes is True