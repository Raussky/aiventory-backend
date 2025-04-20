import pytest
import uuid
import nanoid
from datetime import datetime, timedelta

from app.models.base import Base
from app.models.users import User, UserRole, VerificationToken
from app.models.inventory import (
    Category, Product, WarehouseItem, WarehouseItemStatus,
    StoreItem, StoreItemStatus, Sale, Discount, TimeFrame, Prediction
)


def test_base_model_generate_sid():
    """Test the generation of short IDs"""
    sid = Base.generate_sid()
    assert isinstance(sid, str)
    assert len(sid) == 22  # NanoID default length


@pytest.mark.asyncio
async def test_user_model_relationships(db_session):
    """Test User model relationships"""
    # Create user
    user = User(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        email="test_relationships@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.ADMIN
    )
    db_session.add(user)
    await db_session.commit()

    # Create verification token for user
    token = VerificationToken(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        user_id=user.id,
        token="123456",
        expires_at=datetime.utcnow() + timedelta(days=1)
    )
    db_session.add(token)
    await db_session.commit()

    # Verify relationship
    result = await db_session.execute(
        "SELECT * FROM verificationtoken WHERE user_id = :user_id",
        {"user_id": user.id}
    )
    token_record = result.mappings().one_or_none()
    assert token_record is not None
    assert token_record["token"] == "123456"


@pytest.mark.asyncio
async def test_inventory_models_cascade(db_session):
    """Test cascade behavior of inventory models"""
    # Create a category
    category = Category(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        name="Test Cascade Category"
    )
    db_session.add(category)
    await db_session.commit()

    # Create a product in the category
    product = Product(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        category_sid=category.sid,
        name="Test Cascade Product",
        barcode="9988776655443",
        default_unit="шт",
        default_price=200.0
    )
    db_session.add(product)
    await db_session.commit()

    # Create a user for upload
    user = User(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        email="test_cascade@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.MANAGER
    )
    db_session.add(user)
    await db_session.commit()

    # Create an upload record
    upload = Upload(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        user_sid=user.sid,
        file_name="cascade_test.csv",
        uploaded_at=datetime.utcnow(),
        rows_imported=1
    )
    db_session.add(upload)
    await db_session.commit()

    # Create a warehouse item
    warehouse_item = WarehouseItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        upload_sid=upload.sid,
        product_sid=product.sid,
        batch_code="CASCADE-001",
        quantity=10,
        expire_date=datetime.utcnow().date() + timedelta(days=30),
        received_at=datetime.utcnow().date(),
        status=WarehouseItemStatus.IN_STOCK
    )
    db_session.add(warehouse_item)
    await db_session.commit()

    # Create a store item
    store_item = StoreItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        warehouse_item_sid=warehouse_item.sid,
        quantity=5,
        price=220.0,
        moved_at=datetime.utcnow(),
        status=StoreItemStatus.ACTIVE
    )
    db_session.add(store_item)
    await db_session.commit()

    # Create a sale
    sale = Sale(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        store_item_sid=store_item.sid,
        sold_qty=1,
        sold_price=220.0,
        sold_at=datetime.utcnow(),
        cashier_sid=user.sid
    )
    db_session.add(sale)
    await db_session.commit()

    # Now query to check relationships
    result = await db_session.execute("""
        SELECT 
            p.name as product_name, 
            c.name as category_name,
            wi.batch_code,
            si.quantity as store_quantity,
            s.sold_qty
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN product p ON wi.product_sid = p.sid
        JOIN category c ON p.category_sid = c.sid
        WHERE s.sid = :sale_sid
    """, {"sale_sid": sale.sid})

    record = result.mappings().one_or_none()
    assert record is not None
    assert record["product_name"] == "Test Cascade Product"
    assert record["category_name"] == "Test Cascade Category"
    assert record["batch_code"] == "CASCADE-001"
    assert record["store_quantity"] == 5
    assert record["sold_qty"] == 1


@pytest.mark.asyncio
async def test_enum_types(db_session):
    """Test that enum types work correctly in all models"""
    # Test UserRole enum
    user = User(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        email="test_enum@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.MANAGER
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    assert user.role == UserRole.MANAGER

    # Create models for inventory enum testing
    category = Category(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        name="Enum Test Category"
    )
    db_session.add(category)

    product = Product(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        category_sid=category.sid,
        name="Enum Test Product",
        barcode="1122334455667",
        default_unit="шт",
        default_price=150.0
    )
    db_session.add(product)

    # Test WarehouseItemStatus enum
    warehouse_item = WarehouseItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        upload_sid=user.sid,  # Just using user sid for simplicity in this test
        product_sid=product.sid,
        batch_code="ENUM-001",
        quantity=10,
        expire_date=datetime.utcnow().date() + timedelta(days=30),
        received_at=datetime.utcnow().date(),
        status=WarehouseItemStatus.MOVED  # Test non-default enum value
    )
    db_session.add(warehouse_item)

    # Test StoreItemStatus enum
    store_item = StoreItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        warehouse_item_sid=warehouse_item.sid,
        quantity=5,
        price=170.0,
        moved_at=datetime.utcnow(),
        status=StoreItemStatus.EXPIRED  # Test non-default enum value
    )
    db_session.add(store_item)

    # Test TimeFrame enum
    prediction = Prediction(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        product_sid=product.sid,
        timeframe=TimeFrame.WEEK,  # Test non-default enum value
        period_start=datetime.utcnow().date(),
        period_end=datetime.utcnow().date() + timedelta(days=7),
        forecast_qty=35.0,
        generated_at=datetime.utcnow(),
        model_version="enum_test_v1"
    )
    db_session.add(prediction)

    await db_session.commit()

    # Refresh all models to load from DB
    await db_session.refresh(warehouse_item)
    await db_session.refresh(store_item)
    await db_session.refresh(prediction)

    # Validate enum values
    assert warehouse_item.status == WarehouseItemStatus.MOVED
    assert store_item.status == StoreItemStatus.EXPIRED
    assert prediction.timeframe == TimeFrame.WEEK


@pytest.mark.asyncio
async def test_discount_date_validation(db_session):
    """Test that discount dates are validated correctly"""
    # Create necessary models first
    category = Category(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        name="Discount Test Category"
    )
    db_session.add(category)

    product = Product(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        category_sid=category.sid,
        name="Discount Test Product",
        barcode="8877665544332",
        default_unit="шт",
        default_price=300.0
    )
    db_session.add(product)

    user = User(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        email="discount_test@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.MANAGER
    )
    db_session.add(user)

    warehouse_item = WarehouseItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        upload_sid=user.sid,  # Just using user sid for simplicity in this test
        product_sid=product.sid,
        batch_code="DISCOUNT-001",
        quantity=20,
        expire_date=datetime.utcnow().date() + timedelta(days=60),
        received_at=datetime.utcnow().date(),
        status=WarehouseItemStatus.IN_STOCK
    )
    db_session.add(warehouse_item)

    store_item = StoreItem(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        warehouse_item_sid=warehouse_item.sid,
        quantity=15,
        price=320.0,
        moved_at=datetime.utcnow(),
        status=StoreItemStatus.ACTIVE
    )
    db_session.add(store_item)

    await db_session.commit()

    # Now create a discount with end date before start date
    now = datetime.utcnow()
    invalid_discount = Discount(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        store_item_sid=store_item.sid,
        percentage=25.0,
        starts_at=now + timedelta(days=10),  # Starts in 10 days
        ends_at=now + timedelta(days=5),  # Ends in 5 days (before start)
        created_by_sid=user.sid
    )
    db_session.add(invalid_discount)

    # This should raise an exception on constraint or validation depending on implementation
    with pytest.raises(Exception):  # Generic exception since exact type depends on implementation
        await db_session.commit()

    # Rollback and create a valid discount
    await db_session.rollback()

    valid_discount = Discount(
        id=uuid.uuid4(),
        sid=Base.generate_sid(),
        store_item_sid=store_item.sid,
        percentage=25.0,
        starts_at=now,  # Starts now
        ends_at=now + timedelta(days=10),  # Ends in 10 days
        created_by_sid=user.sid
    )
    db_session.add(valid_discount)

    await db_session.commit()
    await db_session.refresh(valid_discount)

    assert valid_discount.percentage == 25.0
    assert valid_discount.starts_at <= valid_discount.ends_at