import pytest
from httpx import AsyncClient
import os
from datetime import datetime, timedelta
import io
import csv

from app.core.config import settings
from app.models.inventory import WarehouseItemStatus, StoreItemStatus


@pytest.mark.asyncio
async def test_get_warehouse_items(authorized_client, test_warehouse_item):
    """Test getting warehouse items"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/warehouse/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert "sid" in items[0]
    assert "product" in items[0]
    assert items[0]["sid"] == test_warehouse_item.sid


@pytest.mark.asyncio
async def test_get_warehouse_items_with_pagination(authorized_client, test_warehouse_item):
    """Test pagination in warehouse items"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/warehouse/items",
        params={"skip": 0, "limit": 5}
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) <= 5


@pytest.mark.asyncio
async def test_filter_by_upload(authorized_client, test_warehouse_item, test_upload):
    """Test filtering warehouse items by upload"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/warehouse/items",
        params={"upload_sid": test_upload.sid}
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert items[0]["upload_sid"] == test_upload.sid


@pytest.mark.asyncio
async def test_filter_expiring_items(authorized_client, db_session, test_product, test_upload):
    """Test filtering warehouse items with expiring date"""
    # Create an item with expiring date
    from app.models.inventory import WarehouseItem
    import uuid
    import nanoid

    expiring_item = WarehouseItem(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        upload_sid=test_upload.sid,
        product_sid=test_product.sid,
        batch_code="EXPIRE-001",
        quantity=5,
        expire_date=datetime.utcnow().date() + timedelta(days=5),  # expires in 5 days
        received_at=datetime.utcnow().date(),
        status=WarehouseItemStatus.IN_STOCK
    )
    db_session.add(expiring_item)
    await db_session.commit()

    response = await authorized_client.get(
        f"{settings.API_V1_STR}/warehouse/items",
        params={"expire_soon": True}
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0

    # Find the expiring item
    expiring_found = False
    for item in items:
        if item["sid"] == expiring_item.sid:
            expiring_found = True
            assert item["batch_code"] == "EXPIRE-001"

    assert expiring_found, "Expiring item not found in filtered results"


@pytest.mark.asyncio
async def test_upload_csv_success(authorized_client, mock_file_parser, test_user):
    """Test successful CSV file upload"""
    # Create a simple CSV file
    csv_content = io.StringIO()
    writer = csv.writer(csv_content)
    writer.writerow(["name", "category", "barcode", "quantity", "expire_date", "batch_code", "unit", "price"])
    writer.writerow(["Test Product", "Test Category", "1234567890123", "10", "2025-05-01", "TEST-001", "шт", "100.0"])
    csv_content.seek(0)

    files = {"file": ("test.csv", csv_content.getvalue().encode(), "text/csv")}

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/upload",
        files=files
    )
    assert response.status_code == 200
    result = response.json()
    assert result["file_name"] == "test.csv"
    assert result["rows_imported"] > 0
    assert "sid" in result

    # Check that file parser was called
    mock_file_parser.assert_called_once()


@pytest.mark.asyncio
async def test_upload_excel_success(authorized_client, mock_file_parser, test_user):
    """Test successful Excel file upload"""
    # Just mock the file since we're mocking the parser
    files = {"file": (
    "test.xlsx", b"mock excel content", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/upload",
        files=files
    )
    assert response.status_code == 200
    result = response.json()
    assert result["file_name"] == "test.xlsx"
    assert "sid" in result

    # Check that file parser was called
    mock_file_parser.assert_called_once()


@pytest.mark.asyncio
async def test_upload_invalid_file(authorized_client):
    """Test upload with invalid file type"""
    files = {"file": ("test.txt", b"Invalid file content", "text/plain")}

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/upload",
        files=files
    )
    assert response.status_code == 500  # or 400, depends on implementation
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_move_to_store_by_item_sid(authorized_client, test_warehouse_item, mock_redis):
    """Test moving item to store by item_sid"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/to-store",
        data={
            "item_sid": test_warehouse_item.sid,
            "quantity": 2,
            "price": 125.0
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "store_item_sid" in result
    assert "message" in result
    assert "successfully" in result["message"].lower()

    # Check Redis lock was set and released
    mock_redis.set.assert_called_once()
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_move_to_store_with_barcode(authorized_client, test_product, test_warehouse_item, mock_barcode_service,
                                          mock_redis):
    """Test moving item to store using barcode image"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/to-store",
        data={
            "barcode_image": "base64_encoded_image_data",
            "quantity": 1,
            "price": 110.0
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "store_item_sid" in result

    # Check barcode service was called
    mock_barcode_service.assert_called_once_with("base64_encoded_image_data")

    # Check Redis lock was set and released
    mock_redis.set.assert_called_once()
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_move_to_store_insufficient_quantity(authorized_client, db_session, test_warehouse_item):
    """Test moving more items than available to store"""
    # Set quantity to 1
    await db_session.execute(
        f"UPDATE warehouseitem SET quantity = 1 WHERE sid = '{test_warehouse_item.sid}'"
    )
    await db_session.commit()

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/to-store",
        data={
            "item_sid": test_warehouse_item.sid,
            "quantity": 5,  # More than available
            "price": 125.0
        }
    )
    assert response.status_code == 400
    assert "not enough quantity" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_move_to_store_nonexistent_item(authorized_client):
    """Test moving nonexistent item to store"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/to-store",
        data={
            "item_sid": "nonexistent_sid",
            "quantity": 1,
            "price": 100.0
        }
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_move_to_store_status_update(authorized_client, db_session, test_warehouse_item):
    """Test that warehouse item status is updated when all units are moved"""
    # First set quantity to 1
    await db_session.execute(
        f"UPDATE warehouseitem SET quantity = 1 WHERE sid = '{test_warehouse_item.sid}'"
    )
    await db_session.commit()

    # Now move that 1 item to store
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/to-store",
        data={
            "item_sid": test_warehouse_item.sid,
            "quantity": 1,  # All available
            "price": 125.0
        }
    )
    assert response.status_code == 200

    # Check status was updated
    result = await db_session.execute(
        f"SELECT status FROM warehouseitem WHERE sid = '{test_warehouse_item.sid}'"
    )
    status = result.scalar_one()
    assert status == WarehouseItemStatus.MOVED