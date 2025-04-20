import pytest
import io
import csv
from httpx import AsyncClient

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_warehouse_items(authorized_client):
    """Тест получения списка товаров на складе"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/warehouse/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert "sid" in items[0]
    assert "product" in items[0]


@pytest.mark.asyncio
async def test_filter_expiring_items(authorized_client):
    """Тест фильтрации товаров с истекающим сроком годности"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/warehouse/items",
        params={"expire_soon": True}
    )
    assert response.status_code == 200


@pytest.fixture
def csv_file():
    """Создает тестовый CSV-файл"""
    file_content = io.StringIO()
    writer = csv.writer(file_content)
    writer.writerow(
        ["name", "category", "barcode", "quantity", "expire_date", "received_at", "batch_code", "unit", "price"])
    writer.writerow(
        ["Test CSV Product", "Test Category", "9876543210123", "15", "2025-06-01", "2025-04-20", "CSV-TEST-001", "шт",
         "150.00"])
    file_content.seek(0)
    return file_content.getvalue().encode()


@pytest.mark.asyncio
async def test_upload_csv(authorized_client, csv_file):
    """Тест загрузки CSV-файла"""
    files = {"file": ("test_upload.csv", csv_file, "text/csv")}
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/warehouse/upload",
        files=files
    )
    assert response.status_code == 200
    json = response.json()
    assert json["file_name"] == "test_upload.csv"
    assert json["rows_imported"] >= 1


@pytest.mark.asyncio
async def test_move_to_store(authorized_client, db_session):
    """Тест перемещения товара на витрину"""
    # Получаем sid товара на складе
    query = """
        SELECT sid FROM warehouseitem 
        WHERE status = 'in_stock' AND quantity > 0
        LIMIT 1
    """
    result = await db_session.execute(query)
    item_sid = result.scalar_one_or_none()

    if item_sid:
        response = await authorized_client.post(
            f"{settings.API_V1_STR}/warehouse/to-store",
            data={
                "item_sid": item_sid,
                "quantity": 1,
                "price": 100.0
            }
        )
        assert response.status_code == 200
        json = response.json()
        assert "store_item_sid" in json
    else:
        pytest.skip("No available warehouse items found")