import pytest
from datetime import datetime, timedelta

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_store_items(authorized_client):
    """Тест получения списка товаров в магазине"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/store/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert "sid" in items[0]
    assert "product" in items[0]


@pytest.mark.asyncio
async def test_create_discount(authorized_client, db_session):
    """Тест создания скидки на товар"""
    # Получаем sid товара в магазине
    query = """
        SELECT sid FROM storeitem 
        WHERE status = 'active'
        LIMIT 1
    """
    result = await db_session.execute(query)
    store_item_sid = result.scalar_one_or_none()

    if store_item_sid:
        now = datetime.utcnow()
        discount_data = {
            "store_item_sid": store_item_sid,
            "percentage": 10.0,
            "starts_at": now.isoformat(),
            "ends_at": (now + timedelta(days=7)).isoformat()
        }

        response = await authorized_client.post(
            f"{settings.API_V1_STR}/store/discount",
            json=discount_data
        )
        assert response.status_code == 200
        json = response.json()
        assert json["store_item_sid"] == store_item_sid
        assert json["percentage"] == 10.0
    else:
        pytest.skip("No available store items found")


@pytest.mark.asyncio
async def test_expire_item(authorized_client, db_session):
    """Тест отметки товара как просроченного"""
    # Получаем sid товара в магазине
    query = """
        SELECT sid FROM storeitem 
        WHERE status = 'active'
        LIMIT 1
    """
    result = await db_session.execute(query)
    store_item_sid = result.scalar_one_or_none()

    if store_item_sid:
        response = await authorized_client.post(
            f"{settings.API_V1_STR}/store/expire/{store_item_sid}"
        )
        assert response.status_code == 200
        json = response.json()
        assert json["status"] == "expired"
    else:
        pytest.skip("No available store items found")


@pytest.mark.asyncio
async def test_record_sale(authorized_client, db_session):
    """Тест регистрации продажи"""
    # Получаем sid товара в магазине
    query = """
        SELECT sid FROM storeitem 
        WHERE status = 'active' AND quantity > 0
        LIMIT 1
    """
    result = await db_session.execute(query)
    store_item_sid = result.scalar_one_or_none()

    if store_item_sid:
        sale_data = {
            "store_item_sid": store_item_sid,
            "sold_qty": 1,
            "sold_price": 150.0
        }

        response = await authorized_client.post(
            f"{settings.API_V1_STR}/store/sales",
            json=sale_data
        )
        assert response.status_code == 200
        json = response.json()
        assert json["store_item_sid"] == store_item_sid
        assert json["sold_qty"] == 1
        assert json["sold_price"] == 150.0
    else:
        pytest.skip("No available store items found")


@pytest.mark.asyncio
async def test_get_reports(authorized_client):
    """Тест получения отчетов о продажах"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/store/reports")
    assert response.status_code == 200
    json = response.json()
    assert "summary" in json
    assert "sales" in json
    assert "expired" in json
    assert "discounts" in json