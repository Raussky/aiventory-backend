import pytest

from app.core.config import settings


@pytest.mark.asyncio
async def test_get_forecast(authorized_client, db_session):
    """Тест получения прогноза для товара"""
    # Получаем sid продукта
    query = """
        SELECT sid FROM product 
        LIMIT 1
    """
    result = await db_session.execute(query)
    product_sid = result.scalar_one_or_none()

    if product_sid:
        response = await authorized_client.get(
            f"{settings.API_V1_STR}/prediction/forecast/{product_sid}"
        )
        assert response.status_code in [200, 400]  # 400 если недостаточно данных для прогноза
    else:
        pytest.skip("No products found")


@pytest.mark.asyncio
async def test_get_stats(authorized_client):
    """Тест получения статистики продаж"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/prediction/stats")
    assert response.status_code == 200
    json = response.json()
    assert "dates" in json
    assert "products" in json
    assert "quantity_data" in json or "data" in json
    assert "revenue_data" in json or "data" in json