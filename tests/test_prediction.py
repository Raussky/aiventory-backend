import pytest
from datetime import datetime, timedelta

from app.core.config import settings
from app.models.inventory import TimeFrame


@pytest.mark.asyncio
async def test_get_forecast(authorized_client, test_product, mock_prediction_service):
    """Test getting forecast for a product"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}"
    )
    assert response.status_code == 200
    forecasts = response.json()
    assert len(forecasts) > 0
    assert "sid" in forecasts[0]
    assert "timeframe" in forecasts[0]
    assert "period_start" in forecasts[0]
    assert "period_end" in forecasts[0]
    assert "forecast_qty" in forecasts[0]

    # Check prediction service was used
    mock_prediction_service.generate_forecast.assert_called_once()
    mock_prediction_service.save_forecast.assert_called_once()


@pytest.mark.asyncio
async def test_get_forecast_with_refresh(authorized_client, test_product, mock_prediction_service):
    """Test getting forecast with refresh parameter"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}",
        params={"refresh": True}
    )
    assert response.status_code == 200
    forecasts = response.json()
    assert len(forecasts) > 0

    # Check prediction service was used (forced refresh)
    mock_prediction_service.generate_forecast.assert_called_once()
    mock_prediction_service.save_forecast.assert_called_once()


@pytest.mark.asyncio
async def test_get_forecast_with_custom_timeframe(authorized_client, test_product, mock_prediction_service):
    """Test getting forecast with custom timeframe"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}",
        params={"timeframe": "week"}
    )
    assert response.status_code == 200
    forecasts = response.json()
    assert len(forecasts) > 0
    assert forecasts[0]["timeframe"] == "week"

    # Check prediction service was called with right parameters
    mock_prediction_service.generate_forecast.assert_called_once_with(
        product_sid=test_product.sid,
        timeframe=TimeFrame.WEEK,
        periods_ahead=3
    )


@pytest.mark.asyncio
async def test_get_forecast_with_custom_periods(authorized_client, test_product, mock_prediction_service):
    """Test getting forecast with custom number of periods"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}",
        params={"periods": 5}
    )
    assert response.status_code == 200
    forecasts = response.json()
    assert len(forecasts) > 0

    # Check prediction service was called with right parameters
    mock_prediction_service.generate_forecast.assert_called_once_with(
        product_sid=test_product.sid,
        timeframe=TimeFrame.MONTH,  # Default
        periods_ahead=5
    )


@pytest.mark.asyncio
async def test_get_forecast_nonexistent_product(authorized_client):
    """Test getting forecast for nonexistent product"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/nonexistent_sid"
    )
    assert response.status_code == 404
    assert "product not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_forecast_insufficient_data(authorized_client, test_product, mock_prediction_service):
    """Test getting forecast when there's not enough data"""
    # Make prediction service return empty list (not enough data)
    mock_prediction_service.generate_forecast.return_value = []

    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}"
    )
    assert response.status_code == 400
    assert "not enough sales data" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_stats(authorized_client, test_sale):
    """Test getting prediction stats"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/prediction/stats")
    assert response.status_code == 200
    stats = response.json()

    # Check structure
    assert "dates" in stats
    assert "products" in stats
    assert "quantity_data" in stats or "data" in stats
    assert "revenue_data" in stats or "data" in stats


@pytest.mark.asyncio
async def test_get_stats_with_product_filter(authorized_client, test_product, test_sale):
    """Test getting prediction stats filtered by product"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/stats",
        params={"product_sid": test_product.sid}
    )
    assert response.status_code == 200
    stats = response.json()

    # Basic structure checks
    assert "dates" in stats
    assert "products" in stats

    # Product-specific checks
    if stats["products"]:  # If any data available
        product_found = False
        for product in stats["products"]:
            if product["product_sid"] == test_product.sid:
                product_found = True
                break
        assert product_found, "Filtered product not found in results"


@pytest.mark.asyncio
async def test_get_stats_with_date_range(authorized_client, test_sale):
    """Test getting prediction stats with date range filter"""
    start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
    end_date = datetime.utcnow().isoformat()

    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/stats",
        params={
            "start_date": start_date,
            "end_date": end_date
        }
    )
    assert response.status_code == 200
    stats = response.json()

    # Basic structure checks
    assert "dates" in stats
    assert "products" in stats


@pytest.mark.asyncio
async def test_get_stats_no_data(authorized_client, db_session):
    """Test getting prediction stats when no sales data exists"""
    # Clear sales data
    await db_session.execute("DELETE FROM sale")
    await db_session.commit()

    response = await authorized_client.get(f"{settings.API_V1_STR}/prediction/stats")
    assert response.status_code == 200
    stats = response.json()

    # Should return empty data
    assert stats["dates"] == []
    assert stats["products"] == []
    assert "data" in stats or "quantity_data" in stats  # Either of these should exist


@pytest.mark.asyncio
async def test_prediction_model_correctness(authorized_client, test_product, db_session, test_user, test_store_item):
    """Test that prediction model produces reasonable results (basic sanity check)"""
    from app.services.prediction import PredictionService
    from app.models.inventory import Sale, StoreItem
    import uuid
    import nanoid

    # Create some historical sales data
    today = datetime.utcnow().date()

    # Add sales for the last 30 days with an increasing trend
    for i in range(30, 0, -1):
        sale_date = today - timedelta(days=i)
        sale = Sale(
            id=uuid.uuid4(),
            sid=nanoid.generate(size=22),
            store_item_sid=test_store_item.sid,
            sold_qty=i // 2 + 5,  # Increasing trend
            sold_price=100.0,
            sold_at=datetime.combine(sale_date, datetime.min.time()),
            cashier_sid=test_user["sid"]
        )
        db_session.add(sale)

    await db_session.commit()

    # Now create a real prediction service
    prediction_service = PredictionService(db_session)

    # Generate a forecast
    forecasts = await prediction_service.generate_forecast(
        product_sid=test_product.sid,
        timeframe=TimeFrame.DAY,
        periods_ahead=5
    )

    # Basic sanity checks - may vary by implementation
    assert len(forecasts) == 5
    for forecast in forecasts:
        assert forecast["product_sid"] == test_product.sid
        assert forecast["timeframe"] == TimeFrame.DAY
        assert forecast["forecast_qty"] > 0  # Should predict some demand


@pytest.mark.asyncio
async def test_prediction_service_error_handling(authorized_client, test_product, mock_prediction_service):
    """Test error handling in prediction service"""
    # Make prediction service raise an exception
    mock_prediction_service.generate_forecast.side_effect = Exception("Test exception")

    response = await authorized_client.get(
        f"{settings.API_V1_STR}/prediction/forecast/{test_product.sid}"
    )
    assert response.status_code == 400
    assert "could not generate forecast" in response.json()["detail"].lower()