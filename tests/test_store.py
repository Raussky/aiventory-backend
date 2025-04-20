import pytest
from datetime import datetime, timedelta
import json

from app.core.config import settings
from app.models.inventory import StoreItemStatus


@pytest.mark.asyncio
async def test_get_store_items(authorized_client, test_store_item):
    """Test retrieving store items"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/store/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert "sid" in items[0]
    assert "product" in items[0]
    assert items[0]["sid"] == test_store_item.sid


@pytest.mark.asyncio
async def test_filter_store_items_by_status(authorized_client, test_store_item):
    """Test filtering store items by status"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/store/items",
        params={"status": "active"}
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) > 0
    assert items[0]["status"] == "active"

    # Test with different status
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/store/items",
        params={"status": "expired"}
    )
    assert response.status_code == 200
    # May be empty, but should be a valid response


@pytest.mark.asyncio
async def test_get_store_items_with_pagination(authorized_client, test_store_item):
    """Test pagination in store items endpoint"""
    response = await authorized_client.get(
        f"{settings.API_V1_STR}/store/items",
        params={"skip": 0, "limit": 5}
    )
    assert response.status_code == 200
    items = response.json()
    assert len(items) <= 5


@pytest.mark.asyncio
async def test_create_discount_success(authorized_client, test_store_item, test_user, mock_redis):
    """Test creating a discount successfully"""
    now = datetime.utcnow()
    discount_data = {
        "store_item_sid": test_store_item.sid,
        "percentage": 15.0,
        "starts_at": now.isoformat(),
        "ends_at": (now + timedelta(days=7)).isoformat()
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/discount",
        json=discount_data
    )
    assert response.status_code == 200
    result = response.json()
    assert result["store_item_sid"] == test_store_item.sid
    assert result["percentage"] == 15.0
    assert "sid" in result
    assert "created_by_sid" in result

    # Verify cache invalidation
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_create_discount_invalid_percentage(authorized_client, test_store_item):
    """Test creating a discount with invalid percentage"""
    now = datetime.utcnow()
    # Try with negative percentage
    discount_data = {
        "store_item_sid": test_store_item.sid,
        "percentage": -5.0,  # Negative
        "starts_at": now.isoformat(),
        "ends_at": (now + timedelta(days=7)).isoformat()
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/discount",
        json=discount_data
    )
    assert response.status_code == 422  # Validation error

    # Try with percentage over 100
    discount_data["percentage"] = 110.0
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/discount",
        json=discount_data
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_discount_overlapping(authorized_client, db_session, test_store_item, test_user):
    """Test creating a discount that overlaps with existing one"""
    from app.models.inventory import Discount
    import uuid
    import nanoid

    # Create an existing discount
    now = datetime.utcnow()
    existing_discount = Discount(
        id=uuid.uuid4(),
        sid=nanoid.generate(size=22),
        store_item_sid=test_store_item.sid,
        percentage=10.0,
        starts_at=now,
        ends_at=now + timedelta(days=7),
        created_by_sid=test_user["sid"]
    )
    db_session.add(existing_discount)
    await db_session.commit()

    # Try to create an overlapping discount
    discount_data = {
        "store_item_sid": test_store_item.sid,
        "percentage": 15.0,
        "starts_at": (now + timedelta(days=3)).isoformat(),  # Overlaps
        "ends_at": (now + timedelta(days=10)).isoformat()
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/discount",
        json=discount_data
    )
    assert response.status_code == 400
    assert "overlapping" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_expire_item(authorized_client, test_store_item, mock_redis):
    """Test marking an item as expired"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/expire/{test_store_item.sid}"
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sid"] == test_store_item.sid
    assert result["status"] == "expired"

    # Verify cache invalidation
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_expire_nonexistent_item(authorized_client):
    """Test trying to expire a nonexistent item"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/expire/nonexistent_sid"
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_remove_item(authorized_client, test_store_item, mock_redis):
    """Test removing an item from store"""
    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/remove/{test_store_item.sid}"
    )
    assert response.status_code == 200
    result = response.json()
    assert result["sid"] == test_store_item.sid
    assert result["status"] == "removed"

    # Verify cache invalidation
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_record_sale_success(authorized_client, test_store_item, mock_redis):
    """Test recording a sale successfully"""
    sale_data = {
        "store_item_sid": test_store_item.sid,
        "sold_qty": 1,
        "sold_price": 130.0
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/sales",
        json=sale_data
    )
    assert response.status_code == 200
    result = response.json()
    assert result["store_item_sid"] == test_store_item.sid
    assert result["sold_qty"] == 1
    assert result["sold_price"] == 130.0
    assert "sid" in result
    assert "sold_at" in result
    assert "cashier_sid" in result

    # Verify cache invalidation
    mock_redis.delete.assert_called_once()

    # Verify pub/sub notification
    mock_redis.publish.assert_called_once()
    # Check the format of published message
    channel, message = mock_redis.publish.call_args[0]
    assert "sales:" in channel
    message_data = json.loads(message)
    assert message_data["type"] == "new_sale"
    assert "product_name" in message_data
    assert message_data["quantity"] == 1
    assert message_data["price"] == 130.0


@pytest.mark.asyncio
async def test_record_sale_insufficient_quantity(authorized_client, db_session, test_store_item):
    """Test recording a sale with insufficient quantity"""
    # Set quantity to 1
    await db_session.execute(
        f"UPDATE storeitem SET quantity = 1 WHERE sid = '{test_store_item.sid}'"
    )
    await db_session.commit()

    # Try to sell more than available
    sale_data = {
        "store_item_sid": test_store_item.sid,
        "sold_qty": 2,  # More than available
        "sold_price": 130.0
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/sales",
        json=sale_data
    )
    assert response.status_code == 400
    assert "not enough quantity" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_record_sale_auto_remove(authorized_client, db_session, test_store_item, mock_redis):
    """Test that store item is auto-removed when quantity reaches zero"""
    # Set quantity to 1
    await db_session.execute(
        f"UPDATE storeitem SET quantity = 1 WHERE sid = '{test_store_item.sid}'"
    )
    await db_session.commit()

    # Sell the last item
    sale_data = {
        "store_item_sid": test_store_item.sid,
        "sold_qty": 1,
        "sold_price": 130.0
    }

    response = await authorized_client.post(
        f"{settings.API_V1_STR}/store/sales",
        json=sale_data
    )
    assert response.status_code == 200

    # Check item status was updated
    result = await db_session.execute(
        f"SELECT status FROM storeitem WHERE sid = '{test_store_item.sid}'"
    )
    status = result.scalar_one()
    assert status == StoreItemStatus.REMOVED


@pytest.mark.asyncio
async def test_get_reports(authorized_client, test_sale, mock_redis):
    """Test getting store reports"""
    response = await authorized_client.get(f"{settings.API_V1_STR}/store/reports")
    assert response.status_code == 200
    result = response.json()

    # Check structure
    assert "period" in result
    assert "sales" in result
    assert "discounts" in result
    assert "expired" in result
    assert "summary" in result

    # Check summary fields
    summary = result["summary"]
    assert "total_sales" in summary
    assert "total_items_sold" in summary
    assert "total_expired_value" in summary
    assert "total_expired_items" in summary
    assert "total_discount_savings" in summary


@pytest.mark.asyncio
async def test_get_reports_with_date_filter(authorized_client, test_sale):
    """Test getting store reports with date filtering"""
    start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()
    end_date = datetime.utcnow().isoformat()

    response = await authorized_client.get(
        f"{settings.API_V1_STR}/store/reports",
        params={
            "start_date": start_date,
            "end_date": end_date
        }
    )
    assert response.status_code == 200
    result = response.json()

    # Check period is respected
    assert result["period"]["start_date"] == start_date
    assert result["period"]["end_date"] == end_date


@pytest.mark.asyncio
async def test_reports_caching(authorized_client, test_sale, mock_redis):
    """Test that reports are cached"""
    # Set mock to return a cached result
    mock_redis.get.return_value = json.dumps({
        "period": {
            "start_date": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end_date": datetime.utcnow().isoformat()
        },
        "sales": [],
        "discounts": [],
        "expired": [],
        "summary": {
            "total_sales": 0,
            "total_items_sold": 0,
            "total_expired_value": 0,
            "total_expired_items": 0,
            "total_discount_savings": 0
        }
    })

    response = await authorized_client.get(f"{settings.API_V1_STR}/store/reports")
    assert response.status_code == 200

    # Verify cache was checked
    mock_redis.get.assert_called_once()
    # Verify no cache set since we got a hit
    assert not mock_redis.set.called