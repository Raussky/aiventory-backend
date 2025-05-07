# app/services/pricing.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.models.inventory import StorageDurationType


def calculate_warehouse_stay_limit(product, category) -> int:
    if not product.storage_duration:
        return 30

    storage_duration_days = product.storage_duration
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = product.storage_duration * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = product.storage_duration * 365

    if category.name.lower() in ["dairy", "milk", "молоко", "молочные", "йогурт", "кисломолочные"]:
        return max(1, storage_duration_days // 2)

    if category.name.lower() in ["bakery", "bread", "хлеб", "выпечка", "хлебобулочные"]:
        return max(1, storage_duration_days // 3)

    if category.name.lower() in ["meat", "fish", "seafood", "мясо", "рыба", "морепродукты"]:
        return max(1, int(storage_duration_days * 2 / 3))

    return max(1, int(storage_duration_days * 0.7))


def calculate_store_price(warehouse_item, base_price, category) -> float:
    if not base_price:
        return 0

    if not warehouse_item.expire_date:
        markup = 1.2
        if category.name.lower() in ["electronics", "appliances", "электроника", "техника"]:
            markup = 1.3
        elif category.name.lower() in ["grocery", "food", "dairy", "продукты", "молочные"]:
            markup = 1.25
        return round(base_price * markup, 2)

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    if days_until_expiry < 0:
        return 0

    product = warehouse_item.product

    storage_duration_days = product.storage_duration or 30
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    shelf_life_remaining = days_until_expiry / storage_duration_days

    if shelf_life_remaining > 0.8:
        markup = 1.35
    elif shelf_life_remaining > 0.5:
        markup = 1.25
    elif shelf_life_remaining > 0.3:
        markup = 1.15
    else:
        markup = 1.05

    return round(base_price * markup, 2)


def suggest_discount(warehouse_item, store_price, base_price, category) -> Optional[Dict[str, Any]]:
    if not warehouse_item.expire_date or not store_price or not base_price:
        return None

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    if days_until_expiry < 0:
        return None

    product = warehouse_item.product

    storage_duration_days = product.storage_duration or 30
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    if days_until_expiry > storage_duration_days * 0.3:
        return None

    if days_until_expiry <= 2:
        discount_percent = 50
    elif days_until_expiry <= 5:
        discount_percent = 30
    elif days_until_expiry <= storage_duration_days * 0.2:
        discount_percent = 20
    else:
        discount_percent = 10

    discounted_price = store_price * (1 - discount_percent / 100)

    if discounted_price < base_price:
        max_discount = ((store_price - base_price) / store_price) * 100
        discount_percent = min(discount_percent, max_discount)
        discounted_price = store_price * (1 - discount_percent / 100)

    return {
        "discount_percent": round(discount_percent, 1),
        "discounted_price": round(discounted_price, 2),
        "days_until_expiry": days_until_expiry
    }


def suggest_warehouse_action(warehouse_item, category) -> Optional[Dict[str, Any]]:
    if not warehouse_item.expire_date:
        return None

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days
    days_in_warehouse = (today - warehouse_item.received_at).days

    product = warehouse_item.product

    storage_duration_days = product.storage_duration or 30
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    stay_limit = calculate_warehouse_stay_limit(product, category)

    if days_in_warehouse >= stay_limit:
        return {
            "action": "move_to_store",
            "urgency": "high" if days_in_warehouse > stay_limit * 1.2 else "medium",
            "reason": f"Товар находится на складе {days_in_warehouse} дней (лимит: {stay_limit} дней)"
        }

    if 0 < days_until_expiry <= 7:
        if days_until_expiry <= 3:
            return {
                "action": "move_to_store_with_discount",
                "urgency": "high",
                "discount_suggestion": 30,
                "reason": f"Срок годности товара истекает через {days_until_expiry} дней"
            }
        else:
            return {
                "action": "move_to_store",
                "urgency": "medium",
                "reason": f"Срок годности товара истекает через {days_until_expiry} дней"
            }

    return None