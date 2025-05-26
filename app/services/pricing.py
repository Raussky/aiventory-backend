from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.models.inventory import StorageDurationType
import math


def calculate_total_storage_days(product) -> int:
    """Calculate total storage days based on product configuration"""
    storage_duration_days = product.storage_duration or 30

    if product.storage_duration_type == StorageDurationType.DAY:
        return storage_duration_days
    elif product.storage_duration_type == StorageDurationType.MONTH:
        return storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        return storage_duration_days * 365

    return storage_duration_days


def calculate_warehouse_stay_limit(product, category) -> int:
    """Calculate optimal warehouse stay duration"""
    total_days = calculate_total_storage_days(product)
    category_name = category.name.lower()

    # Category-specific warehouse limits
    if any(word in category_name for word in ["dairy", "milk", "молоко", "молочные", "йогурт", "кисломолочные"]):
        return max(3, total_days // 4)
    elif any(word in category_name for word in ["bakery", "bread", "хлеб", "выпечка", "хлебобулочные", "торты", "пирожные"]):
        return max(2, total_days // 5)
    elif any(word in category_name for word in ["meat", "fish", "seafood", "мясо", "рыба", "морепродукты", "птица"]):
        return max(3, total_days // 3)
    elif any(word in category_name for word in ["vegetables", "fruits", "овощи", "фрукты", "зелень"]):
        return max(3, total_days // 4)
    elif any(word in category_name for word in ["напитки", "drinks", "beverages", "соки", "juice"]):
        return max(7, int(total_days * 0.6))
    else:
        return max(7, int(total_days * 0.5))


def calculate_store_price(warehouse_item, base_price, category) -> float:
    """Calculate optimal store price with smart markup"""
    if not base_price:
        return 0

    category_name = category.name.lower()

    # Base markup by category
    base_markup_map = {
        "electronics": 1.25,
        "электроника": 1.25,
        "техника": 1.25,
        "appliances": 1.25,
        "grocery": 1.30,
        "продукты": 1.30,
        "food": 1.30,
        "dairy": 1.35,
        "молочные": 1.35,
        "meat": 1.40,
        "мясо": 1.40,
        "fish": 1.40,
        "рыба": 1.40,
        "vegetables": 1.45,
        "овощи": 1.45,
        "fruits": 1.45,
        "фрукты": 1.45,
        "bakery": 1.50,
        "хлеб": 1.50,
        "выпечка": 1.50,
        "напитки": 1.35,
        "drinks": 1.35,
        "beverages": 1.35,
        "алкоголь": 1.60,
        "alcohol": 1.60,
        "косметика": 1.80,
        "cosmetics": 1.80,
        "бытовая химия": 1.40,
        "household": 1.40,
        "канцтовары": 1.50,
        "stationery": 1.50,
    }

    base_markup = 1.25
    for keyword, markup in base_markup_map.items():
        if keyword in category_name:
            base_markup = markup
            break

    # If no expiry date, use base markup
    if not warehouse_item.expire_date:
        return round(base_price * base_markup, 2)

    # Calculate price based on shelf life remaining
    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    if days_until_expiry <= 0:
        return round(base_price * 0.5, 2)  # 50% of base price for expired items

    total_shelf_life = calculate_total_storage_days(warehouse_item.product)
    shelf_life_remaining = days_until_expiry / total_shelf_life

    # Progressive markup reduction based on shelf life
    if shelf_life_remaining > 0.9:
        markup_multiplier = 1.0
    elif shelf_life_remaining > 0.8:
        markup_multiplier = 0.98
    elif shelf_life_remaining > 0.7:
        markup_multiplier = 0.95
    elif shelf_life_remaining > 0.6:
        markup_multiplier = 0.92
    elif shelf_life_remaining > 0.5:
        markup_multiplier = 0.88
    elif shelf_life_remaining > 0.4:
        markup_multiplier = 0.83
    elif shelf_life_remaining > 0.3:
        markup_multiplier = 0.75
    elif shelf_life_remaining > 0.2:
        markup_multiplier = 0.65
    elif shelf_life_remaining > 0.1:
        markup_multiplier = 0.55
    else:
        markup_multiplier = 0.45

    final_price = base_price * base_markup * markup_multiplier

    # Ensure minimum profit margin
    min_price = base_price * 1.05  # At least 5% markup
    if final_price < min_price:
        final_price = min_price

    return round(final_price, 2)


def suggest_discount(warehouse_item, store_price, base_price, category) -> Optional[Dict[str, Any]]:
    """Smart discount suggestion with detailed calculations"""
    if not warehouse_item.expire_date or not store_price or not base_price:
        return None

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    if days_until_expiry <= 0:
        return {
            "discount_percent": 50.0,
            "discounted_price": round(store_price * 0.5, 2),
            "days_until_expiry": 0,
            "urgency": "critical",
            "reason": "Срок годности истек",
            "profit_margin": -50.0,
            "recommendation": "Срочно списать или утилизировать"
        }

    total_shelf_life = calculate_total_storage_days(warehouse_item.product)
    shelf_life_remaining = days_until_expiry / total_shelf_life

    category_name = category.name.lower()

    # Category-specific discount thresholds
    if any(word in category_name for word in ["dairy", "молочные", "meat", "мясо", "fish", "рыба"]):
        discount_threshold = 0.25
    elif any(word in category_name for word in ["bakery", "хлеб", "выпечка"]):
        discount_threshold = 0.20
    elif any(word in category_name for word in ["vegetables", "овощи", "fruits", "фрукты"]):
        discount_threshold = 0.30
    else:
        discount_threshold = 0.35

    if shelf_life_remaining > discount_threshold:
        return None

    # Calculate optimal discount percentage
    if days_until_expiry <= 1:
        discount_percent = 40  # Max 40% for last day
        urgency = "critical"
    elif days_until_expiry <= 3:
        discount_percent = 35
        urgency = "high"
    elif days_until_expiry <= 5:
        discount_percent = 30
        urgency = "high"
    elif days_until_expiry <= 7:
        discount_percent = 25
        urgency = "medium"
    elif shelf_life_remaining <= 0.15:
        discount_percent = 20
        urgency = "medium"
    elif shelf_life_remaining <= 0.25:
        discount_percent = 15
        urgency = "low"
    else:
        discount_percent = 10
        urgency = "low"

    # Calculate discounted price and profit margin
    discounted_price = store_price * (1 - discount_percent / 100)
    min_price = base_price * 1.1  # Minimum 10% profit margin

    # Ensure minimum profitability
    if discounted_price < min_price:
        discounted_price = min_price
        discount_percent = round((1 - discounted_price / store_price) * 100, 1)

    # Calculate actual profit margin
    profit_margin = ((discounted_price - base_price) / base_price) * 100

    # Detailed calculation explanation
    calculation_details = {
        "base_price": base_price,
        "store_price": store_price,
        "initial_markup": round((store_price / base_price - 1) * 100, 1),
        "suggested_discount": discount_percent,
        "discounted_price": round(discounted_price, 2),
        "final_profit_margin": round(profit_margin, 1),
        "min_acceptable_price": round(min_price, 2)
    }

    return {
        "discount_percent": round(discount_percent, 1),
        "discounted_price": round(discounted_price, 2),
        "days_until_expiry": days_until_expiry,
        "urgency": urgency,
        "shelf_life_remaining": round(shelf_life_remaining * 100, 1),
        "reason": f"Осталось {round(shelf_life_remaining * 100)}% срока годности",
        "profit_margin": round(profit_margin, 1),
        "calculation_details": calculation_details,
        "recommendation": get_discount_recommendation(urgency, profit_margin, days_until_expiry)
    }


def get_discount_recommendation(urgency: str, profit_margin: float, days_until_expiry: int) -> str:
    """Get actionable recommendation based on urgency and profit margin"""
    if urgency == "critical":
        if profit_margin < 5:
            return "Критическая ситуация! Продать любой ценой или списать"
        else:
            return f"Срочная распродажа! Осталось {days_until_expiry} дней"
    elif urgency == "high":
        if profit_margin < 10:
            return "Рекомендуется активная промоакция для быстрой реализации"
        else:
            return "Установить скидку и разместить на видном месте"
    elif urgency == "medium":
        return "Планово установить скидку для стимулирования продаж"
    else:
        return "Можно установить небольшую скидку для привлечения покупателей"


def suggest_warehouse_action(warehouse_item, category) -> Optional[Dict[str, Any]]:
    """Suggest warehouse actions with detailed reasoning"""
    if not warehouse_item.expire_date:
        return {
            "action": "check_expiry",
            "urgency": "medium",
            "reason": "Отсутствует информация о сроке годности",
            "recommendation": "Проверить и добавить срок годности"
        }

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days
    days_in_warehouse = (today - warehouse_item.received_at).days

    product = warehouse_item.product
    total_shelf_life = calculate_total_storage_days(product)
    stay_limit = calculate_warehouse_stay_limit(product, category)

    if days_until_expiry <= 0:
        return {
            "action": "dispose",
            "urgency": "critical",
            "reason": "Срок годности истек",
            "days_expired": abs(days_until_expiry),
            "recommendation": "Немедленно списать и утилизировать"
        }

    shelf_life_remaining = days_until_expiry / total_shelf_life
    warehouse_time_percent = days_in_warehouse / stay_limit if stay_limit > 0 else 1.0

    category_name = category.name.lower()

    # Category-specific thresholds
    if any(word in category_name for word in ["dairy", "молочные", "meat", "мясо", "fish", "рыба"]):
        critical_threshold = 0.15
        urgent_threshold = 0.25
        medium_threshold = 0.35
    else:
        critical_threshold = 0.10
        urgent_threshold = 0.20
        medium_threshold = 0.30

    # Determine action based on multiple factors
    if shelf_life_remaining <= critical_threshold or days_until_expiry <= 3:
        optimal_discount = min(40, round((1 - shelf_life_remaining) * 50))
        return {
            "action": "move_to_store_urgent",
            "urgency": "critical",
            "discount_suggestion": optimal_discount,
            "reason": f"Критически мало времени до истечения срока ({days_until_expiry} дней)",
            "days_until_expiry": days_until_expiry,
            "shelf_life_remaining": round(shelf_life_remaining * 100, 1),
            "recommendation": f"Немедленно переместить в магазин со скидкой {optimal_discount}%"
        }

    if warehouse_time_percent >= 1.2:
        return {
            "action": "move_to_store",
            "urgency": "high",
            "reason": f"Превышен лимит хранения на складе ({days_in_warehouse} дней из {stay_limit})",
            "days_in_warehouse": days_in_warehouse,
            "warehouse_limit": stay_limit,
            "recommendation": "Переместить в магазин для оптимизации оборота"
        }

    if shelf_life_remaining <= urgent_threshold:
        optimal_discount = round((1 - shelf_life_remaining) * 25)
        return {
            "action": "move_to_store_with_discount",
            "urgency": "high",
            "discount_suggestion": optimal_discount,
            "reason": f"Приближается окончание срока годности ({days_until_expiry} дней)",
            "days_until_expiry": days_until_expiry,
            "shelf_life_remaining": round(shelf_life_remaining * 100, 1),
            "recommendation": f"Переместить в магазин со скидкой {optimal_discount}%"
        }

    if warehouse_time_percent >= 0.8 or shelf_life_remaining <= medium_threshold:
        return {
            "action": "plan_to_move",
            "urgency": "medium",
            "reason": f"Рекомендуется планировать перемещение в магазин",
            "days_until_expiry": days_until_expiry,
            "days_in_warehouse": days_in_warehouse,
            "shelf_life_remaining": round(shelf_life_remaining * 100, 1),
            "recommendation": "Запланировать перемещение на ближайшие дни"
        }

    return {
        "action": "monitor",
        "urgency": "low",
        "reason": "Товар в хорошем состоянии",
        "days_until_expiry": days_until_expiry,
        "shelf_life_remaining": round(shelf_life_remaining * 100, 1),
        "recommendation": "Продолжать мониторинг"
    }