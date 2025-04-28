# app/services/pricing.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.models.inventory import StorageDurationType


def calculate_warehouse_stay_limit(product, category) -> int:
    """
    Рассчитывает, сколько дней товар может оставаться на складе
    на основе срока хранения и категории
    """
    if not product.storage_duration:
        return 30  # По умолчанию, если срок хранения не указан

    # Convert storage_duration to days based on type
    storage_duration_days = product.storage_duration
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = product.storage_duration * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = product.storage_duration * 365

    # Для скоропортящихся товаров (молоко и т.д.) - не более половины срока хранения
    if category.name.lower() in ["dairy", "milk", "молоко", "молочные", "йогурт", "кисломолочные"]:
        return max(1, storage_duration_days // 2)

    # Для хлебобулочных изделий - не более 1/3 срока хранения
    if category.name.lower() in ["bakery", "bread", "хлеб", "выпечка", "хлебобулочные"]:
        return max(1, storage_duration_days // 3)

    # Для мяса и рыбы - не более 2/3 срока хранения
    if category.name.lower() in ["meat", "fish", "seafood", "мясо", "рыба", "морепродукты"]:
        return max(1, int(storage_duration_days * 2 / 3))

    # Для остальных товаров - до 70% срока хранения
    return max(1, int(storage_duration_days * 0.7))


def calculate_store_price(warehouse_item, base_price, category) -> float:
    """
    Рассчитывает рекомендуемую цену для товара на витрине на основе:
    - Базовая цена (стандартная для продукта или указанная)
    - Категория
    - Оставшиеся дни до истечения срока годности
    - Срок хранения

    Возвращает рекомендуемую цену, не приводящую к убыткам.
    """
    if not base_price:
        return 0

    # Если нет срока годности, используем базовую цену с наценкой
    if not warehouse_item.expire_date:
        # Добавляем наценку 20-30% в зависимости от категории
        markup = 1.2  # Наценка по умолчанию
        if category.name.lower() in ["electronics", "appliances", "электроника", "техника"]:
            markup = 1.3
        elif category.name.lower() in ["grocery", "food", "dairy", "продукты", "молочные"]:
            markup = 1.25
        return round(base_price * markup, 2)

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    # Если уже просрочен, нет рекомендуемой цены
    if days_until_expiry < 0:
        return 0

    product = warehouse_item.product

    # Convert storage_duration to days based on type
    storage_duration_days = product.storage_duration or 30  # По умолчанию 30 дней
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    # Рассчитываем процент оставшегося срока годности
    shelf_life_remaining = days_until_expiry / storage_duration_days

    # Корректируем наценку в зависимости от оставшегося срока годности
    if shelf_life_remaining > 0.8:  # Более 80% срока годности осталось
        markup = 1.35  # Максимальная наценка
    elif shelf_life_remaining > 0.5:  # Более 50% срока годности осталось
        markup = 1.25  # Стандартная наценка
    elif shelf_life_remaining > 0.3:  # Более 30% срока годности осталось
        markup = 1.15  # Пониженная наценка
    else:  # Менее 30% срока годности осталось
        markup = 1.05  # Минимальная наценка

    return round(base_price * markup, 2)


def suggest_discount(warehouse_item, store_price, base_price, category) -> Optional[Dict[str, Any]]:
    """
    Предлагает скидку для товаров с истекающим сроком годности.
    Возвращает процент скидки и рекомендуемую акционную цену.
    """
    if not warehouse_item.expire_date or not store_price or not base_price:
        return None

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days

    # Если уже просрочен, нет рекомендации по скидке
    if days_until_expiry < 0:
        return None

    product = warehouse_item.product

    # Convert storage_duration to days based on type
    storage_duration_days = product.storage_duration or 30  # По умолчанию, если не указано
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    # Не предлагаем скидку, если осталось много времени
    if days_until_expiry > storage_duration_days * 0.3:
        return None

    # Рассчитываем скидку на основе оставшегося срока годности
    if days_until_expiry <= 2:  # 2 дня или меньше
        discount_percent = 50  # 50% скидка для скоро истекающих товаров
    elif days_until_expiry <= 5:  # 5 дней или меньше
        discount_percent = 30  # 30% скидка для товаров с истекающим сроком
    elif days_until_expiry <= storage_duration_days * 0.2:  # Осталось меньше 20% срока
        discount_percent = 20  # 20% скидка
    else:  # Осталось меньше 30% срока
        discount_percent = 10  # 10% скидка

    # Убеждаемся, что со скидкой цена не будет ниже базовой
    discounted_price = store_price * (1 - discount_percent / 100)

    # Если цена со скидкой будет ниже базовой, корректируем скидку
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
    """
    Предлагает действия для товаров на складе на основе:
    - Дни на складе
    - Срок годности
    - Срок хранения

    Возвращает рекомендуемое действие (переместить в магазин, применить скидку и т.д.)
    """
    if not warehouse_item.expire_date:
        return None

    today = datetime.now().date()
    days_until_expiry = (warehouse_item.expire_date - today).days
    days_in_warehouse = (today - warehouse_item.received_at).days

    product = warehouse_item.product

    # Convert storage_duration to days based on type
    storage_duration_days = product.storage_duration or 30  # По умолчанию
    if product.storage_duration_type == StorageDurationType.MONTH:
        storage_duration_days = storage_duration_days * 30
    elif product.storage_duration_type == StorageDurationType.YEAR:
        storage_duration_days = storage_duration_days * 365

    # Рассчитываем максимальный срок хранения на складе
    stay_limit = calculate_warehouse_stay_limit(product, category)

    # Если товар находится на складе слишком долго
    if days_in_warehouse >= stay_limit:
        return {
            "action": "move_to_store",
            "urgency": "high" if days_in_warehouse > stay_limit * 1.2 else "medium",
            "reason": f"Товар находится на складе {days_in_warehouse} дней (лимит: {stay_limit} дней)"
        }

    # Если до истечения срока годности осталось мало времени
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

    # Если срочных действий не требуется
    return None