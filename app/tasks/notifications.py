# app/tasks/notifications.py
from celery import shared_task
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from app.models.inventory import WarehouseItem, StoreItem, WarehouseItemStatus, StoreItemStatus
from app.models.users import User
from app.core.config import settings
from app.services.email import send_expiry_notification


@shared_task
def check_expiring_items():
    """Проверяет товары, у которых скоро истекает срок годности, и отправляет уведомления"""
    # Создаем соединение с БД
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Проверяем товары на складе
        today = datetime.now().date()
        expiry_threshold = today + timedelta(days=7)

        # Товары на складе
        warehouse_query = select(WarehouseItem).join(Product).where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date > today,
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )

        warehouse_items = db.execute(warehouse_query).scalars().all()

        # Товары в магазине
        store_query = select(StoreItem).join(WarehouseItem).join(Product).where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date > today,
            StoreItem.status == StoreItemStatus.ACTIVE
        )

        store_items = db.execute(store_query).scalars().all()

        # Получаем всех admin/owner пользователей для уведомлений
        user_query = select(User).where(User.role.in_(['owner', 'admin']))
        users = db.execute(user_query).scalars().all()

        # Отправляем уведомления
        for user in users:
            if warehouse_items or store_items:
                send_expiry_notification.delay(
                    user_email=user.email,
                    warehouse_items=[{
                        "product_name": item.product.name,
                        "quantity": item.quantity,
                        "expire_date": item.expire_date.strftime("%Y-%m-%d")
                    } for item in warehouse_items],
                    store_items=[{
                        "product_name": item.warehouse_item.product.name,
                        "quantity": item.quantity,
                        "expire_date": item.warehouse_item.expire_date.strftime("%Y-%m-%d"),
                        "price": item.price
                    } for item in store_items]
                )

        # Обновляем статус для просроченных товаров в магазине
        expired_query = select(StoreItem).join(WarehouseItem).where(
            WarehouseItem.expire_date < today,
            StoreItem.status == StoreItemStatus.ACTIVE
        )

        expired_items = db.execute(expired_query).scalars().all()

        for item in expired_items:
            item.status = StoreItemStatus.EXPIRED

        db.commit()

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@shared_task
def update_predictions():
    """Периодически обновляет прогнозы для всех товаров"""
    # Создаем соединение с БД
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Получаем все продукты
        products = db.execute(select(Product)).scalars().all()

        for product in products:
            # Запускаем задачу на обновление прогноза для каждого продукта
            generate_product_forecast.delay(product.sid)

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


@shared_task
def generate_product_forecast(product_sid: str):
    """Генерирует прогноз для конкретного продукта"""
    # Создаем соединение с БД
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Инициализируем сервис прогнозирования
        from app.services.prediction import PredictionService
        prediction_service = PredictionService(db)

        # Генерируем прогнозы для разных временных интервалов
        for timeframe in [TimeFrame.DAY, TimeFrame.WEEK, TimeFrame.MONTH]:
            periods = 7 if timeframe == TimeFrame.DAY else 4 if timeframe == TimeFrame.WEEK else 3

            forecasts = prediction_service.generate_forecast(
                product_sid=product_sid,
                timeframe=timeframe,
                periods_ahead=periods
            )

            if forecasts:
                prediction_service.save_forecast(forecasts)

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()