# app/api/v1/warehouse.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any
import datetime

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Product, Category, Upload, WarehouseItem, WarehouseItemStatus, StoreItem, \
    StoreItemStatus, Currency
from app.models.base import Base
from app.schemas.inventory import (
    WarehouseItemResponse, UploadResponse, WarehouseItemCreate
)
from redis.asyncio import Redis
from app.core.dependencies import get_current_user
from app.services.file_parser import detect_and_parse_file
from app.services.barcode import decode_barcode_from_base64
from app.db.redis import get_redis
from app.services.pricing import calculate_store_price, suggest_discount, suggest_warehouse_action

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    try:
        # Создаем запись о загрузке файла
        upload = Upload(
            sid=Base.generate_sid(),
            user_sid=current_user.sid,
            file_name=file.filename,
            uploaded_at=datetime.datetime.utcnow(),
            rows_imported=0,
        )
        db.add(upload)
        await db.commit()
        await db.refresh(upload)

        # Парсим файл
        records = await detect_and_parse_file(file)

        # Обрабатываем данные и создаем записи в базе
        for record in records:
            # Находим или создаем категорию
            category_name = record.get('category', 'Default')
            category_query = await db.execute(
                select(Category).where(Category.name == category_name)
            )
            category = category_query.scalar_one_or_none()

            if not category:
                category = Category(
                    sid=Base.generate_sid(),
                    name=category_name,
                )
                db.add(category)
                await db.commit()
                await db.refresh(category)

            # Находим продукт по имени в данной категории
            product_name = record.get('name')
            product_query = await db.execute(
                select(Product).where(
                    (Product.name == product_name) &
                    (Product.category_sid == category.sid)
                )
            )
            product = product_query.scalar_one_or_none()

            if not product:
                # Создаем новый продукт
                product = Product(
                    sid=Base.generate_sid(),
                    category_sid=category.sid,
                    name=product_name,
                    barcode=str(record.get('barcode')) if record.get('barcode') is not None else None,
                    default_unit=record.get('unit'),
                    default_price=float(record.get('price')) if record.get('price') is not None else None,
                    currency=Currency.KZT,
                    storage_duration=int(record.get('storage_duration', 30)),
                )
                db.add(product)
                await db.commit()
                await db.refresh(product)

            # Преобразуем строковые даты в объекты datetime.date
            expire_date = None
            if record.get('expire_date'):
                if isinstance(record.get('expire_date'), str):
                    expire_date = datetime.datetime.strptime(record.get('expire_date'), '%Y-%m-%d').date()
                else:
                    expire_date = record.get('expire_date')

            received_at = datetime.date.today()
            if record.get('received_at'):
                if isinstance(record.get('received_at'), str):
                    received_at = datetime.datetime.strptime(record.get('received_at'), '%Y-%m-%d').date()
                else:
                    received_at = record.get('received_at')

            # Создаем складскую запись
            warehouse_item = WarehouseItem(
                sid=Base.generate_sid(),
                upload_sid=upload.sid,
                product_sid=product.sid,
                batch_code=record.get('batch_code'),
                quantity=int(record.get('quantity', 0)),
                expire_date=expire_date,
                received_at=received_at,
                status=WarehouseItemStatus.IN_STOCK,
            )
            db.add(warehouse_item)

            # Увеличиваем счетчик импортированных строк
            upload.rows_imported += 1

        await db.commit()
        await db.refresh(upload)

        return upload

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/items", response_model=List[WarehouseItemResponse])
async def get_warehouse_items(
        skip: int = 0,
        limit: int = 100,
        upload_sid: str = None,
        expire_soon: bool = False,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    # Используем selectinload для eager loading связанного продукта И категории
    query = select(WarehouseItem).options(
        selectinload(WarehouseItem.product).selectinload(Product.category)
    ).where(
        WarehouseItem.quantity > 0  # Add filter to exclude zero quantity items
    )

    if upload_sid:
        query = query.where(WarehouseItem.upload_sid == upload_sid)

    if expire_soon:
        # Товары, у которых срок годности заканчивается в течение 7 дней
        expiry_threshold = datetime.date.today() + datetime.timedelta(days=7)
        query = query.where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date >= datetime.date.today(),
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )

    result = await db.execute(
        query.order_by(WarehouseItem.received_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()

    response_items = []
    for item in items:
        response_item = WarehouseItemResponse(
            sid=item.sid,
            product_sid=item.product_sid,
            upload_sid=item.upload_sid,
            batch_code=item.batch_code,
            quantity=item.quantity,
            expire_date=item.expire_date,
            received_at=item.received_at,
            status=item.status,
            product=item.product  # Теперь продукт включает категорию
        )
        response_items.append(response_item)

    return response_items


@router.post("/to-store", response_model=Dict[str, Any])
async def move_to_store(
        barcode_image: str = Form(None),
        item_sid: str = Form(None),
        quantity: int = Form(...),
        price: float = Form(None),  # Сделали опциональным для использования рекомендуемой цены
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    if not barcode_image and not item_sid:
        raise HTTPException(
            status_code=400,
            detail="Either barcode image or item_sid must be provided"
        )

    # Если предоставлено изображение со штрих-кодом, декодируем его
    if barcode_image:
        barcode = await decode_barcode_from_base64(barcode_image)

        # Находим продукт по штрих-коду
        product_query = await db.execute(
            select(Product).options(selectinload(Product.category)).where(Product.barcode == barcode)
        )
        product = product_query.scalar_one_or_none()

        if not product:
            raise HTTPException(status_code=404, detail="Product not found for this barcode")

        # Находим доступную складскую позицию этого товара (в порядке сроков годности)
        warehouse_query = await db.execute(
            select(WarehouseItem)
            .where(
                WarehouseItem.product_sid == product.sid,
                WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
                WarehouseItem.quantity >= quantity
            )
            .order_by(WarehouseItem.expire_date.asc())
        )
        warehouse_item = warehouse_query.scalar_one_or_none()

        if not warehouse_item:
            raise HTTPException(status_code=404, detail="No available items in warehouse")

        item_sid = warehouse_item.sid

    # Теперь работаем с item_sid
    # Устанавливаем лок в Redis для предотвращения гонок
    lock_key = f"lock:item:{item_sid}"
    lock_acquired = await redis.set(lock_key, str(current_user.id), nx=True, ex=5)

    if not lock_acquired:
        raise HTTPException(
            status_code=409,
            detail="Another operation is in progress for this item"
        )

    try:
        warehouse_query = await db.execute(
            select(WarehouseItem)
            .options(selectinload(WarehouseItem.product).selectinload(Product.category))
            .where(WarehouseItem.sid == item_sid)
        )
        warehouse_item = warehouse_query.scalar_one_or_none()

        if not warehouse_item:
            raise HTTPException(status_code=404, detail="Warehouse item not found")

        if warehouse_item.status != WarehouseItemStatus.IN_STOCK:
            raise HTTPException(status_code=400, detail="Item is not available in stock")

        if warehouse_item.quantity < quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough quantity available (requested: {quantity}, available: {warehouse_item.quantity})"
            )

        # Рассчитываем рекомендуемую цену, если не указана пользователем
        product = warehouse_item.product
        category = product.category
        base_price = product.default_price or 0

        suggested_price = calculate_store_price(
            warehouse_item=warehouse_item,
            base_price=base_price,
            category=category
        )

        # Используем предоставленную цену или рекомендуемую
        final_price = price if price is not None else suggested_price

        # Получаем рекомендации по скидкам
        discount_suggestion = suggest_discount(
            warehouse_item=warehouse_item,
            store_price=final_price,
            base_price=base_price,
            category=category
        )

        # Получаем рекомендации по действиям со складом
        warehouse_action = suggest_warehouse_action(
            warehouse_item=warehouse_item,
            category=category
        )

        # Создаем запись в магазине
        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=final_price,
            moved_at=datetime.datetime.utcnow(),
            status=StoreItemStatus.ACTIVE,
        )
        db.add(store_item)

        # Уменьшаем количество на складе
        warehouse_item.quantity -= quantity

        # Если на складе не осталось, меняем статус
        if warehouse_item.quantity == 0:
            warehouse_item.status = WarehouseItemStatus.MOVED

        await db.commit()

        response = {
            "message": "Item moved to store successfully",
            "store_item_sid": store_item.sid,
            "price": final_price
        }

        # Включаем рекомендации по ценам и скидкам в ответ
        if price is None:
            response["suggested_price"] = suggested_price

        if discount_suggestion:
            response["discount_suggestion"] = discount_suggestion

        if warehouse_action:
            response["warehouse_action"] = warehouse_action

        return response

    finally:
        # Освобождаем лок
        await redis.delete(lock_key)