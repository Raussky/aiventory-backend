# app/api/v1/warehouse.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any
import datetime

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Product, Category, Upload, WarehouseItem
from app.schemas.inventory import (
    WarehouseItemResponse, UploadResponse, WarehouseItemCreate
)
from redis.asyncio import Redis
from app.core.dependencies import get_current_user
from app.services.file_parser import detect_and_parse_file
from app.services.barcode import decode_barcode_from_base64
from app.db.redis import get_redis
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
            # Находим или создаем продукт по штрих-коду/имени
            product_query = await db.execute(
                select(Product).where(
                    (Product.barcode == record.get('barcode')) |
                    (Product.name == record.get('name'))
                )
            )
            product = product_query.scalar_one_or_none()

            if not product:
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

                # Создаем новый продукт
                product = Product(
                    sid=Base.generate_sid(),
                    category_sid=category.sid,
                    name=record.get('name'),
                    barcode=record.get('barcode'),
                    default_unit=record.get('unit'),
                    default_price=record.get('price'),
                )
                db.add(product)
                await db.commit()
                await db.refresh(product)

            # Создаем складскую запись
            warehouse_item = WarehouseItem(
                sid=Base.generate_sid(),
                upload_sid=upload.sid,
                product_sid=product.sid,
                batch_code=record.get('batch_code'),
                quantity=record.get('quantity', 0),
                expire_date=record.get('expire_date'),
                received_at=record.get('received_at', datetime.date.today()),
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
    query = select(WarehouseItem).join(Product)

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

    return items


@router.post("/to-store", response_model=Dict[str, str])
async def move_to_store(
        barcode_image: str = Form(None),
        item_sid: str = Form(None),
        quantity: int = Form(...),
        price: float = Form(...),
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
            select(Product).where(Product.barcode == barcode)
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
            select(WarehouseItem).where(WarehouseItem.sid == item_sid)
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

        # Создаем запись в магазине
        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=price,
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

        return {"message": "Item moved to store successfully", "store_item_sid": store_item.sid}

    finally:
        # Освобождаем лок
        await redis.delete(lock_key)