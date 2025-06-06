from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import or_, and_, func, desc, asc, case
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date, timezone

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Product, Category, Upload, WarehouseItem, WarehouseItemStatus, StoreItem, \
    StoreItemStatus, Currency, StorageDurationType, UrgencyLevel
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


@router.post("/upload", response_model=Dict[str, Any])
async def upload_file(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    try:
        upload = Upload(
            sid=Base.generate_sid(),
            user_sid=current_user.sid,
            file_name=file.filename,
            uploaded_at=datetime.utcnow(),
            rows_imported=0,
        )
        db.add(upload)
        await db.commit()
        await db.refresh(upload)

        records = await detect_and_parse_file(file)
        updated_items_info = []
        new_items_with_existing_barcode = []
        items_updated_count = 0

        for record in records:
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

            product_name = record.get('name')
            barcode = str(record.get('barcode')) if record.get('barcode') is not None else None

            product_query = await db.execute(
                select(Product).where(
                    (Product.name == product_name) &
                    (Product.category_sid == category.sid)
                )
            )
            product = product_query.scalar_one_or_none()

            if not product:
                storage_duration = int(record.get('storage_duration', 30))
                storage_duration_type_str = record.get('storage_duration_type', 'day').lower()

                if storage_duration_type_str == 'month':
                    storage_duration_type = StorageDurationType.MONTH
                elif storage_duration_type_str == 'year':
                    storage_duration_type = StorageDurationType.YEAR
                else:
                    storage_duration_type = StorageDurationType.DAY

                product = Product(
                    sid=Base.generate_sid(),
                    category_sid=category.sid,
                    name=product_name,
                    barcode=barcode,
                    default_unit=record.get('unit'),
                    default_price=float(record.get('price')) if record.get('price') is not None else None,
                    currency=Currency.KZT,
                    storage_duration=storage_duration,
                    storage_duration_type=storage_duration_type,
                )
                db.add(product)
                await db.commit()
                await db.refresh(product)

            batch_code = record.get('batch_code')
            if barcode and product.barcode and batch_code:
                existing_item_query = await db.execute(
                    select(WarehouseItem)
                    .join(Upload)
                    .join(Product)
                    .where(
                        Product.barcode == barcode,
                        WarehouseItem.batch_code == batch_code,
                        WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
                        Upload.user_sid == current_user.sid
                    )
                )
                existing_item = existing_item_query.scalar_one_or_none()

                if existing_item:
                    new_quantity = int(record.get('quantity', 0))
                    existing_item.quantity += new_quantity
                    items_updated_count += 1

                    updated_items_info.append({
                        "product_name": product_name,
                        "barcode": barcode,
                        "batch_code": batch_code,
                        "previous_quantity": existing_item.quantity - new_quantity,
                        "added_quantity": new_quantity,
                        "new_total_quantity": existing_item.quantity,
                        "message": f"Количество товара '{product_name}' (партия: {batch_code}) увеличено на {new_quantity}"
                    })

                    upload.rows_imported += 1
                    continue

            expire_date = None
            if record.get('expire_date'):
                if isinstance(record.get('expire_date'), str):
                    expire_date = datetime.strptime(record.get('expire_date'), '%Y-%m-%d').date()
                else:
                    expire_date = record.get('expire_date')

            received_at = date.today()
            if record.get('received_at'):
                if isinstance(record.get('received_at'), str):
                    received_at = datetime.strptime(record.get('received_at'), '%Y-%m-%d').date()
                else:
                    received_at = record.get('received_at')

            urgency_level = UrgencyLevel.NORMAL
            if expire_date:
                days_until_expiry = (expire_date - date.today()).days
                if days_until_expiry <= 3:
                    urgency_level = UrgencyLevel.CRITICAL
                elif days_until_expiry <= 7:
                    urgency_level = UrgencyLevel.URGENT

            warehouse_item = WarehouseItem(
                sid=Base.generate_sid(),
                upload_sid=upload.sid,
                product_sid=product.sid,
                batch_code=batch_code,
                quantity=int(record.get('quantity', 0)),
                expire_date=expire_date,
                received_at=received_at,
                status=WarehouseItemStatus.IN_STOCK,
                urgency_level=urgency_level,
            )
            db.add(warehouse_item)
            upload.rows_imported += 1

        await db.commit()
        await db.refresh(upload)

        response_data = {
            "sid": upload.sid,
            "file_name": upload.file_name,
            "uploaded_at": upload.uploaded_at.isoformat(),
            "rows_imported": upload.rows_imported,
            "items_updated": items_updated_count,
            "new_items_created": upload.rows_imported - items_updated_count,
            "updated_items": updated_items_info,
            "summary": {
                "total_processed": upload.rows_imported,
                "items_updated": items_updated_count,
                "new_items_created": upload.rows_imported - items_updated_count,
                "message": f"Обработано {upload.rows_imported} записей. Обновлено: {items_updated_count}, создано новых: {upload.rows_imported - items_updated_count}"
            }
        }

        return response_data

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/items", response_model=Dict[str, Any])
async def get_warehouse_items(
        skip: int = 0,
        limit: Optional[int] = None,
        upload_sid: Optional[str] = None,
        expire_soon: bool = False,
        urgency_level: Optional[UrgencyLevel] = None,
        category_sid: Optional[str] = None,
        status: Optional[WarehouseItemStatus] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = select(WarehouseItem).options(
        selectinload(WarehouseItem.product).selectinload(Product.category),
        selectinload(WarehouseItem.upload)
    ).join(Upload).where(
        Upload.user_sid == current_user.sid,
        WarehouseItem.quantity > 0
    )

    count_query = select(func.count()).select_from(WarehouseItem).join(Upload).where(
        Upload.user_sid == current_user.sid,
        WarehouseItem.quantity > 0
    )

    if upload_sid:
        query = query.where(WarehouseItem.upload_sid == upload_sid)
        count_query = count_query.where(WarehouseItem.upload_sid == upload_sid)

    if expire_soon:
        expiry_threshold = date.today() + timedelta(days=7)
        query = query.where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date >= date.today(),
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )
        count_query = count_query.where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date >= date.today(),
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )

    if urgency_level:
        query = query.where(WarehouseItem.urgency_level == urgency_level)
        count_query = count_query.where(WarehouseItem.urgency_level == urgency_level)

    if status:
        query = query.where(WarehouseItem.status == status)
        count_query = count_query.where(WarehouseItem.status == status)

    if category_sid:
        query = query.join(Product).where(Product.category_sid == category_sid)
        count_query = count_query.join(Product).where(Product.category_sid == category_sid)

    total_result = await db.execute(count_query)
    total_count = total_result.scalar()

    urgency_order = case(
        (WarehouseItem.urgency_level == UrgencyLevel.CRITICAL, 3),
        (WarehouseItem.urgency_level == UrgencyLevel.URGENT, 2),
        (WarehouseItem.urgency_level == UrgencyLevel.NORMAL, 1),
        else_=0
    )

    query = query.order_by(
        desc(urgency_order),
        asc(WarehouseItem.expire_date),
        desc(WarehouseItem.received_at)
    )

    query = query.offset(skip)
    if limit:
        query = query.limit(limit)

    result = await db.execute(query)
    items = result.scalars().all()

    expiry_threshold = date.today() + timedelta(days=7)
    for item in items:
        if item.expire_date:
            days_until_expiry = (item.expire_date - date.today()).days
            if days_until_expiry <= 3:
                item.urgency_level = UrgencyLevel.CRITICAL
            elif days_until_expiry <= 7:
                item.urgency_level = UrgencyLevel.URGENT
            else:
                item.urgency_level = UrgencyLevel.NORMAL
        else:
            item.urgency_level = UrgencyLevel.NORMAL

    categories_query = select(Category).join(Product).join(WarehouseItem).join(Upload).where(
        Upload.user_sid == current_user.sid,
        WarehouseItem.quantity > 0
    ).distinct()
    categories_result = await db.execute(categories_query)
    categories = categories_result.scalars().all()

    response_items = []
    for item in items:
        product = item.product
        category = product.category
        base_price = product.default_price or 0

        suggested_price = calculate_store_price(
            warehouse_item=item,
            base_price=base_price,
            category=category
        )

        warehouse_action = suggest_warehouse_action(
            warehouse_item=item,
            category=category
        )

        response_item = WarehouseItemResponse(
            sid=item.sid,
            product_sid=item.product_sid,
            upload_sid=item.upload_sid,
            batch_code=item.batch_code,
            quantity=item.quantity,
            expire_date=item.expire_date,
            received_at=item.received_at,
            status=item.status,
            urgency_level=item.urgency_level,
            product=item.product,
            suggested_price=suggested_price,
            wholesale_price=base_price,
            warehouse_action=warehouse_action
        )
        response_items.append(response_item)

    return {
        "items": response_items,
        "total": total_count,
        "skip": skip,
        "limit": limit,
        "categories": [{"sid": cat.sid, "name": cat.name} for cat in categories]
    }


@router.delete("/items", response_model=Dict[str, Any])
async def delete_warehouse_items(
        item_sids: List[str],
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    items_query = await db.execute(
        select(WarehouseItem)
        .join(Upload)
        .where(
            WarehouseItem.sid.in_(item_sids),
            Upload.user_sid == current_user.sid,
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )
    )
    items = items_query.scalars().all()

    if len(items) != len(item_sids):
        raise HTTPException(
            status_code=400,
            detail="Some items not found or not available for deletion"
        )

    deleted_count = 0
    for item in items:
        item.status = WarehouseItemStatus.DISCARDED
        item.quantity = 0
        deleted_count += 1

    await db.commit()

    return {
        "message": f"Successfully deleted {deleted_count} items",
        "deleted_count": deleted_count
    }


@router.post("/to-store", response_model=Dict[str, Any])
async def move_to_store(
        barcode_image: str = Form(None),
        item_sid: str = Form(None),
        quantity: int = Form(...),
        price: float = Form(None),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    if not barcode_image and not item_sid:
        raise HTTPException(
            status_code=400,
            detail="Either barcode image or item_sid must be provided"
        )

    if barcode_image:
        barcode = await decode_barcode_from_base64(barcode_image)

        product_query = await db.execute(
            select(Product).options(selectinload(Product.category)).where(Product.barcode == barcode)
        )
        product = product_query.scalar_one_or_none()

        if not product:
            raise HTTPException(status_code=404, detail="Product not found for this barcode")

        warehouse_query = await db.execute(
            select(WarehouseItem)
            .join(Upload)
            .where(
                WarehouseItem.product_sid == product.sid,
                WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
                WarehouseItem.quantity >= quantity,
                Upload.user_sid == current_user.sid
            )
            .order_by(
                desc(case(
                    (WarehouseItem.urgency_level == UrgencyLevel.CRITICAL, 3),
                    (WarehouseItem.urgency_level == UrgencyLevel.URGENT, 2),
                    (WarehouseItem.urgency_level == UrgencyLevel.NORMAL, 1),
                    else_=0
                )),
                asc(WarehouseItem.expire_date),
                asc(WarehouseItem.received_at)
            )
        )
        warehouse_item = warehouse_query.scalar_one_or_none()

        if not warehouse_item:
            raise HTTPException(status_code=404, detail="No available items in warehouse")

        item_sid = warehouse_item.sid

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
            .join(Upload)
            .where(
                WarehouseItem.sid == item_sid,
                Upload.user_sid == current_user.sid
            )
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

        product = warehouse_item.product
        category = product.category
        base_price = product.default_price or 0

        suggested_price = calculate_store_price(
            warehouse_item=warehouse_item,
            base_price=base_price,
            category=category
        )

        final_price = price if price is not None else suggested_price

        warehouse_action = suggest_warehouse_action(
            warehouse_item=warehouse_item,
            category=category
        )

        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=final_price,
            moved_at=datetime.utcnow(),
            status=StoreItemStatus.ACTIVE,
        )
        db.add(store_item)

        warehouse_item.quantity -= quantity

        if warehouse_item.quantity == 0:
            warehouse_item.status = WarehouseItemStatus.MOVED

        if warehouse_item.urgency_level != UrgencyLevel.NORMAL:
            warehouse_item.urgency_level = UrgencyLevel.NORMAL

        await db.commit()

        response = {
            "message": "Item moved to store successfully",
            "store_item_sid": store_item.sid,
            "price": final_price,
            "product_name": product.name,
            "category": category.name,
            "expire_date": warehouse_item.expire_date.isoformat() if warehouse_item.expire_date else None,
            "wholesale_price": base_price,
            "suggested_price": suggested_price
        }

        if warehouse_action:
            response["warehouse_action"] = warehouse_action

        return response

    finally:
        await redis.delete(lock_key)


@router.post("/to-store-by-barcode", response_model=Dict[str, Any])
async def move_to_store_by_barcode(
        barcode: str = Form(...),
        quantity: int = Form(...),
        price: float = Form(None),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    product_query = await db.execute(
        select(Product).options(selectinload(Product.category)).where(Product.barcode == barcode)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found for this barcode")

    warehouse_query = await db.execute(
        select(WarehouseItem)
        .options(selectinload(WarehouseItem.product).selectinload(Product.category))
        .join(Upload)
        .where(
            WarehouseItem.product_sid == product.sid,
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
            WarehouseItem.quantity >= quantity,
            Upload.user_sid == current_user.sid
        )
        .order_by(
            desc(case(
                (WarehouseItem.urgency_level == UrgencyLevel.CRITICAL, 3),
                (WarehouseItem.urgency_level == UrgencyLevel.URGENT, 2),
                (WarehouseItem.urgency_level == UrgencyLevel.NORMAL, 1),
                else_=0
            )),
            asc(WarehouseItem.expire_date),
            asc(WarehouseItem.received_at)
        )
    )
    warehouse_item = warehouse_query.scalar_one_or_none()

    if not warehouse_item:
        raise HTTPException(status_code=404, detail="No available items in warehouse")

    lock_key = f"lock:item:{warehouse_item.sid}"
    lock_acquired = await redis.set(lock_key, str(current_user.id), nx=True, ex=5)

    if not lock_acquired:
        raise HTTPException(
            status_code=409,
            detail="Another operation is in progress for this item"
        )

    try:
        if warehouse_item.status != WarehouseItemStatus.IN_STOCK:
            raise HTTPException(status_code=400, detail="Item is not available in stock")

        if warehouse_item.quantity < quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough quantity available (requested: {quantity}, available: {warehouse_item.quantity})"
            )

        product = warehouse_item.product
        category = product.category
        base_price = product.default_price or 0

        suggested_price = calculate_store_price(
            warehouse_item=warehouse_item,
            base_price=base_price,
            category=category
        )

        final_price = price if price is not None else suggested_price

        warehouse_action = suggest_warehouse_action(
            warehouse_item=warehouse_item,
            category=category
        )

        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=final_price,
            moved_at=datetime.utcnow(),
            status=StoreItemStatus.ACTIVE,
        )
        db.add(store_item)

        warehouse_item.quantity -= quantity

        if warehouse_item.quantity == 0:
            warehouse_item.status = WarehouseItemStatus.MOVED

        await db.commit()

        response = {
            "message": "Item moved to store successfully",
            "store_item_sid": store_item.sid,
            "price": final_price,
            "product_name": product.name,
            "category": category.name,
            "expire_date": warehouse_item.expire_date.isoformat() if warehouse_item.expire_date else None,
            "was_urgent": warehouse_item.urgency_level != UrgencyLevel.NORMAL,
            "wholesale_price": base_price,
            "suggested_price": suggested_price
        }

        if warehouse_action:
            response["warehouse_action"] = warehouse_action

        return response

    finally:
        await redis.delete(lock_key)