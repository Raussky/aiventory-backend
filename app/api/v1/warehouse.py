from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any, Optional
import datetime

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
            uploaded_at=datetime.datetime.utcnow(),
            rows_imported=0,
        )
        db.add(upload)
        await db.commit()
        await db.refresh(upload)

        records = await detect_and_parse_file(file)
        updated_items_info = []
        new_items_with_existing_barcode = []

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

            existing_items_updated = False
            total_existing_quantity = 0
            oldest_expire_date = None

            if barcode and product.barcode:
                existing_items_query = await db.execute(
                    select(WarehouseItem)
                    .join(Upload)
                    .join(Product)
                    .where(
                        Product.barcode == barcode,
                        WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
                        Upload.user_sid == current_user.sid,
                        WarehouseItem.quantity > 0
                    )
                    .order_by(WarehouseItem.expire_date.asc())
                )
                existing_items = existing_items_query.scalars().all()

                if existing_items:
                    existing_items_updated = True

                    for existing_item in existing_items:
                        total_existing_quantity += existing_item.quantity

                        if existing_item.expire_date and (
                                not oldest_expire_date or existing_item.expire_date < oldest_expire_date):
                            oldest_expire_date = existing_item.expire_date

                        if existing_item.expire_date:
                            days_until_expiry = (existing_item.expire_date - datetime.date.today()).days
                            if days_until_expiry <= 7:
                                existing_item.urgency_level = UrgencyLevel.CRITICAL
                            elif days_until_expiry <= 14:
                                existing_item.urgency_level = UrgencyLevel.URGENT
                            else:
                                existing_item.urgency_level = UrgencyLevel.URGENT
                        else:
                            existing_item.urgency_level = UrgencyLevel.URGENT

                    new_items_with_existing_barcode.append({
                        "product_name": product_name,
                        "barcode": barcode,
                        "existing_quantity": total_existing_quantity,
                        "oldest_expire_date": oldest_expire_date.isoformat() if oldest_expire_date else None,
                        "new_quantity": int(record.get('quantity', 0))
                    })

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

            warehouse_item = WarehouseItem(
                sid=Base.generate_sid(),
                upload_sid=upload.sid,
                product_sid=product.sid,
                batch_code=record.get('batch_code'),
                quantity=int(record.get('quantity', 0)),
                expire_date=expire_date,
                received_at=received_at,
                status=WarehouseItemStatus.IN_STOCK,
                urgency_level=UrgencyLevel.NORMAL,
            )
            db.add(warehouse_item)

            upload.rows_imported += 1

            if existing_items_updated:
                updated_items_info.append({
                    "product_name": product_name,
                    "barcode": barcode,
                    "existing_items_count": len(existing_items),
                    "total_existing_quantity": total_existing_quantity,
                    "message": f"Внимание! На складе уже есть {total_existing_quantity} единиц товара '{product_name}' с штрих-кодом {barcode}. Рекомендуется срочно реализовать старые партии перед размещением новой поставки."
                })

        await db.commit()
        await db.refresh(upload)

        response_data = {
            "sid": upload.sid,
            "file_name": upload.file_name,
            "uploaded_at": upload.uploaded_at.isoformat(),
            "rows_imported": upload.rows_imported,
            "urgent_actions_required": len(updated_items_info) > 0,
            "updated_items": updated_items_info,
            "summary": {
                "total_new_items": upload.rows_imported,
                "items_with_existing_stock": len(new_items_with_existing_barcode),
                "message": f"Загружено {upload.rows_imported} новых позиций. " +
                           (
                               f"Обнаружено {len(new_items_with_existing_barcode)} товаров с существующими остатками на складе. Необходимо срочно реализовать старые партии!" if new_items_with_existing_barcode else "")
            }
        }

        if new_items_with_existing_barcode:
            response_data["existing_stock_details"] = new_items_with_existing_barcode

        return response_data

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/items", response_model=List[WarehouseItemResponse])
async def get_warehouse_items(
        skip: int = 0,
        limit: int = 100,
        upload_sid: Optional[str] = None,
        expire_soon: bool = False,
        urgency_level: Optional[UrgencyLevel] = None,
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

    if upload_sid:
        query = query.where(WarehouseItem.upload_sid == upload_sid)

    if expire_soon:
        expiry_threshold = datetime.date.today() + datetime.timedelta(days=7)
        query = query.where(
            WarehouseItem.expire_date <= expiry_threshold,
            WarehouseItem.expire_date >= datetime.date.today(),
            WarehouseItem.status == WarehouseItemStatus.IN_STOCK
        )

    if urgency_level:
        query = query.where(WarehouseItem.urgency_level == urgency_level)

    result = await db.execute(
        query.order_by(WarehouseItem.urgency_level.desc(), WarehouseItem.expire_date.asc(),
                       WarehouseItem.received_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()

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

        discount_suggestion = suggest_discount(
            warehouse_item=item,
            store_price=suggested_price,
            base_price=base_price,
            category=category
        )

        warehouse_action = suggest_warehouse_action(
            warehouse_item=item,
            category=category
        )

        if item.urgency_level in [UrgencyLevel.URGENT, UrgencyLevel.CRITICAL]:
            duplicate_items_query = await db.execute(
                select(WarehouseItem)
                .join(Product)
                .join(Upload)
                .where(
                    Product.barcode == product.barcode,
                    WarehouseItem.sid != item.sid,
                    WarehouseItem.status == WarehouseItemStatus.IN_STOCK,
                    Upload.user_sid == current_user.sid,
                    WarehouseItem.quantity > 0,
                    WarehouseItem.received_at > item.received_at
                )
            )
            newer_items = duplicate_items_query.scalars().all()

            if newer_items:
                if not warehouse_action or warehouse_action.get('urgency') != 'critical':
                    warehouse_action = {
                        "action": "move_to_store_urgent",
                        "urgency": "critical",
                        "reason": f"Поступила новая партия этого товара. Необходимо срочно реализовать текущую партию!",
                        "newer_batches_count": len(newer_items),
                        "newer_batches_quantity": sum(item.quantity for item in newer_items)
                    }

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
            discount_suggestion=discount_suggestion,
            warehouse_action=warehouse_action
        )
        response_items.append(response_item)

    return response_items


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
            .order_by(WarehouseItem.urgency_level.desc(), WarehouseItem.expire_date.asc())
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

        discount_suggestion = suggest_discount(
            warehouse_item=warehouse_item,
            store_price=final_price,
            base_price=base_price,
            category=category
        )

        warehouse_action = suggest_warehouse_action(
            warehouse_item=warehouse_item,
            category=category
        )

        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=final_price,
            moved_at=datetime.datetime.utcnow(),
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
            "expire_date": warehouse_item.expire_date.isoformat() if warehouse_item.expire_date else None
        }

        if price is None:
            response["suggested_price"] = suggested_price
            response["price_calculation"] = {
                "base_price": base_price,
                "final_price": final_price,
                "markup_applied": round((final_price / base_price - 1) * 100, 2) if base_price > 0 else 0
            }

        if discount_suggestion:
            response["discount_suggestion"] = discount_suggestion

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
        .order_by(WarehouseItem.urgency_level.desc(), WarehouseItem.expire_date.asc())
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

        discount_suggestion = suggest_discount(
            warehouse_item=warehouse_item,
            store_price=final_price,
            base_price=base_price,
            category=category
        )

        warehouse_action = suggest_warehouse_action(
            warehouse_item=warehouse_item,
            category=category
        )

        store_item = StoreItem(
            sid=Base.generate_sid(),
            warehouse_item_sid=warehouse_item.sid,
            quantity=quantity,
            price=final_price,
            moved_at=datetime.datetime.utcnow(),
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
            "was_urgent": warehouse_item.urgency_level != UrgencyLevel.NORMAL
        }

        if price is None:
            response["suggested_price"] = suggested_price
            response["price_calculation"] = {
                "base_price": base_price,
                "final_price": final_price,
                "markup_applied": round((final_price / base_price - 1) * 100, 2) if base_price > 0 else 0
            }

        if discount_suggestion:
            response["discount_suggestion"] = discount_suggestion

        if warehouse_action:
            response["warehouse_action"] = warehouse_action

        return response

    finally:
        await redis.delete(lock_key)