from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload, joinedload
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import json
from pydantic import BaseModel

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import StoreItem, StoreItemStatus, Discount, Sale, WarehouseItem, Product, Upload, Category
from app.schemas.store import (
    StoreItemResponse, StoreItemCreate, DiscountCreate,
    DiscountResponse, SaleCreate, SaleResponse, RemovedItemsResponse
)
from app.schemas.inventory import ProductResponse, CategoryResponse
from app.models.base import Base
from app.core.dependencies import get_current_user

router = APIRouter()


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        return super().default(obj)


class PartialRemoveRequest(BaseModel):
    quantity: int


@router.get("/items", response_model=List[StoreItemResponse])
async def get_store_items(
        status: Optional[StoreItemStatus] = None,
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = (
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(Upload.user_sid == current_user.sid)
    )

    if status:
        query = query.where(StoreItem.status == status)
    else:
        query = query.where(StoreItem.quantity > 0)

    result = await db.execute(
        query.order_by(StoreItem.moved_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()

    response = []
    for item in items:
        if item.quantity <= 0 and status != StoreItemStatus.REMOVED:
            continue

        current_discounts = []
        for d in item.discounts:
            if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
                current_discounts.append({
                    "sid": d.sid,
                    "store_item_sid": d.store_item_sid,
                    "percentage": d.percentage,
                    "starts_at": d.starts_at,
                    "ends_at": d.ends_at,
                    "created_by_sid": d.created_by_sid,
                })

        category_response = None
        if item.warehouse_item.product.category:
            category_response = CategoryResponse(
                sid=item.warehouse_item.product.category.sid,
                name=item.warehouse_item.product.category.name
            )

        response_item = StoreItemResponse(
            sid=item.sid,
            warehouse_item_sid=item.warehouse_item_sid,
            quantity=max(0, item.quantity),
            price=item.price,
            moved_at=item.moved_at,
            status=item.status,
            product=ProductResponse(
                sid=item.warehouse_item.product.sid,
                name=item.warehouse_item.product.name,
                category_sid=item.warehouse_item.product.category_sid,
                barcode=item.warehouse_item.product.barcode,
                default_unit=item.warehouse_item.product.default_unit,
                default_price=item.warehouse_item.product.default_price,
                currency=item.warehouse_item.product.currency.value if item.warehouse_item.product.currency else None,
                storage_duration=item.warehouse_item.product.storage_duration,
                storage_duration_type=item.warehouse_item.product.storage_duration_type.value if item.warehouse_item.product.storage_duration_type else None,
                category=category_response
            ),
            expire_date=item.warehouse_item.expire_date,
            current_discounts=current_discounts,
            batch_code=item.warehouse_item.batch_code,
            days_until_expiry=(
                    item.warehouse_item.expire_date - datetime.now().date()).days if item.warehouse_item.expire_date else None
        )
        response.append(response_item)

    return response


@router.get("/removed-items", response_model=List[RemovedItemsResponse])
async def get_removed_items(
        skip: int = 0,
        limit: int = 100,
        reason: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = (
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            Upload.user_sid == current_user.sid,
            StoreItem.status.in_([StoreItemStatus.EXPIRED, StoreItemStatus.REMOVED]),
            StoreItem.quantity > 0
        )
    )

    if reason == 'expired':
        query = query.where(StoreItem.status == StoreItemStatus.EXPIRED)
    elif reason == 'manual':
        query = query.where(StoreItem.status == StoreItemStatus.REMOVED)

    result = await db.execute(
        query.order_by(StoreItem.moved_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()

    response = []
    for item in items:
        lost_value = item.price * max(0, item.quantity)

        if item.status == StoreItemStatus.EXPIRED:
            removal_reason = "Истек срок годности"
        else:
            removal_reason = "Убран вручную"

        category_response = None
        if item.warehouse_item.product.category:
            category_response = CategoryResponse(
                sid=item.warehouse_item.product.category.sid,
                name=item.warehouse_item.product.category.name
            )

        response_item = RemovedItemsResponse(
            sid=item.sid,
            warehouse_item_sid=item.warehouse_item_sid,
            quantity=max(0, item.quantity),
            price=item.price,
            moved_at=item.moved_at,
            removed_at=item.moved_at,
            status=item.status,
            product=ProductResponse(
                sid=item.warehouse_item.product.sid,
                name=item.warehouse_item.product.name,
                category_sid=item.warehouse_item.product.category_sid,
                barcode=item.warehouse_item.product.barcode,
                default_unit=item.warehouse_item.product.default_unit,
                default_price=item.warehouse_item.product.default_price,
                currency=item.warehouse_item.product.currency.value if item.warehouse_item.product.currency else None,
                storage_duration=item.warehouse_item.product.storage_duration,
                storage_duration_type=item.warehouse_item.product.storage_duration_type.value if item.warehouse_item.product.storage_duration_type else None,
                category=category_response
            ),
            expire_date=item.warehouse_item.expire_date,
            batch_code=item.warehouse_item.batch_code,
            lost_value=lost_value,
            removal_reason=removal_reason
        )
        response.append(response_item)

    return response


@router.post("/sales", response_model=SaleResponse)
async def record_sale(
        sale: SaleCreate,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == sale.store_item_sid,
            StoreItem.status == StoreItemStatus.ACTIVE,
            Upload.user_sid == current_user.sid
        )
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(
            status_code=404,
            detail="Active store item not found"
        )

    if store_item.quantity < sale.sold_qty:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough quantity available (requested: {sale.sold_qty}, available: {store_item.quantity})"
        )

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append(d)

    final_price = sale.sold_price
    if current_discounts and not sale.ignore_discount:
        max_discount = max(d.percentage for d in current_discounts)
        final_price = store_item.price * (1 - max_discount / 100)

    new_sale = Sale(
        sid=Base.generate_sid(),
        store_item_sid=sale.store_item_sid,
        sold_qty=sale.sold_qty,
        sold_price=final_price,
        sold_at=datetime.now(timezone.utc),
        cashier_sid=current_user.sid,
    )

    store_item.quantity -= sale.sold_qty

    db.add(new_sale)
    await db.commit()
    await db.refresh(new_sale)

    category_response = None
    if store_item.warehouse_item.product.category:
        category_response = CategoryResponse(
            sid=store_item.warehouse_item.product.category.sid,
            name=store_item.warehouse_item.product.category.name
        )

    return SaleResponse(
        sid=new_sale.sid,
        store_item_sid=new_sale.store_item_sid,
        sold_qty=new_sale.sold_qty,
        sold_price=new_sale.sold_price,
        sold_at=new_sale.sold_at,
        cashier_sid=new_sale.cashier_sid,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
            currency=store_item.warehouse_item.product.currency.value if store_item.warehouse_item.product.currency else None,
            storage_duration=store_item.warehouse_item.product.storage_duration,
            storage_duration_type=store_item.warehouse_item.product.storage_duration_type.value if store_item.warehouse_item.product.storage_duration_type else None,
            category=category_response
        ),
        total_amount=new_sale.sold_qty * new_sale.sold_price
    )


@router.post("/sales-by-barcode", response_model=SaleResponse)
async def record_sale_by_barcode(
        barcode: str,
        sold_qty: int,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Product)
        .join(Upload)
        .where(
            Product.barcode == barcode,
            StoreItem.status == StoreItemStatus.ACTIVE,
            Upload.user_sid == current_user.sid
        )
        .order_by(StoreItem.warehouse_item.has(WarehouseItem.expire_date).asc())
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(
            status_code=404,
            detail="Active store item with this barcode not found"
        )

    if store_item.quantity < sold_qty:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough quantity available (requested: {sold_qty}, available: {store_item.quantity})"
        )

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append(d)

    final_price = store_item.price
    if current_discounts:
        max_discount = max(d.percentage for d in current_discounts)
        final_price = store_item.price * (1 - max_discount / 100)

    new_sale = Sale(
        sid=Base.generate_sid(),
        store_item_sid=store_item.sid,
        sold_qty=sold_qty,
        sold_price=final_price,
        sold_at=datetime.now(timezone.utc),
        cashier_sid=current_user.sid,
    )

    store_item.quantity -= sold_qty

    db.add(new_sale)
    await db.commit()
    await db.refresh(new_sale)

    category_response = None
    if store_item.warehouse_item.product.category:
        category_response = CategoryResponse(
            sid=store_item.warehouse_item.product.category.sid,
            name=store_item.warehouse_item.product.category.name
        )

    return SaleResponse(
        sid=new_sale.sid,
        store_item_sid=new_sale.store_item_sid,
        sold_qty=new_sale.sold_qty,
        sold_price=new_sale.sold_price,
        sold_at=new_sale.sold_at,
        cashier_sid=new_sale.cashier_sid,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
            currency=store_item.warehouse_item.product.currency.value if store_item.warehouse_item.product.currency else None,
            storage_duration=store_item.warehouse_item.product.storage_duration,
            storage_duration_type=store_item.warehouse_item.product.storage_duration_type.value if store_item.warehouse_item.product.storage_duration_type else None,
            category=category_response
        ),
        total_amount=new_sale.sold_qty * new_sale.sold_price
    )


@router.get("/sales", response_model=List[SaleResponse])
async def get_sales(
        skip: int = 0,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = (
        select(Sale)
        .options(
            selectinload(Sale.store_item)
            .selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(Sale.store_item)
            .selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload)
        )
        .join(StoreItem)
        .join(WarehouseItem)
        .join(Upload)
        .where(Upload.user_sid == current_user.sid)
    )

    if start_date:
        query = query.where(Sale.sold_at >= start_date)
    if end_date:
        query = query.where(Sale.sold_at <= end_date)

    result = await db.execute(
        query.order_by(Sale.sold_at.desc())
        .offset(skip)
        .limit(limit)
    )
    sales = result.scalars().all()

    response = []
    for sale in sales:
        category_response = None
        if sale.store_item.warehouse_item.product.category:
            category_response = CategoryResponse(
                sid=sale.store_item.warehouse_item.product.category.sid,
                name=sale.store_item.warehouse_item.product.category.name
            )

        response_item = SaleResponse(
            sid=sale.sid,
            store_item_sid=sale.store_item_sid,
            sold_qty=sale.sold_qty,
            sold_price=sale.sold_price,
            sold_at=sale.sold_at,
            cashier_sid=sale.cashier_sid,
            product=ProductResponse(
                sid=sale.store_item.warehouse_item.product.sid,
                name=sale.store_item.warehouse_item.product.name,
                category_sid=sale.store_item.warehouse_item.product.category_sid,
                barcode=sale.store_item.warehouse_item.product.barcode,
                default_unit=sale.store_item.warehouse_item.product.default_unit,
                default_price=sale.store_item.warehouse_item.product.default_price,
                currency=sale.store_item.warehouse_item.product.currency.value if sale.store_item.warehouse_item.product.currency else None,
                storage_duration=sale.store_item.warehouse_item.product.storage_duration,
                storage_duration_type=sale.store_item.warehouse_item.product.storage_duration_type.value if sale.store_item.warehouse_item.product.storage_duration_type else None,
                category=category_response
            ),
            total_amount=sale.sold_qty * sale.sold_price
        )
        response.append(response_item)

    return response


@router.post("/discount", response_model=DiscountResponse)
async def create_discount(
        discount: DiscountCreate,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == discount.store_item_sid,
            Upload.user_sid == current_user.sid
        )
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(status_code=404, detail="Store item not found")

    overlapping_query = await db.execute(
        select(Discount).where(
            Discount.store_item_sid == discount.store_item_sid,
            ((Discount.starts_at <= discount.starts_at) & (Discount.ends_at >= discount.starts_at)) |
            ((Discount.starts_at <= discount.ends_at) & (Discount.ends_at >= discount.ends_at)) |
            ((Discount.starts_at >= discount.starts_at) & (Discount.ends_at <= discount.ends_at))
        )
    )

    if overlapping_query.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="There's an overlapping discount for this time period"
        )

    new_discount = Discount(
        sid=Base.generate_sid(),
        store_item_sid=discount.store_item_sid,
        percentage=discount.percentage,
        starts_at=discount.starts_at,
        ends_at=discount.ends_at,
        created_by_sid=current_user.sid,
    )

    db.add(new_discount)
    await db.commit()
    await db.refresh(new_discount)

    return new_discount


@router.post("/expire/{store_item_sid}", response_model=StoreItemResponse)
async def mark_as_expired(
        store_item_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == store_item_sid,
            Upload.user_sid == current_user.sid
        )
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(status_code=404, detail="Store item not found")

    store_item.status = StoreItemStatus.EXPIRED
    await db.commit()
    await db.refresh(store_item)

    category_response = None
    if store_item.warehouse_item.product.category:
        category_response = CategoryResponse(
            sid=store_item.warehouse_item.product.category.sid,
            name=store_item.warehouse_item.product.category.name
        )

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append({
                "sid": d.sid,
                "store_item_sid": d.store_item_sid,
                "percentage": d.percentage,
                "starts_at": d.starts_at,
                "ends_at": d.ends_at,
                "created_by_sid": d.created_by_sid,
            })

    return StoreItemResponse(
        sid=store_item.sid,
        warehouse_item_sid=store_item.warehouse_item_sid,
        quantity=max(0, store_item.quantity),
        price=store_item.price,
        moved_at=store_item.moved_at,
        status=store_item.status,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
            currency=store_item.warehouse_item.product.currency.value if store_item.warehouse_item.product.currency else None,
            storage_duration=store_item.warehouse_item.product.storage_duration,
            storage_duration_type=store_item.warehouse_item.product.storage_duration_type.value if store_item.warehouse_item.product.storage_duration_type else None,
            category=category_response
        ),
        expire_date=store_item.warehouse_item.expire_date,
        current_discounts=current_discounts,
        batch_code=store_item.warehouse_item.batch_code,
        days_until_expiry=(
                store_item.warehouse_item.expire_date - datetime.now().date()).days if store_item.warehouse_item.expire_date else None
    )


@router.post("/remove/{store_item_sid}", response_model=StoreItemResponse)
async def remove_from_store(
        store_item_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == store_item_sid,
            Upload.user_sid == current_user.sid
        )
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(status_code=404, detail="Store item not found")

    store_item.status = StoreItemStatus.REMOVED
    await db.commit()
    await db.refresh(store_item)

    category_response = None
    if store_item.warehouse_item.product.category:
        category_response = CategoryResponse(
            sid=store_item.warehouse_item.product.category.sid,
            name=store_item.warehouse_item.product.category.name
        )

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append({
                "sid": d.sid,
                "store_item_sid": d.store_item_sid,
                "percentage": d.percentage,
                "starts_at": d.starts_at,
                "ends_at": d.ends_at,
                "created_by_sid": d.created_by_sid,
            })

    return StoreItemResponse(
        sid=store_item.sid,
        warehouse_item_sid=store_item.warehouse_item_sid,
        quantity=max(0, store_item.quantity),
        price=store_item.price,
        moved_at=store_item.moved_at,
        status=store_item.status,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
            currency=store_item.warehouse_item.product.currency.value if store_item.warehouse_item.product.currency else None,
            storage_duration=store_item.warehouse_item.product.storage_duration,
            storage_duration_type=store_item.warehouse_item.product.storage_duration_type.value if store_item.warehouse_item.product.storage_duration_type else None,
            category=category_response
        ),
        expire_date=store_item.warehouse_item.expire_date,
        current_discounts=current_discounts,
        batch_code=store_item.warehouse_item.batch_code,
        days_until_expiry=(
                store_item.warehouse_item.expire_date - datetime.now().date()).days if store_item.warehouse_item.expire_date else None
    )


@router.post("/partial-remove/{store_item_sid}", response_model=StoreItemResponse)
async def partial_remove_from_store(
        store_item_sid: str,
        request: PartialRemoveRequest = Body(...),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.upload),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == store_item_sid,
            Upload.user_sid == current_user.sid
        )
    )
    store_item = store_item_query.scalar_one_or_none()

    if not store_item:
        raise HTTPException(status_code=404, detail="Store item not found")

    if store_item.quantity < request.quantity:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough quantity available (requested: {request.quantity}, available: {store_item.quantity})"
        )

    store_item.quantity -= request.quantity

    if store_item.quantity == 0:
        store_item.status = StoreItemStatus.REMOVED

    await db.commit()
    await db.refresh(store_item)

    category_response = None
    if store_item.warehouse_item.product.category:
        category_response = CategoryResponse(
            sid=store_item.warehouse_item.product.category.sid,
            name=store_item.warehouse_item.product.category.name
        )

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append({
                "sid": d.sid,
                "store_item_sid": d.store_item_sid,
                "percentage": d.percentage,
                "starts_at": d.starts_at,
                "ends_at": d.ends_at,
                "created_by_sid": d.created_by_sid,
            })

    return StoreItemResponse(
        sid=store_item.sid,
        warehouse_item_sid=store_item.warehouse_item_sid,
        quantity=max(0, store_item.quantity),
        price=store_item.price,
        moved_at=store_item.moved_at,
        status=store_item.status,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
            currency=store_item.warehouse_item.product.currency.value if store_item.warehouse_item.product.currency else None,
            storage_duration=store_item.warehouse_item.product.storage_duration,
            storage_duration_type=store_item.warehouse_item.product.storage_duration_type.value if store_item.warehouse_item.product.storage_duration_type else None,
            category=category_response
        ),
        expire_date=store_item.warehouse_item.expire_date,
        current_discounts=current_discounts,
        batch_code=store_item.warehouse_item.batch_code,
        days_until_expiry=(
                store_item.warehouse_item.expire_date - datetime.now().date()).days if store_item.warehouse_item.expire_date else None
    )


@router.get("/reports", response_model=Dict[str, Any])
async def get_store_reports(
        start_date: datetime = Query(None),
        end_date: datetime = Query(None),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    if not start_date:
        start_date = datetime.now(timezone.utc) - timedelta(days=30)

    if not end_date:
        end_date = datetime.now(timezone.utc)

    sales_query = """
        SELECT 
            DATE(s.sold_at) as date,
            p.name as product_name,
            c.name as category_name,
            SUM(s.sold_qty) as quantity,
            SUM(s.sold_qty * s.sold_price) as revenue
        FROM 
            sale s
        JOIN 
            storeitem si ON s.store_item_sid = si.sid
        JOIN 
            warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        JOIN
            category c ON p.category_sid = c.sid
        JOIN
            upload u ON wi.upload_sid = u.sid
        WHERE 
            s.sold_at BETWEEN :start_date AND :end_date
            AND u.user_sid = :user_sid
        GROUP BY 
            DATE(s.sold_at), p.name, c.name
        ORDER BY 
            date DESC, revenue DESC
    """

    sales_result = await db.execute(
        text(sales_query),
        {"start_date": start_date, "end_date": end_date, "user_sid": current_user.sid}
    )
    sales_data = sales_result.fetchall()

    discounts_query = """
        SELECT 
            p.name as product_name,
            c.name as category_name,
            d.percentage as discount_percentage,
            d.starts_at as start_date,
            d.ends_at as end_date,
            COUNT(s.id) as sales_count,
            SUM(s.sold_qty) as sold_quantity,
            SUM(s.sold_qty * s.sold_price) as discounted_revenue,
            SUM(s.sold_qty * p.default_price) as regular_revenue
        FROM 
            discount d
        JOIN 
            storeitem si ON d.store_item_sid = si.sid
        JOIN 
            warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        JOIN
            category c ON p.category_sid = c.sid
        JOIN
            upload u ON wi.upload_sid = u.sid
        LEFT JOIN 
            sale s ON si.sid = s.store_item_sid AND s.sold_at BETWEEN d.starts_at AND d.ends_at
        WHERE 
            d.starts_at <= :end_date AND d.ends_at >= :start_date
            AND u.user_sid = :user_sid
        GROUP BY 
            p.name, c.name, d.percentage, d.starts_at, d.ends_at
        ORDER BY 
            d.starts_at DESC
    """

    discounts_result = await db.execute(
        text(discounts_query),
        {"start_date": start_date, "end_date": end_date, "user_sid": current_user.sid}
    )
    discounts_data = discounts_result.fetchall()

    removed_query = """
        SELECT 
            p.name as product_name,
            c.name as category_name,
            SUM(si.quantity) as removed_quantity,
            SUM(si.quantity * si.price) as removed_value,
            COUNT(si.id) as removed_items_count,
            si.status as removal_reason
        FROM 
            storeitem si
        JOIN 
            warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        JOIN
            category c ON p.category_sid = c.sid
        JOIN
            upload u ON wi.upload_sid = u.sid
        WHERE 
            si.status IN ('EXPIRED', 'REMOVED') AND
            u.user_sid = :user_sid AND
            si.moved_at BETWEEN :start_date AND :end_date
            AND si.quantity > 0
        GROUP BY 
            p.name, c.name, si.status
        ORDER BY 
            removed_value DESC
    """

    removed_result = await db.execute(
        text(removed_query),
        {"start_date": start_date, "end_date": end_date, "user_sid": current_user.sid}
    )
    removed_data = removed_result.fetchall()

    report = {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "sales": [
            {
                "date": row.date.isoformat(),
                "product_name": row.product_name,
                "category_name": row.category_name,
                "quantity": float(row.quantity) if row.quantity else 0,
                "revenue": float(row.revenue) if row.revenue else 0
            }
            for row in sales_data
        ],
        "discounts": [
            {
                "product_name": row.product_name,
                "category_name": row.category_name,
                "discount_percentage": float(row.discount_percentage) if row.discount_percentage else 0,
                "start_date": row.start_date.isoformat(),
                "end_date": row.end_date.isoformat(),
                "sales_count": int(row.sales_count) if row.sales_count else 0,
                "sold_quantity": float(row.sold_quantity) if row.sold_quantity else 0,
                "discounted_revenue": float(row.discounted_revenue) if row.discounted_revenue else 0,
                "regular_revenue": float(row.regular_revenue) if row.regular_revenue else 0,
                "savings": float(
                    row.regular_revenue - row.discounted_revenue) if row.regular_revenue and row.discounted_revenue else 0
            }
            for row in discounts_data
        ],
        "removed": [
            {
                "product_name": row.product_name,
                "category_name": row.category_name,
                "removed_quantity": float(row.removed_quantity) if row.removed_quantity else 0,
                "removed_value": float(row.removed_value) if row.removed_value else 0,
                "removed_items_count": int(row.removed_items_count) if row.removed_items_count else 0,
                "removal_reason": "Истек срок годности" if row.removal_reason == "EXPIRED" else "Убран вручную"
            }
            for row in removed_data
        ],
        "summary": {
            "total_sales": float(sum(row.revenue for row in sales_data)) if sales_data else 0,
            "total_items_sold": float(sum(row.quantity for row in sales_data)) if sales_data else 0,
            "total_removed_value": float(sum(row.removed_value for row in removed_data)) if removed_data else 0,
            "total_removed_items": float(sum(row.removed_quantity for row in removed_data)) if removed_data else 0,
            "total_discount_savings": float(sum(
                (row.regular_revenue - row.discounted_revenue)
                for row in discounts_data
                if row.regular_revenue and row.discounted_revenue
            )) if discounts_data else 0
        }
    }

    return report