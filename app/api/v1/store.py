from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import json
from redis.asyncio import Redis

from app.db.session import get_db
from app.db.redis import get_redis
from app.models.users import User
from app.models.inventory import StoreItem, StoreItemStatus, Discount, Sale, WarehouseItem, Product, Upload, CartItem
from app.schemas.store import (
    StoreItemResponse, StoreItemCreate, DiscountCreate,
    DiscountResponse, SaleCreate, SaleResponse, CartItemCreate,
    CartItemResponse, CheckoutRequest, CheckoutResponse
)
from app.schemas.inventory import ProductResponse
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

@router.get("/items", response_model=List[StoreItemResponse])
async def get_store_items(
        status: Optional[StoreItemStatus] = None,
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
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

    result = await db.execute(
        query.order_by(StoreItem.moved_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()

    response = []
    for item in items:
        current_discounts = []
        for d in item.discounts:
            if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
                current_discounts.append(
                    DiscountResponse(
                        sid=d.sid,
                        store_item_sid=d.store_item_sid,
                        percentage=d.percentage,
                        starts_at=d.starts_at,
                        ends_at=d.ends_at,
                        created_by_sid=d.created_by_sid,
                    )
                )

        response_item = StoreItemResponse(
            sid=item.sid,
            warehouse_item_sid=item.warehouse_item_sid,
            quantity=item.quantity,
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
            ),
            expire_date=item.warehouse_item.expire_date,
            current_discounts=current_discounts
        )
        response.append(response_item)

    return response

@router.get("/cart/items", response_model=List[CartItemResponse])
async def get_cart_items(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = (
        select(CartItem)
        .options(
            selectinload(CartItem.store_item)
            .selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
            .selectinload(Product.category),
            selectinload(CartItem.store_item)
            .selectinload(StoreItem.discounts)
        )
        .where(CartItem.user_sid == current_user.sid)
        .order_by(CartItem.added_at.desc())
    )

    result = await db.execute(query)
    cart_items = result.scalars().all()

    response = []
    for cart_item in cart_items:
        response_item = CartItemResponse(
            sid=cart_item.sid,
            store_item_sid=cart_item.store_item_sid,
            quantity=cart_item.quantity,
            price_per_unit=cart_item.price_per_unit,
            added_at=cart_item.added_at,
            user_sid=cart_item.user_sid,
            product=ProductResponse(
                sid=cart_item.store_item.warehouse_item.product.sid,
                name=cart_item.store_item.warehouse_item.product.name,
                category_sid=cart_item.store_item.warehouse_item.product.category_sid,
                barcode=cart_item.store_item.warehouse_item.product.barcode,
                default_unit=cart_item.store_item.warehouse_item.product.default_unit,
                default_price=cart_item.store_item.warehouse_item.product.default_price,
            ),
            expire_date=cart_item.store_item.warehouse_item.expire_date,
            total_price=cart_item.quantity * cart_item.price_per_unit
        )
        response.append(response_item)

    return response

@router.post("/cart/add", response_model=CartItemResponse)
async def add_to_cart(
        cart_item: CartItemCreate,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product),
            selectinload(StoreItem.discounts)
        )
        .join(WarehouseItem)
        .join(Upload)
        .where(
            StoreItem.sid == cart_item.store_item_sid,
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

    if store_item.quantity < cart_item.quantity:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough quantity available (requested: {cart_item.quantity}, available: {store_item.quantity})"
        )

    existing_cart_item_query = await db.execute(
        select(CartItem).where(
            CartItem.store_item_sid == cart_item.store_item_sid,
            CartItem.user_sid == current_user.sid
        )
    )
    existing_cart_item = existing_cart_item_query.scalar_one_or_none()

    current_discounts = []
    for d in store_item.discounts:
        if d.starts_at <= datetime.now(timezone.utc) <= d.ends_at:
            current_discounts.append(d)

    price_per_unit = store_item.price
    if current_discounts:
        max_discount = max(d.percentage for d in current_discounts)
        price_per_unit = store_item.price * (1 - max_discount / 100)

    if existing_cart_item:
        if existing_cart_item.quantity + cart_item.quantity > store_item.quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Total quantity exceeds available stock"
            )
        existing_cart_item.quantity += cart_item.quantity
        existing_cart_item.price_per_unit = price_per_unit
        await db.commit()
        await db.refresh(existing_cart_item)
        new_cart_item = existing_cart_item
    else:
        new_cart_item = CartItem(
            sid=Base.generate_sid(),
            store_item_sid=cart_item.store_item_sid,
            user_sid=current_user.sid,
            quantity=cart_item.quantity,
            price_per_unit=price_per_unit,
        )
        db.add(new_cart_item)
        await db.commit()
        await db.refresh(new_cart_item)

    return CartItemResponse(
        sid=new_cart_item.sid,
        store_item_sid=new_cart_item.store_item_sid,
        quantity=new_cart_item.quantity,
        price_per_unit=new_cart_item.price_per_unit,
        added_at=new_cart_item.added_at,
        user_sid=new_cart_item.user_sid,
        product=ProductResponse(
            sid=store_item.warehouse_item.product.sid,
            name=store_item.warehouse_item.product.name,
            category_sid=store_item.warehouse_item.product.category_sid,
            barcode=store_item.warehouse_item.product.barcode,
            default_unit=store_item.warehouse_item.product.default_unit,
            default_price=store_item.warehouse_item.product.default_price,
        ),
        expire_date=store_item.warehouse_item.expire_date,
        total_price=new_cart_item.quantity * new_cart_item.price_per_unit
    )

@router.delete("/cart/remove/{cart_item_sid}")
async def remove_from_cart(
        cart_item_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    cart_item_query = await db.execute(
        select(CartItem).where(
            CartItem.sid == cart_item_sid,
            CartItem.user_sid == current_user.sid
        )
    )
    cart_item = cart_item_query.scalar_one_or_none()

    if not cart_item:
        raise HTTPException(status_code=404, detail="Cart item not found")

    await db.delete(cart_item)
    await db.commit()

    return {"message": "Item removed from cart"}

@router.post("/cart/checkout", response_model=CheckoutResponse)
async def checkout_cart(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    cart_items_query = await db.execute(
        select(CartItem)
        .options(
            selectinload(CartItem.store_item)
            .selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
        )
        .where(CartItem.user_sid == current_user.sid)
    )
    cart_items = cart_items_query.scalars().all()

    if not cart_items:
        raise HTTPException(status_code=400, detail="Cart is empty")

    sales = []
    total_amount = 0
    items_count = 0

    for cart_item in cart_items:
        store_item = cart_item.store_item

        if store_item.status != StoreItemStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Item {store_item.warehouse_item.product.name} is no longer active"
            )

        if store_item.quantity < cart_item.quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough quantity for {store_item.warehouse_item.product.name}"
            )

        sale = Sale(
            sid=Base.generate_sid(),
            store_item_sid=cart_item.store_item_sid,
            sold_qty=cart_item.quantity,
            sold_price=cart_item.price_per_unit,
            sold_at=datetime.now(timezone.utc),
            cashier_sid=current_user.sid,
        )

        store_item.quantity -= cart_item.quantity
        if store_item.quantity == 0:
            store_item.status = StoreItemStatus.REMOVED

        db.add(sale)
        sales.append(SaleResponse(
            sid=sale.sid,
            store_item_sid=sale.store_item_sid,
            sold_qty=sale.sold_qty,
            sold_price=sale.sold_price,
            sold_at=sale.sold_at,
            cashier_sid=sale.cashier_sid,
            product=ProductResponse(
                sid=store_item.warehouse_item.product.sid,
                name=store_item.warehouse_item.product.name,
                category_sid=store_item.warehouse_item.product.category_sid,
                barcode=store_item.warehouse_item.product.barcode,
                default_unit=store_item.warehouse_item.product.default_unit,
                default_price=store_item.warehouse_item.product.default_price,
            ),
            total_amount=sale.sold_qty * sale.sold_price
        ))

        total_amount += cart_item.quantity * cart_item.price_per_unit
        items_count += cart_item.quantity

        await db.delete(cart_item)

    await db.commit()

    await redis.publish(
        f"sales:{current_user.sid}",
        json.dumps({
            "type": "checkout_complete",
            "total_amount": total_amount,
            "items_count": items_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    )

    return CheckoutResponse(
        sales=sales,
        total_amount=total_amount,
        items_count=items_count
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
        redis: Redis = Depends(get_redis),
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

    await redis.delete(f"store:items:{current_user.sid}:*")

    return new_discount

@router.post("/expire/{store_item_sid}", response_model=StoreItemResponse)
async def mark_as_expired(
        store_item_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product),
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

    await redis.delete(f"store:items:{current_user.sid}:*")

    return store_item

@router.post("/remove/{store_item_sid}", response_model=StoreItemResponse)
async def remove_from_store(
        store_item_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product),
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

    await redis.delete(f"store:items:{current_user.sid}:*")

    return store_item

@router.post("/sales", response_model=SaleResponse)
async def record_sale(
        sale: SaleCreate,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    store_item_query = await db.execute(
        select(StoreItem)
        .options(
            selectinload(StoreItem.warehouse_item)
            .selectinload(WarehouseItem.product)
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

    new_sale = Sale(
        sid=Base.generate_sid(),
        store_item_sid=sale.store_item_sid,
        sold_qty=sale.sold_qty,
        sold_price=sale.sold_price,
        sold_at=datetime.now(timezone.utc),
        cashier_sid=current_user.sid,
    )

    store_item.quantity -= sale.sold_qty

    if store_item.quantity == 0:
        store_item.status = StoreItemStatus.REMOVED

    db.add(new_sale)
    await db.commit()
    await db.refresh(new_sale)

    await redis.delete(f"store:items:{current_user.sid}:*")

    await redis.publish(
        f"sales:{current_user.sid}",
        json.dumps({
            "type": "new_sale",
            "product_name": store_item.warehouse_item.product.name,
            "quantity": sale.sold_qty,
            "price": sale.sold_price,
            "total": sale.sold_qty * sale.sold_price,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    )

    return new_sale

@router.get("/reports", response_model=Dict[str, Any])
async def get_store_reports(
        start_date: datetime = Query(None),
        end_date: datetime = Query(None),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        redis: Redis = Depends(get_redis),
):
    if not start_date:
        start_date = datetime.now(timezone.utc) - timedelta(days=30)

    if not end_date:
        end_date = datetime.now(timezone.utc)

    cache_key = f"store:reports:{current_user.sid}:{start_date.date()}:{end_date.date()}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    sales_query = """
        SELECT 
            DATE(s.sold_at) as date,
            p.name as product_name,
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
            upload u ON wi.upload_sid = u.sid
        WHERE 
            s.sold_at BETWEEN :start_date AND :end_date
            AND u.user_sid = :user_sid
        GROUP BY 
            DATE(s.sold_at), p.name
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
            upload u ON wi.upload_sid = u.sid
        LEFT JOIN 
            sale s ON si.sid = s.store_item_sid AND s.sold_at BETWEEN d.starts_at AND d.ends_at
        WHERE 
            d.starts_at <= :end_date AND d.ends_at >= :start_date
            AND u.user_sid = :user_sid
        GROUP BY 
            p.name, d.percentage, d.starts_at, d.ends_at
        ORDER BY 
            d.starts_at DESC
    """

    discounts_result = await db.execute(
        text(discounts_query),
        {"start_date": start_date, "end_date": end_date, "user_sid": current_user.sid}
    )
    discounts_data = discounts_result.fetchall()

    expired_query = """
        SELECT 
            p.name as product_name,
            SUM(si.quantity) as expired_quantity,
            SUM(si.quantity * si.price) as expired_value,
            COUNT(si.id) as expired_items_count
        FROM 
            storeitem si
        JOIN 
            warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        JOIN
            upload u ON wi.upload_sid = u.sid
        WHERE 
            si.status = 'EXPIRED' AND
            u.user_sid = :user_sid AND
            (
                si.moved_at BETWEEN :start_date AND :end_date OR
                (wi.expire_date BETWEEN :start_date AND :end_date)
            )
        GROUP BY 
            p.name
        ORDER BY 
            expired_value DESC
    """

    expired_result = await db.execute(
        text(expired_query),
        {"start_date": start_date, "end_date": end_date, "user_sid": current_user.sid}
    )
    expired_data = expired_result.fetchall()

    report = {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "sales": [
            {
                "date": row.date.isoformat(),
                "product_name": row.product_name,
                "quantity": row.quantity,
                "revenue": row.revenue
            }
            for row in sales_data
        ],
        "discounts": [
            {
                "product_name": row.product_name,
                "discount_percentage": row.discount_percentage,
                "start_date": row.start_date.isoformat(),
                "end_date": row.end_date.isoformat(),
                "sales_count": row.sales_count,
                "sold_quantity": row.sold_quantity,
                "discounted_revenue": row.discounted_revenue,
                "regular_revenue": row.regular_revenue,
                "savings": row.regular_revenue - row.discounted_revenue if row.regular_revenue and row.discounted_revenue else 0
            }
            for row in discounts_data
        ],
        "expired": [
            {
                "product_name": row.product_name,
                "expired_quantity": row.expired_quantity,
                "expired_value": row.expired_value,
                "expired_items_count": row.expired_items_count
            }
            for row in expired_data
        ],
        "summary": {
            "total_sales": sum(row.revenue for row in sales_data) if sales_data else 0,
            "total_items_sold": sum(row.quantity for row in sales_data) if sales_data else 0,
            "total_expired_value": sum(row.expired_value for row in expired_data) if expired_data else 0,
            "total_expired_items": sum(row.expired_quantity for row in expired_data) if expired_data else 0,
            "total_discount_savings": sum(
                (row.regular_revenue - row.discounted_revenue)
                for row in discounts_data
                if row.regular_revenue and row.discounted_revenue
            ) if discounts_data else 0
        }
    }

    await redis.set(
        cache_key,
        json.dumps(report, cls=DateTimeEncoder),
        ex=1800
    )

    return report