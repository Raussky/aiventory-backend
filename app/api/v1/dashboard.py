from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from datetime import datetime, timedelta
from typing import Dict, Any

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import (
    Product, WarehouseItem, StoreItem, Sale,
    WarehouseItemStatus, StoreItemStatus, Category
)
from app.core.dependencies import get_current_user

router = APIRouter()


@router.get("/stats", response_model=Dict[str, Any])
async def get_dashboard_stats(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    today = datetime.now().date()
    thirty_days_ago = today - timedelta(days=30)
    sixty_days_ago = today - timedelta(days=60)
    expiry_threshold = today + timedelta(days=7)

    total_products = await db.execute(
        select(func.count(Product.id))
    )
    total_products_count = total_products.scalar()

    warehouse_query = """
        SELECT COUNT(DISTINCT wi.product_sid)
        FROM warehouseitem wi
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE wi.status = 'IN_STOCK' 
        AND wi.quantity > 0
        AND u.user_sid = :user_sid
    """

    warehouse_result = await db.execute(
        text(warehouse_query),
        {"user_sid": current_user.sid}
    )
    products_in_warehouse = warehouse_result.scalar()

    store_query = """
        SELECT COUNT(DISTINCT wi.product_sid)
        FROM storeitem si
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE si.status = 'ACTIVE' 
        AND si.quantity > 0
        AND u.user_sid = :user_sid
    """

    store_result = await db.execute(
        text(store_query),
        {"user_sid": current_user.sid}
    )
    products_in_store = store_result.scalar()

    expiring_query = """
        SELECT COUNT(DISTINCT wi.product_sid)
        FROM (
            SELECT wi.product_sid
            FROM warehouseitem wi
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE wi.status = 'IN_STOCK' 
            AND wi.quantity > 0
            AND wi.expire_date <= :expiry_threshold
            AND wi.expire_date > :today
            AND u.user_sid = :user_sid

            UNION

            SELECT wi.product_sid
            FROM storeitem si
            JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE si.status = 'ACTIVE' 
            AND si.quantity > 0
            AND wi.expire_date <= :expiry_threshold
            AND wi.expire_date > :today
            AND u.user_sid = :user_sid
        ) AS expiring_products
    """

    expiring_result = await db.execute(
        text(expiring_query),
        {
            "user_sid": current_user.sid,
            "expiry_threshold": expiry_threshold,
            "today": today
        }
    )
    products_expiring_soon = expiring_result.scalar()

    revenue_30_days_query = """
        SELECT COALESCE(SUM(s.sold_qty * s.sold_price), 0) as revenue
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE s.sold_at >= :start_date
        AND u.user_sid = :user_sid
    """

    revenue_result = await db.execute(
        text(revenue_30_days_query),
        {"start_date": thirty_days_ago, "user_sid": current_user.sid}
    )
    total_revenue_last_30_days = float(revenue_result.scalar() or 0)

    revenue_60_to_30_days_query = """
        SELECT COALESCE(SUM(s.sold_qty * s.sold_price), 0) as revenue
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE s.sold_at >= :start_date AND s.sold_at < :end_date
        AND u.user_sid = :user_sid
    """

    prev_revenue_result = await db.execute(
        text(revenue_60_to_30_days_query),
        {
            "start_date": sixty_days_ago,
            "end_date": thirty_days_ago,
            "user_sid": current_user.sid
        }
    )
    prev_revenue = float(prev_revenue_result.scalar() or 0)

    revenue_change = 0
    if prev_revenue > 0:
        revenue_change = ((total_revenue_last_30_days - prev_revenue) / prev_revenue) * 100

    sales_30_days_query = """
        SELECT COUNT(*) as sales_count
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE s.sold_at >= :start_date
        AND u.user_sid = :user_sid
    """

    sales_result = await db.execute(
        text(sales_30_days_query),
        {"start_date": thirty_days_ago, "user_sid": current_user.sid}
    )
    total_sales_last_30_days = int(sales_result.scalar() or 0)

    prev_sales_result = await db.execute(
        text("""
            SELECT COUNT(*) as sales_count
            FROM sale s
            JOIN storeitem si ON s.store_item_sid = si.sid
            JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE s.sold_at >= :start_date AND s.sold_at < :end_date
            AND u.user_sid = :user_sid
        """),
        {
            "start_date": sixty_days_ago,
            "end_date": thirty_days_ago,
            "user_sid": current_user.sid
        }
    )
    prev_sales = int(prev_sales_result.scalar() or 0)

    sales_change = 0
    if prev_sales > 0:
        sales_change = ((total_sales_last_30_days - prev_sales) / prev_sales) * 100

    avg_check = 0
    if total_sales_last_30_days > 0:
        avg_check = total_revenue_last_30_days / total_sales_last_30_days

    prev_avg_check = 0
    if prev_sales > 0:
        prev_avg_check = prev_revenue / prev_sales

    avg_check_change = 0
    if prev_avg_check > 0:
        avg_check_change = ((avg_check - prev_avg_check) / prev_avg_check) * 100

    total_store_items_query = """
        SELECT COUNT(DISTINCT si.sid) as total_items
        FROM storeitem si
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE si.moved_at >= :start_date
        AND u.user_sid = :user_sid
    """

    store_items_result = await db.execute(
        text(total_store_items_query),
        {"start_date": thirty_days_ago, "user_sid": current_user.sid}
    )
    total_store_items = int(store_items_result.scalar() or 0)

    conversion_rate = 0
    if total_store_items > 0:
        conversion_rate = (total_sales_last_30_days / total_store_items) * 100

    prev_store_items_result = await db.execute(
        text("""
            SELECT COUNT(DISTINCT si.sid) as total_items
            FROM storeitem si
            JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE si.moved_at >= :start_date AND si.moved_at < :end_date
            AND u.user_sid = :user_sid
        """),
        {
            "start_date": sixty_days_ago,
            "end_date": thirty_days_ago,
            "user_sid": current_user.sid
        }
    )
    prev_store_items = int(prev_store_items_result.scalar() or 0)

    prev_conversion_rate = 0
    if prev_store_items > 0:
        prev_conversion_rate = (prev_sales / prev_store_items) * 100

    conversion_change = 0
    if prev_conversion_rate > 0:
        conversion_change = conversion_rate - prev_conversion_rate

    category_distribution_query = """
        WITH product_inventory AS (
            SELECT 
                p.category_sid,
                c.name as category_name,
                COUNT(DISTINCT p.sid) as product_count,
                COALESCE(SUM(wi.quantity), 0) as warehouse_quantity,
                COALESCE(SUM(si.quantity), 0) as store_quantity
            FROM product p
            JOIN category c ON p.category_sid = c.sid
            LEFT JOIN warehouseitem wi ON p.sid = wi.product_sid 
                AND wi.status = 'IN_STOCK' 
                AND wi.quantity > 0
                AND EXISTS (
                    SELECT 1 FROM upload u 
                    WHERE u.sid = wi.upload_sid 
                    AND u.user_sid = :user_sid
                )
            LEFT JOIN storeitem si ON wi.sid = si.warehouse_item_sid 
                AND si.status = 'ACTIVE' 
                AND si.quantity > 0
            GROUP BY p.category_sid, c.name
        )
        SELECT 
            category_name,
            product_count,
            warehouse_quantity + store_quantity as total_quantity
        FROM product_inventory
        WHERE warehouse_quantity + store_quantity > 0
        ORDER BY total_quantity DESC
    """

    category_result = await db.execute(
        text(category_distribution_query),
        {"user_sid": current_user.sid}
    )
    category_data = category_result.fetchall()

    category_distribution = [
        {
            "name": row.category_name,
            "value": int(row.total_quantity),
            "product_count": int(row.product_count)
        }
        for row in category_data
    ]

    top_products_query = """
        SELECT 
            p.sid,
            p.name,
            c.name as category_name,
            SUM(s.sold_qty) as total_sold,
            SUM(s.sold_qty * s.sold_price) as total_revenue
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN product p ON wi.product_sid = p.sid
        JOIN category c ON p.category_sid = c.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE s.sold_at >= :start_date
        AND u.user_sid = :user_sid
        GROUP BY p.sid, p.name, c.name
        ORDER BY total_sold DESC
        LIMIT 10
    """

    top_products_result = await db.execute(
        text(top_products_query),
        {"start_date": thirty_days_ago, "user_sid": current_user.sid}
    )
    top_products_data = top_products_result.fetchall()

    top_products = [
        {
            "name": row.name,
            "category": row.category_name,
            "quantity": int(row.total_sold),
            "revenue": float(row.total_revenue)
        }
        for row in top_products_data
    ]

    return {
        "total_products": total_products_count or 0,
        "products_in_warehouse": products_in_warehouse or 0,
        "products_in_store": products_in_store or 0,
        "products_expiring_soon": products_expiring_soon or 0,
        "total_revenue_last_30_days": total_revenue_last_30_days,
        "total_sales_last_30_days": total_sales_last_30_days,
        "revenue_change": round(revenue_change, 1),
        "sales_change": round(sales_change, 1),
        "avg_check": avg_check,
        "avg_check_change": round(avg_check_change, 1),
        "conversion_rate": round(conversion_rate, 1),
        "conversion_change": round(conversion_change, 1),
        "category_distribution": category_distribution,
        "top_products": top_products
    }