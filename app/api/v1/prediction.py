from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy.orm import selectinload

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Prediction, Product, TimeFrame, Category
from app.schemas.prediction import (
    PredictionResponse, PredictionCreate, PredictionRequest,
    PredictionStatResponse, ProductAnalyticsResponse
)
from app.schemas.inventory import ProductResponse, CategoryResponse
from app.core.dependencies import get_current_user
from app.services.prediction import PredictionService

router = APIRouter()


@router.get("/products", response_model=List[ProductResponse])
async def get_products(
        skip: int = 0,
        limit: int = 100,
        category_sid: Optional[str] = None,
        search: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = select(Product).options(selectinload(Product.category))

    if category_sid:
        query = query.where(Product.category_sid == category_sid)

    if search:
        query = query.where(Product.name.ilike(f"%{search}%"))

    result = await db.execute(
        query.order_by(Product.name)
        .offset(skip)
        .limit(limit)
    )
    products = result.scalars().all()

    return products


@router.get("/categories", response_model=List[CategoryResponse])
async def get_categories(
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = select(Category)

    if search:
        query = query.where(Category.name.ilike(f"%{search}%"))

    result = await db.execute(
        query.order_by(Category.name)
        .offset(skip)
        .limit(limit)
    )
    categories = result.scalars().all()

    return categories


@router.get("/sales-history/{product_sid}", response_model=Dict[str, Any])
async def get_sales_history(
        product_sid: str,
        days_back: int = Query(90, ge=7, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
        SELECT 
            DATE(s.sold_at) as date,
            SUM(s.sold_qty) as quantity,
            AVG(s.sold_price) as avg_price
        FROM 
            sale s
        JOIN 
            storeitem si ON s.store_item_sid = si.sid
        JOIN 
            warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN 
            upload u ON wi.upload_sid = u.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        WHERE 
            p.sid = :product_sid
            AND s.sold_at >= :min_date
            AND u.user_sid = :user_sid
        GROUP BY 
            DATE(s.sold_at)
        ORDER BY 
            date
    """)

    min_date = datetime.now() - timedelta(days=days_back)

    result = await db.execute(
        query,
        {"product_sid": product_sid, "min_date": min_date, "user_sid": current_user.sid}
    )
    rows = result.fetchall()

    if not rows:
        return {
            "dates": [],
            "quantities": [],
            "has_data": False
        }

    dates = []
    quantities = []

    for row in rows:
        dates.append(row.date.strftime("%Y-%m-%d"))
        quantities.append(float(row.quantity))

    return {
        "dates": dates,
        "quantities": quantities,
        "has_data": True
    }


@router.get("/category-sales/{category_sid}", response_model=Dict[str, Any])
async def get_category_sales(
        category_sid: str,
        days_back: int = Query(90, ge=7, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
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
            upload u ON wi.upload_sid = u.sid
        JOIN 
            product p ON wi.product_sid = p.sid
        WHERE 
            p.category_sid = :category_sid
            AND s.sold_at >= :min_date
            AND u.user_sid = :user_sid
        GROUP BY 
            DATE(s.sold_at), p.name
        ORDER BY 
            date, quantity DESC
    """)

    min_date = datetime.now() - timedelta(days=days_back)

    result = await db.execute(
        query,
        {"category_sid": category_sid, "min_date": min_date, "user_sid": current_user.sid}
    )
    rows = result.fetchall()

    if not rows:
        return {
            "dates": [],
            "products": [],
            "data": [],
            "has_data": False
        }

    df = pd.DataFrame(rows, columns=['date', 'product_name', 'quantity', 'revenue'])

    dates = sorted(df['date'].unique())
    products = df['product_name'].unique().tolist()

    data = []
    for date in dates:
        date_data = {'date': date.strftime("%Y-%m-%d")}
        date_df = df[df['date'] == date]

        for _, row in date_df.iterrows():
            date_data[row['product_name']] = float(row['quantity'])

        for product in products:
            if product not in date_data:
                date_data[product] = 0

        data.append(date_data)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "products": products[:10],
        "data": data,
        "has_data": True
    }


@router.get("/forecast/{product_sid}", response_model=List[PredictionResponse])
async def get_forecast(
        product_sid: str,
        periods: int = Query(90, ge=7, le=365),
        refresh: bool = False,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    prediction_service = PredictionService(db, current_user.sid)

    if not refresh:
        check_column_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'prediction' 
            AND column_name = 'user_sid'
        """)

        column_result = await db.execute(check_column_query)
        has_user_sid_column = column_result.scalar() is not None

        if has_user_sid_column:
            existing_query = await db.execute(
                select(Prediction)
                .options(
                    selectinload(Prediction.product).selectinload(Product.category)
                )
                .where(
                    and_(
                        Prediction.product_sid == product_sid,
                        Prediction.user_sid == current_user.sid,
                        Prediction.period_start >= datetime.now().date(),
                        Prediction.period_end <= datetime.now().date() + timedelta(days=periods)
                    )
                )
                .order_by(Prediction.period_start.asc())
            )
        else:
            existing_query = await db.execute(
                text("""
                    SELECT p.* FROM prediction p
                    WHERE p.product_sid = :product_sid
                    AND p.period_start >= :start_date
                    AND p.period_end <= :end_date
                    ORDER BY p.period_start ASC
                """),
                {
                    "product_sid": product_sid,
                    "start_date": datetime.now().date(),
                    "end_date": datetime.now().date() + timedelta(days=periods)
                }
            )

        existing = existing_query.scalars().all() if has_user_sid_column else []

        if existing and len(existing) >= periods * 0.5:
            return existing

    sales_check_query = text("""
        SELECT COUNT(DISTINCT DATE(s.sold_at)) as sale_days
        FROM sale s
        JOIN storeitem si ON s.store_item_sid = si.sid
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE 
            wi.product_sid = :product_sid
            AND u.user_sid = :user_sid
    """)

    sales_result = await db.execute(
        sales_check_query,
        {"product_sid": product_sid, "user_sid": current_user.sid}
    )
    sales_count = sales_result.scalar()

    if not sales_count or sales_count < 7:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough sales data to generate forecast. Found {sales_count or 0} days of sales, need at least 7."
        )

    forecasts = await prediction_service.generate_forecast(
        product_sid=product_sid,
        timeframe=TimeFrame.DAY,
        periods_ahead=periods
    )

    if not forecasts:
        raise HTTPException(
            status_code=400,
            detail="Failed to generate forecast. Please check the sales data."
        )

    saved_predictions = await prediction_service.save_forecast(forecasts)

    check_column_query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'prediction' 
        AND column_name = 'user_sid'
    """)

    column_result = await db.execute(check_column_query)
    has_user_sid_column = column_result.scalar() is not None

    if has_user_sid_column:
        result_query = await db.execute(
            select(Prediction)
            .options(
                selectinload(Prediction.product).selectinload(Product.category)
            )
            .where(
                and_(
                    Prediction.product_sid == product_sid,
                    Prediction.user_sid == current_user.sid,
                    Prediction.period_start >= datetime.now().date()
                )
            )
            .order_by(Prediction.period_start.asc())
            .limit(periods)
        )
    else:
        result_query = await db.execute(
            select(Prediction)
            .options(
                selectinload(Prediction.product).selectinload(Product.category)
            )
            .where(
                and_(
                    Prediction.product_sid == product_sid,
                    Prediction.period_start >= datetime.now().date()
                )
            )
            .order_by(Prediction.period_start.asc())
            .limit(periods)
        )

    predictions = result_query.scalars().all()
    return predictions


@router.get("/analytics/{product_sid}", response_model=Dict[str, Any])
async def get_product_analytics(
        product_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    product_query = await db.execute(
        select(Product)
        .options(selectinload(Product.category))
        .where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    sales_history = await get_sales_history(
        product_sid=product_sid,
        days_back=90,
        current_user=current_user,
        db=db
    )

    forecasts = await get_forecast(
        product_sid=product_sid,
        periods=90,
        refresh=False,
        current_user=current_user,
        db=db
    )

    analytics_query = text("""
        WITH sales_stats AS (
            SELECT 
                COUNT(DISTINCT DATE(s.sold_at)) as sale_days,
                SUM(s.sold_qty) as total_quantity,
                SUM(s.sold_qty * s.sold_price) as total_revenue,
                AVG(s.sold_qty) as avg_daily_quantity,
                STDDEV(s.sold_qty) as std_daily_quantity,
                MAX(s.sold_at) as last_sale_date
            FROM sale s
            JOIN storeitem si ON s.store_item_sid = si.sid
            JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE 
                wi.product_sid = :product_sid
                AND u.user_sid = :user_sid
                AND s.sold_at >= :min_date
        ),
        inventory_stats AS (
            SELECT 
                COALESCE(SUM(wi.quantity), 0) as warehouse_quantity,
                COALESCE(SUM(si.quantity), 0) as store_quantity
            FROM warehouseitem wi
            LEFT JOIN storeitem si ON wi.sid = si.warehouse_item_sid AND si.status = 'ACTIVE'
            JOIN upload u ON wi.upload_sid = u.sid
            WHERE 
                wi.product_sid = :product_sid
                AND wi.status = 'IN_STOCK'
                AND u.user_sid = :user_sid
        )
        SELECT 
            ss.*,
            ins.warehouse_quantity,
            ins.store_quantity
        FROM sales_stats ss
        CROSS JOIN inventory_stats ins
    """)

    stats_result = await db.execute(
        analytics_query,
        {
            "product_sid": product_sid,
            "user_sid": current_user.sid,
            "min_date": datetime.now() - timedelta(days=90)
        }
    )
    stats = stats_result.fetchone()

    response = {
        "product": {
            "sid": product.sid,
            "name": product.name,
            "category": product.category.name if product.category else None,
            "barcode": product.barcode
        },
        "current_inventory": {
            "warehouse": int(stats.warehouse_quantity) if stats else 0,
            "store": int(stats.store_quantity) if stats else 0,
            "total": int((stats.warehouse_quantity or 0) + (stats.store_quantity or 0)) if stats else 0
        },
        "sales_statistics": {
            "total_quantity": float(stats.total_quantity) if stats and stats.total_quantity else 0,
            "total_revenue": float(stats.total_revenue) if stats and stats.total_revenue else 0,
            "avg_daily_quantity": float(stats.avg_daily_quantity) if stats and stats.avg_daily_quantity else 0,
            "sale_days": int(stats.sale_days) if stats and stats.sale_days else 0,
            "last_sale_date": stats.last_sale_date.isoformat() if stats and stats.last_sale_date else None
        },
        "sales_history": sales_history,
        "forecast_90_days": [
            {
                "date": pred.period_start.isoformat(),
                "forecast_qty": pred.forecast_qty,
                "forecast_qty_lower": pred.forecast_qty_lower,
                "forecast_qty_upper": pred.forecast_qty_upper
            }
            for pred in forecasts
        ],
        "forecast_summary": {
            "next_7_days": sum(p.forecast_qty for p in forecasts[:7]),
            "next_30_days": sum(p.forecast_qty for p in forecasts[:30]),
            "next_90_days": sum(p.forecast_qty for p in forecasts[:90])
        }
    }

    return response