from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
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


@router.get("/forecast/{product_sid}", response_model=List[PredictionResponse])
async def get_forecast(
        product_sid: str,
        refresh: bool = False,
        timeframe: TimeFrame = TimeFrame.MONTH,
        periods: int = 3,
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

    if not refresh:
        predictions_query = await db.execute(
            select(Prediction)
            .options(selectinload(Prediction.product).selectinload(Product.category))
            .where(
                Prediction.product_sid == product_sid,
                Prediction.timeframe == timeframe,
                Prediction.period_start >= datetime.now().date()
            )
            .order_by(Prediction.period_start.asc())
            .limit(periods)
        )
        predictions = predictions_query.scalars().all()

        if predictions and len(predictions) == periods:
            response_predictions = []
            for pred in predictions:
                category_response = None
                if pred.product.category:
                    category_response = CategoryResponse(
                        sid=pred.product.category.sid,
                        name=pred.product.category.name
                    )

                product_response = ProductResponse(
                    sid=pred.product.sid,
                    name=pred.product.name,
                    category_sid=pred.product.category_sid,
                    barcode=pred.product.barcode,
                    default_unit=pred.product.default_unit,
                    default_price=pred.product.default_price,
                    currency=pred.product.currency.value if pred.product.currency else None,
                    storage_duration=pred.product.storage_duration,
                    storage_duration_type=pred.product.storage_duration_type.value if pred.product.storage_duration_type else None,
                    category=category_response
                )

                response_predictions.append(PredictionResponse(
                    sid=pred.sid,
                    product_sid=pred.product_sid,
                    timeframe=pred.timeframe,
                    period_start=pred.period_start,
                    period_end=pred.period_end,
                    forecast_qty=pred.forecast_qty,
                    generated_at=pred.generated_at,
                    model_version=pred.model_version,
                    product=product_response,
                    forecast_qty_lower=pred.forecast_qty * 0.8,
                    forecast_qty_upper=pred.forecast_qty * 1.2
                ))

            return response_predictions

    prediction_service = PredictionService(db)
    forecasts = await prediction_service.generate_forecast(
        product_sid=product_sid,
        timeframe=timeframe,
        periods_ahead=periods
    )

    if not forecasts:
        raise HTTPException(
            status_code=400,
            detail="Could not generate forecast. Not enough sales data."
        )

    saved_predictions = await prediction_service.save_forecast(forecasts)

    product_query = await db.execute(
        select(Product)
        .options(selectinload(Product.category))
        .where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    category_response = None
    if product.category:
        category_response = CategoryResponse(
            sid=product.category.sid,
            name=product.category.name
        )

    product_response = ProductResponse(
        sid=product.sid,
        name=product.name,
        category_sid=product.category_sid,
        barcode=product.barcode,
        default_unit=product.default_unit,
        default_price=product.default_price,
        currency=product.currency.value if product.currency else None,
        storage_duration=product.storage_duration,
        storage_duration_type=product.storage_duration_type.value if product.storage_duration_type else None,
        category=category_response
    )

    response_predictions = []
    for pred in saved_predictions:
        response_predictions.append(PredictionResponse(
            sid=pred["sid"],
            product_sid=pred["product_sid"],
            timeframe=pred["timeframe"],
            period_start=pred["period_start"],
            period_end=pred["period_end"],
            forecast_qty=pred["forecast_qty"],
            generated_at=pred["generated_at"],
            model_version=pred["model_version"],
            product=product_response,
            forecast_qty_lower=pred.get("forecast_qty_lower", pred["forecast_qty"] * 0.8),
            forecast_qty_upper=pred.get("forecast_qty_upper", pred["forecast_qty"] * 1.2)
        ))

    return response_predictions


@router.get("/stats", response_model=Dict[str, Any])
async def get_prediction_stats(
        product_sid: Optional[str] = None,
        category_sid: Optional[str] = None,
        start_date: datetime = Query(None),
        end_date: datetime = Query(None),
        group_by: str = Query("day", regex="^(day|week|month)$"),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    if not start_date:
        start_date = datetime.now() - timedelta(days=90)

    if not end_date:
        end_date = datetime.now()

    if group_by == "day":
        date_format = "DATE(s.sold_at)"
    elif group_by == "week":
        date_format = "DATE_TRUNC('week', s.sold_at)"
    else:
        date_format = "DATE_TRUNC('month', s.sold_at)"

    query_text = f"""
        SELECT 
            {date_format} as date,
            p.name as product_name,
            p.sid as product_sid,
            c.name as category_name,
            c.sid as category_sid,
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
        WHERE 
            s.sold_at BETWEEN :start_date AND :end_date
    """

    params = {"start_date": start_date, "end_date": end_date}

    if product_sid:
        query_text += " AND p.sid = :product_sid"
        params["product_sid"] = product_sid

    if category_sid:
        query_text += " AND c.sid = :category_sid"
        params["category_sid"] = category_sid

    query_text += f"""
        GROUP BY 
            date, p.name, p.sid, c.name, c.sid
        ORDER BY 
            date, product_name
    """

    try:
        result = await db.execute(text(query_text), params)
        rows = result.fetchall()

        df = pd.DataFrame(rows, columns=["date", "product_name", "product_sid",
                                         "category_name", "category_sid",
                                         "quantity", "revenue"])

        if df.empty:
            return {
                "dates": [],
                "products": [],
                "categories": [],
                "quantity_data": [],
                "revenue_data": []
            }

        df["date_str"] = df["date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
        )

        dates = df["date_str"].unique().tolist()
        products = df[["product_sid", "product_name"]].drop_duplicates().to_dict("records")
        categories = df[["category_sid", "category_name"]].drop_duplicates().to_dict("records")

        quantity_pivot = df.pivot_table(
            index="date_str",
            columns="product_sid",
            values="quantity",
            fill_value=0
        ).reset_index().to_dict("records")

        revenue_pivot = df.pivot_table(
            index="date_str",
            columns="product_sid",
            values="revenue",
            fill_value=0
        ).reset_index().to_dict("records")

        if not product_sid:
            category_quantity = df.groupby(["date_str", "category_sid"])["quantity"].sum().reset_index()
            category_revenue = df.groupby(["date_str", "category_sid"])["revenue"].sum().reset_index()

            category_quantity_pivot = category_quantity.pivot_table(
                index="date_str",
                columns="category_sid",
                values="quantity",
                fill_value=0
            ).reset_index().to_dict("records")

            category_revenue_pivot = category_revenue.pivot_table(
                index="date_str",
                columns="category_sid",
                values="revenue",
                fill_value=0
            ).reset_index().to_dict("records")

            return {
                "dates": dates,
                "products": products,
                "categories": categories,
                "quantity_data": quantity_pivot,
                "revenue_data": revenue_pivot,
                "category_quantity_data": category_quantity_pivot,
                "category_revenue_data": category_revenue_pivot
            }

        return {
            "dates": dates,
            "products": products,
            "categories": categories,
            "quantity_data": quantity_pivot,
            "revenue_data": revenue_pivot
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing sales data: {str(e)}")


@router.get("/analytics/{product_sid}", response_model=Dict[str, Any])
async def get_product_analytics(
        product_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    product_query = await db.execute(
        select(Product).where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    prediction_service = PredictionService(db)
    analytics = await prediction_service.get_product_analytics(product_sid)

    if "error" in analytics:
        raise HTTPException(status_code=400, detail=analytics["error"])

    return analytics


@router.get("/trends/{product_sid}", response_model=Dict[str, Any])
async def get_product_trends(
        product_sid: str,
        days_back: int = Query(90, ge=7, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    product_query = await db.execute(
        select(Product).where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    prediction_service = PredictionService(db)
    trends = await prediction_service.get_sales_trends(
        product_sid=product_sid,
        days_back=days_back
    )

    return trends


@router.get("/insights", response_model=Dict[str, Any])
async def get_insights(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
        WITH recent_sales AS (
            SELECT 
                p.sid as product_sid,
                p.name as product_name,
                c.name as category_name,
                SUM(s.sold_qty) as total_quantity,
                SUM(s.sold_qty * s.sold_price) as total_revenue,
                COUNT(DISTINCT DATE(s.sold_at)) as sale_days,
                MAX(s.sold_at) as last_sale_date
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
            WHERE 
                s.sold_at >= :min_date
            GROUP BY 
                p.sid, p.name, c.name
        ),
        inventory_status AS (
            SELECT 
                p.sid as product_sid,
                COALESCE(SUM(wi.quantity), 0) as warehouse_qty,
                COALESCE(SUM(si.quantity), 0) as store_qty,
                MIN(wi.expire_date) as nearest_expiry
            FROM 
                product p
            LEFT JOIN 
                warehouseitem wi ON p.sid = wi.product_sid AND wi.status = 'IN_STOCK'
            LEFT JOIN 
                storeitem si ON wi.sid = si.warehouse_item_sid AND si.status = 'ACTIVE'
            GROUP BY 
                p.sid
        )
        SELECT 
            rs.product_sid,
            rs.product_name,
            rs.category_name,
            rs.total_quantity,
            rs.total_revenue,
            rs.sale_days,
            rs.last_sale_date,
            inv.warehouse_qty,
            inv.store_qty,
            inv.nearest_expiry,
            CASE 
                WHEN rs.sale_days > 0 THEN rs.total_quantity / rs.sale_days::float 
                ELSE 0 
            END as avg_daily_sales,
            CASE 
                WHEN rs.sale_days > 0 AND (inv.warehouse_qty + inv.store_qty) > 0 
                THEN (inv.warehouse_qty + inv.store_qty) / (rs.total_quantity / rs.sale_days::float)
                ELSE 0 
            END as days_of_supply
        FROM 
            recent_sales rs
        JOIN 
            inventory_status inv ON rs.product_sid = inv.product_sid
        ORDER BY 
            rs.total_revenue DESC
    """)

    min_date = datetime.now() - timedelta(days=30)
    result = await db.execute(query, {"min_date": min_date})
    rows = result.fetchall()

    insights = {
        "slow_moving_products": [],
        "out_of_stock_risks": [],
        "overstock_products": [],
        "expiring_products": [],
        "top_performers": [],
        "recommendations": []
    }

    for row in rows[:10]:
        insights["top_performers"].append({
            "product_name": row.product_name,
            "category": row.category_name,
            "revenue": float(row.total_revenue),
            "quantity_sold": float(row.total_quantity)
        })

    for row in rows:
        if row.avg_daily_sales < 1 and row.avg_daily_sales > 0:
            insights["slow_moving_products"].append({
                "product_name": row.product_name,
                "category": row.category_name,
                "avg_daily_sales": float(row.avg_daily_sales),
                "current_stock": int(row.warehouse_qty + row.store_qty)
            })

        if row.days_of_supply < 7 and row.days_of_supply > 0:
            insights["out_of_stock_risks"].append({
                "product_name": row.product_name,
                "category": row.category_name,
                "days_of_supply": float(row.days_of_supply),
                "current_stock": int(row.warehouse_qty + row.store_qty)
            })

        if row.days_of_supply > 60:
            insights["overstock_products"].append({
                "product_name": row.product_name,
                "category": row.category_name,
                "days_of_supply": float(row.days_of_supply),
                "current_stock": int(row.warehouse_qty + row.store_qty)
            })

        if row.nearest_expiry:
            days_until_expiry = (row.nearest_expiry - datetime.now().date()).days
            if days_until_expiry <= 14:
                insights["expiring_products"].append({
                    "product_name": row.product_name,
                    "category": row.category_name,
                    "days_until_expiry": days_until_expiry,
                    "expire_date": row.nearest_expiry.strftime("%Y-%m-%d")
                })

    if insights["slow_moving_products"]:
        insights["recommendations"].append({
            "type": "promotion",
            "title": "Рекомендуется провести акцию",
            "description": f"У вас {len(insights['slow_moving_products'])} медленно продающихся товаров",
            "products": [p["product_name"] for p in insights["slow_moving_products"][:5]]
        })

    if insights["out_of_stock_risks"]:
        insights["recommendations"].append({
            "type": "restock",
            "title": "Требуется пополнение запасов",
            "description": f"{len(insights['out_of_stock_risks'])} товаров могут закончиться в ближайшую неделю",
            "products": [p["product_name"] for p in insights["out_of_stock_risks"][:5]]
        })

    if insights["expiring_products"]:
        insights["recommendations"].append({
            "type": "urgent",
            "title": "Срочная реализация",
            "description": f"{len(insights['expiring_products'])} товаров с истекающим сроком годности",
            "products": [p["product_name"] for p in insights["expiring_products"][:5]]
        })

    return insights


@router.get("/optimization-suggestions", response_model=List[Dict[str, Any]])
async def get_optimization_suggestions(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
        WITH product_metrics AS (
            SELECT 
                p.sid,
                p.name,
                c.name as category_name,
                COALESCE(SUM(s.sold_qty), 0) as total_sold,
                COALESCE(SUM(s.sold_qty * s.sold_price), 0) as total_revenue,
                COALESCE(AVG(s.sold_price), p.default_price) as avg_price,
                COALESCE(SUM(wi.quantity), 0) as warehouse_stock,
                COALESCE(SUM(si.quantity), 0) as store_stock,
                COUNT(DISTINCT DATE(s.sold_at)) as sale_days
            FROM 
                product p
            JOIN 
                category c ON p.category_sid = c.sid
            LEFT JOIN 
                warehouseitem wi ON p.sid = wi.product_sid AND wi.status = 'IN_STOCK'
            LEFT JOIN 
                storeitem si ON wi.sid = si.warehouse_item_sid AND si.status = 'ACTIVE'
            LEFT JOIN 
                sale s ON si.sid = s.store_item_sid AND s.sold_at >= :min_date
            GROUP BY 
                p.sid, p.name, c.name, p.default_price
        )
        SELECT 
            *,
            CASE 
                WHEN sale_days > 0 THEN total_sold::float / sale_days 
                ELSE 0 
            END as avg_daily_sales,
            CASE 
                WHEN total_sold > 0 THEN total_revenue::float / total_sold 
                ELSE avg_price 
            END as actual_avg_price,
            warehouse_stock + store_stock as total_stock
        FROM 
            product_metrics
        WHERE 
            warehouse_stock > 0 OR store_stock > 0
        ORDER BY 
            total_revenue DESC
    """)

    min_date = datetime.now() - timedelta(days=30)
    result = await db.execute(query, {"min_date": min_date})
    rows = result.fetchall()

    suggestions = []

    for row in rows:
        product_suggestions = []

        if row.warehouse_stock > 0 and row.store_stock == 0 and row.avg_daily_sales > 0:
            product_suggestions.append({
                "type": "move_to_store",
                "priority": "high",
                "action": "Переместить в магазин",
                "reason": "Товар есть на складе, но отсутствует в магазине",
                "recommended_quantity": min(row.warehouse_stock, int(row.avg_daily_sales * 14))
            })

        if row.total_stock > 0 and row.avg_daily_sales > 0:
            days_of_supply = row.total_stock / row.avg_daily_sales

            if days_of_supply > 90:
                discount_percentage = min(30, int((days_of_supply - 90) / 10) * 5)
                product_suggestions.append({
                    "type": "discount",
                    "priority": "medium",
                    "action": f"Применить скидку {discount_percentage}%",
                    "reason": f"Избыточные запасы на {days_of_supply:.0f} дней",
                    "expected_impact": "Ускорение оборачиваемости товара"
                })

            elif days_of_supply < 7:
                order_quantity = int(row.avg_daily_sales * 30)
                product_suggestions.append({
                    "type": "reorder",
                    "priority": "high",
                    "action": f"Заказать {order_quantity} единиц",
                    "reason": f"Запасов осталось на {days_of_supply:.1f} дней",
                    "expected_impact": "Предотвращение дефицита товара"
                })

        if row.total_stock > 0 and row.sale_days == 0:
            product_suggestions.append({
                "type": "promotion",
                "priority": "high",
                "action": "Запустить промо-кампанию",
                "reason": "Товар не продавался последние 30 дней",
                "expected_impact": "Стимулирование первичного спроса"
            })

        if product_suggestions:
            suggestions.append({
                "product_name": row.name,
                "category": row.category_name,
                "current_metrics": {
                    "warehouse_stock": int(row.warehouse_stock),
                    "store_stock": int(row.store_stock),
                    "avg_daily_sales": float(row.avg_daily_sales),
                    "total_revenue_30d": float(row.total_revenue)
                },
                "suggestions": product_suggestions
            })

    return suggestions[:20]


@router.get("/turnover-analytics", response_model=Dict[str, Any])
async def get_turnover_analytics(
        product_sid: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    base_query = """
        WITH stock_movement AS (
            SELECT 
                p.sid as product_sid,
                p.name as product_name,
                c.name as category_name,
                wi.received_at,
                COALESCE(s.sold_at, CURRENT_DATE) as sold_at,
                wi.quantity as initial_qty,
                COALESCE(s.sold_qty, 0) as sold_qty,
                wi.id as warehouse_item_id
            FROM 
                warehouseitem wi
            JOIN 
                product p ON wi.product_sid = p.sid
            JOIN 
                category c ON p.category_sid = c.sid
            LEFT JOIN 
                storeitem si ON wi.sid = si.warehouse_item_sid
            LEFT JOIN 
                sale s ON si.sid = s.store_item_sid
            WHERE 
                wi.status != 'DISCARDED'
    """

    params = {}
    if product_sid:
        base_query += " AND p.sid = :product_sid"
        params["product_sid"] = product_sid

    base_query += """
        ),
        turnover_metrics AS (
            SELECT 
                product_sid,
                product_name,
                category_name,
                AVG(EXTRACT(EPOCH FROM (sold_at - received_at)) / 86400) as avg_days_to_sell,
                COUNT(DISTINCT warehouse_item_id) as item_count,
                SUM(sold_qty)::float / NULLIF(SUM(initial_qty), 0) as turnover_rate
            FROM 
                stock_movement
            GROUP BY 
                product_sid, product_name, category_name
        ),
        category_turnover AS (
            SELECT 
                category_name,
                AVG(turnover_rate) as avg_turnover_rate,
                AVG(avg_days_to_sell) as avg_days
            FROM 
                turnover_metrics
            GROUP BY 
                category_name
        ),
        slow_moving AS (
            SELECT 
                p.name,
                SUM(wi.quantity) as quantity,
                AVG(wi.wholesale_price * wi.quantity) as value,
                EXTRACT(EPOCH FROM (CURRENT_DATE - MIN(wi.received_at))) / 86400 as days_in_stock
            FROM 
                warehouseitem wi
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                wi.status = 'IN_STOCK'
                AND wi.received_at < CURRENT_DATE - INTERVAL '30 days'
            GROUP BY 
                p.name
            HAVING 
                SUM(wi.quantity) > 0
        ),
        monthly_turnover AS (
            SELECT 
                DATE_TRUNC('month', sold_at) as month,
                SUM(sold_qty)::float / NULLIF(SUM(initial_qty), 0) as turnover_rate
            FROM 
                stock_movement
            WHERE 
                sold_at >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY 
                DATE_TRUNC('month', sold_at)
            ORDER BY 
                month
        )
        SELECT 
            (SELECT AVG(avg_days_to_sell) FROM turnover_metrics) as avg_realization_days,
            (SELECT json_agg(json_build_object(
                'category', category_name,
                'turnoverRate', avg_turnover_rate,
                'avgDays', ROUND(avg_days)
            )) FROM category_turnover) as category_turnover,
            (SELECT json_agg(json_build_object(
                'name', name,
                'daysInStock', ROUND(days_in_stock),
                'quantity', quantity,
                'value', ROUND(value::numeric, 2)
            )) FROM slow_moving LIMIT 20) as slow_moving_products,
            (SELECT json_agg(json_build_object(
                'month', TO_CHAR(month, 'YYYY-MM'),
                'turnoverRate', ROUND(turnover_rate::numeric, 3)
            )) FROM monthly_turnover) as turnover_trend
    """

    result = await db.execute(text(base_query), params)
    row = result.fetchone()

    return {
        "averageRealizationDays": round(row.avg_realization_days or 0),
        "categoryTurnover": row.category_turnover or [],
        "slowMovingProducts": row.slow_moving_products or [],
        "turnoverTrend": row.turnover_trend or []
    }


@router.get("/loss-analytics", response_model=Dict[str, Any])
async def get_loss_analytics(
        product_sid: Optional[str] = None,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    base_query = """
        WITH monthly_losses AS (
            SELECT 
                DATE_TRUNC('month', si.moved_at) as month,
                SUM(CASE WHEN si.status = 'EXPIRED' THEN si.quantity * si.price ELSE 0 END) as expired_loss,
                SUM(CASE WHEN d.percentage > 0 THEN (si.price * d.percentage / 100) * s.sold_qty ELSE 0 END) as discount_loss
            FROM 
                storeitem si
            LEFT JOIN 
                discount d ON si.sid = d.store_item_sid
            LEFT JOIN 
                sale s ON si.sid = s.store_item_sid
            WHERE 
                si.moved_at >= CURRENT_DATE - INTERVAL '12 months'
    """

    params = {}
    if product_sid:
        base_query += """
            AND si.warehouse_item_sid IN (
                SELECT sid FROM warehouseitem WHERE product_sid = :product_sid
            )
        """
        params["product_sid"] = product_sid

    base_query += """
            GROUP BY 
                DATE_TRUNC('month', si.moved_at)
            ORDER BY 
                month
        ),
        discount_effectiveness AS (
            SELECT 
                CASE 
                    WHEN d.percentage < 10 THEN '0-10%'
                    WHEN d.percentage < 20 THEN '10-20%'
                    WHEN d.percentage < 30 THEN '20-30%'
                    ELSE '30%+'
                END as discount_range,
                AVG((s.sold_qty * s.sold_price) / NULLIF((si.quantity * si.price), 0)) * 100 as roi,
                AVG(s.sold_qty)::float / NULLIF(AVG(si.quantity), 0) * 100 as sales_increase
            FROM 
                discount d
            JOIN 
                storeitem si ON d.store_item_sid = si.sid
            JOIN 
                sale s ON si.sid = s.store_item_sid
            WHERE 
                d.starts_at >= CURRENT_DATE - INTERVAL '3 months'
            GROUP BY 
                discount_range
        ),
        expiry_management AS (
            SELECT 
                COUNT(CASE WHEN si.status = 'ACTIVE' AND wi.expire_date > CURRENT_DATE THEN 1 END) as managed_before,
                COUNT(CASE WHEN si.status = 'EXPIRED' THEN 1 END) as expired,
                AVG(CASE 
                    WHEN d.sid IS NOT NULL THEN 
                        EXTRACT(EPOCH FROM (wi.expire_date - d.starts_at)) / 86400 
                    ELSE NULL 
                END) as avg_days_before_action
            FROM 
                storeitem si
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            LEFT JOIN 
                discount d ON si.sid = d.store_item_sid
            WHERE 
                wi.expire_date IS NOT NULL
        )
        SELECT 
            (SELECT json_agg(json_build_object(
                'month', TO_CHAR(month, 'YYYY-MM'),
                'expired', COALESCE(expired_loss, 0),
                'discounted', COALESCE(discount'discounted', COALESCE(discount_loss, 0),
                'total', COALESCE(expired_loss + discount_loss, 0)
            ) ORDER BY month) FROM monthly_losses) as monthly_losses,
            (SELECT json_agg(json_build_object(
                'discountRange', discount_range,
                'roi', ROUND(roi::numeric, 1),
                'salesIncrease', ROUND(sales_increase::numeric, 1)
            )) FROM discount_effectiveness) as discount_roi,
            (SELECT json_build_object(
                'managedBeforeExpiry', managed_before,
                'expired', expired,
                'avgDaysBeforeAction', ROUND(COALESCE(avg_days_before_action, 0))
            ) FROM expiry_management) as expiry_efficiency,
            (SELECT json_build_object(
                'expired', COALESCE(SUM(CASE WHEN si.status = 'EXPIRED' THEN si.quantity * si.price END), 0),
                'discounts', COALESCE(SUM(CASE WHEN d.percentage > 0 THEN (si.price * d.percentage / 100) * s.sold_qty END), 0),
                'total', COALESCE(SUM(CASE WHEN si.status = 'EXPIRED' THEN si.quantity * si.price END), 0) + 
                         COALESCE(SUM(CASE WHEN d.percentage > 0 THEN (si.price * d.percentage / 100) * s.sold_qty END), 0)
            ) FROM storeitem si
            LEFT JOIN discount d ON si.sid = d.store_item_sid
            LEFT JOIN sale s ON si.sid = s.store_item_sid
            WHERE si.moved_at >= CURRENT_DATE - INTERVAL '30 days') as total_losses
    """

    result = await db.execute(text(base_query), params)
    row = result.fetchone()

    return {
        "monthlyLosses": row[0] or [],
        "discountROI": row[1] or [],
        "expiryEfficiency": row[2] or {"managedBeforeExpiry": 0, "expired": 0, "avgDaysBeforeAction": 0},
        "totalLosses": row[3] or {"expired": 0, "discounts": 0, "total": 0}
    }


@router.get("/abc-xyz-analysis", response_model=Dict[str, Any])
async def get_abc_xyz_analysis(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
        WITH product_sales AS (
            SELECT 
                p.sid as product_sid,
                p.name as product_name,
                c.name as category_name,
                COALESCE(SUM(s.sold_qty * s.sold_price), 0) as revenue,
                COALESCE(SUM(s.sold_qty), 0) as quantity,
                COUNT(DISTINCT DATE(s.sold_at)) as sale_days,
                ARRAY_AGG(s.sold_qty) as daily_quantities
            FROM 
                product p
            JOIN 
                category c ON p.category_sid = c.sid
            LEFT JOIN 
                warehouseitem wi ON p.sid = wi.product_sid
            LEFT JOIN 
                storeitem si ON wi.sid = si.warehouse_item_sid
            LEFT JOIN 
                sale s ON si.sid = s.store_item_sid
            WHERE 
                s.sold_at >= CURRENT_DATE - INTERVAL '90 days'
                OR s.sold_at IS NULL
            GROUP BY 
                p.sid, p.name, c.name
        ),
        revenue_ranked AS (
            SELECT 
                *,
                SUM(revenue) OVER () as total_revenue,
                revenue::float / NULLIF(SUM(revenue) OVER (), 0) * 100 as revenue_share,
                SUM(revenue) OVER (ORDER BY revenue DESC) as cumulative_revenue
            FROM 
                product_sales
        ),
        abc_classified AS (
            SELECT 
                *,
                CASE 
                    WHEN cumulative_revenue <= total_revenue * 0.8 THEN 'A'
                    WHEN cumulative_revenue <= total_revenue * 0.95 THEN 'B'
                    ELSE 'C'
                END as abc_class
            FROM 
                revenue_ranked
        ),
        xyz_classified AS (
            SELECT 
                *,
                CASE 
                    WHEN sale_days = 0 THEN 100
                    ELSE (
                        SELECT STDDEV(q)::float / NULLIF(AVG(q), 0) * 100 
                        FROM UNNEST(daily_quantities) q
                    )
                END as demand_variability,
                CASE 
                    WHEN sale_days = 0 THEN 'Z'
                    WHEN (
                        SELECT STDDEV(q)::float / NULLIF(AVG(q), 0) 
                        FROM UNNEST(daily_quantities) q
                    ) < 0.2 THEN 'X'
                    WHEN (
                        SELECT STDDEV(q)::float / NULLIF(AVG(q), 0) 
                        FROM UNNEST(daily_quantities) q
                    ) < 0.5 THEN 'Y'
                    ELSE 'Z'
                END as xyz_class
            FROM 
                abc_classified
        )
        SELECT 
            json_agg(json_build_object(
                'productSid', product_sid,
                'productName', product_name,
                'category', category_name,
                'abcClass', abc_class,
                'xyzClass', xyz_class,
                'revenue', ROUND(revenue::numeric, 2),
                'revenueShare', ROUND(revenue_share::numeric, 2),
                'demandVariability', ROUND(demand_variability::numeric, 2),
                'quantity', quantity
            ) ORDER BY revenue DESC) as matrix,
            json_build_object(
                'A', json_build_object(
                    'count', COUNT(CASE WHEN abc_class = 'A' THEN 1 END),
                    'revenueShare', ROUND(SUM(CASE WHEN abc_class = 'A' THEN revenue_share ELSE 0 END)::numeric, 1)
                ),
                'B', json_build_object(
                    'count', COUNT(CASE WHEN abc_class = 'B' THEN 1 END),
                    'revenueShare', ROUND(SUM(CASE WHEN abc_class = 'B' THEN revenue_share ELSE 0 END)::numeric, 1)
                ),
                'C', json_build_object(
                    'count', COUNT(CASE WHEN abc_class = 'C' THEN 1 END),
                    'revenueShare', ROUND(SUM(CASE WHEN abc_class = 'C' THEN revenue_share ELSE 0 END)::numeric, 1)
                ),
                'X', json_build_object('count', COUNT(CASE WHEN xyz_class = 'X' THEN 1 END)),
                'Y', json_build_object('count', COUNT(CASE WHEN xyz_class = 'Y' THEN 1 END)),
                'Z', json_build_object('count', COUNT(CASE WHEN xyz_class = 'Z' THEN 1 END))
            ) as summary
        FROM 
            xyz_classified
    """)

    result = await db.execute(query)
    row = result.fetchone()

    return {
        "matrix": row[0] or [],
        "summary": row[1] or {
            "A": {"count": 0, "revenueShare": 0},
            "B": {"count": 0, "revenueShare": 0},
            "C": {"count": 0, "revenueShare": 0},
            "X": {"count": 0},
            "Y": {"count": 0},
            "Z": {"count": 0}
        }
    }


@router.get("/financial-metrics", response_model=Dict[str, Any])
async def get_financial_metrics(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    query = text("""
        WITH category_margins AS (
            SELECT 
                c.name as category_name,
                SUM(s.sold_qty * s.sold_price) as revenue,
                SUM(s.sold_qty * (s.sold_price - COALESCE(wi.wholesale_price, p.default_price * 0.6))) as gross_profit,
                AVG((s.sold_price - COALESCE(wi.wholesale_price, p.default_price * 0.6)) / NULLIF(s.sold_price, 0) * 100) as margin
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
            WHERE 
                s.sold_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY 
                c.name
        ),
        monthly_avg_check AS (
            SELECT 
                DATE_TRUNC('month', s.sold_at) as month,
                SUM(s.sold_qty * s.sold_price) / COUNT(DISTINCT s.cashier_sid) as avg_check,
                COUNT(DISTINCT s.sid) as transaction_count
            FROM 
                sale s
            WHERE 
                s.sold_at >= CURRENT_DATE - INTERVAL '6 months'
            GROUP BY 
                DATE_TRUNC('month', s.sold_at)
            ORDER BY 
                month
        ),
        product_ltv AS (
            SELECT 
                p.name as product_name,
                COUNT(DISTINCT s.cashier_sid) as unique_customers,
                SUM(s.sold_qty * s.sold_price) as total_revenue,
                COUNT(DISTINCT DATE_TRUNC('month', s.sold_at)) as active_months,
                AVG(s.sold_qty * s.sold_price) as avg_order_value,
                COUNT(s.sid)::float / COUNT(DISTINCT s.cashier_sid) as avg_purchase_frequency
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                s.sold_at >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY 
                p.name
            HAVING 
                COUNT(DISTINCT s.cashier_sid) >= 3
        ),
        profitability AS (
            SELECT 
                SUM(s.sold_qty * s.sold_price) as total_revenue,
                SUM(s.sold_qty * (s.sold_price - COALESCE(wi.wholesale_price, p.default_price * 0.6))) as gross_profit,
                SUM(CASE WHEN si.status = 'EXPIRED' THEN si.quantity * wi.wholesale_price ELSE 0 END) as losses
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                s.sold_at >= CURRENT_DATE - INTERVAL '30 days'
        )
        SELECT 
            (SELECT json_agg(json_build_object(
                'category', category_name,
                'margin', ROUND(margin::numeric, 1),
                'revenue', ROUND(revenue::numeric, 2)
            ) ORDER BY revenue DESC) FROM category_margins) as margin_by_category,
            (SELECT json_agg(json_build_object(
                'month', TO_CHAR(month, 'YYYY-MM'),
                'avgCheck', ROUND(avg_check::numeric, 2),
                'transactionCount', transaction_count
            ) ORDER BY month) FROM monthly_avg_check) as avg_check_trend,
            (SELECT json_agg(json_build_object(
                'productName', product_name,
                'ltv', ROUND((total_revenue / unique_customers)::numeric, 2),
                'avgPurchaseFrequency', ROUND(avg_purchase_frequency::numeric, 2),
                'avgOrderValue', ROUND(avg_order_value::numeric, 2)
            ) ORDER BY total_revenue / unique_customers DESC LIMIT 20) FROM product_ltv) as product_ltv,
            (SELECT json_build_object(
                'grossProfit', ROUND(gross_profit::numeric, 2),
                'netProfit', ROUND((gross_profit - losses)::numeric, 2),
                'grossMargin', ROUND((gross_profit / NULLIF(total_revenue, 0) * 100)::numeric, 1),
                'netMargin', ROUND(((gross_profit - losses) / NULLIF(total_revenue, 0) * 100)::numeric, 1),
                'lossesImpact', ROUND((losses / NULLIF(gross_profit, 0) * 100)::numeric, 1)
            ) FROM profitability) as profitability
    """)

    result = await db.execute(query)
    row = result.fetchone()

    return {
        "marginByCategory": row[0] or [],
        "avgCheckTrend": row[1] or [],
        "productLTV": row[2] or [],
        "profitability": row[3] or {
            "grossProfit": 0,
            "netProfit": 0,
            "grossMargin": 0,
            "netMargin": 0,
            "lossesImpact": 0
        }
    }


@router.get("/optimal-purchase/{product_sid}", response_model=Dict[str, Any])
async def get_optimal_purchase(
        product_sid: str,
        lead_time_days: int = Query(7, ge=1, le=30),
        service_level: float = Query(0.95, ge=0.8, le=0.99),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    prediction_service = PredictionService(db)

    query = text("""
        WITH sales_stats AS (
            SELECT 
                AVG(s.sold_qty) as avg_daily_demand,
                STDDEV(s.sold_qty) as demand_std,
                COUNT(DISTINCT DATE(s.sold_at)) as sale_days
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            WHERE 
                wi.product_sid = :product_sid
                AND s.sold_at >= CURRENT_DATE - INTERVAL '30 days'
        ),
        current_inventory AS (
            SELECT 
                COALESCE(SUM(wi.quantity), 0) + COALESCE(SUM(si.quantity), 0) as total_stock,
                COALESCE(AVG(wi.wholesale_price), p.default_price * 0.6) as unit_cost
            FROM 
                product p
            LEFT JOIN 
                warehouseitem wi ON p.sid = wi.product_sid AND wi.status = 'IN_STOCK'
            LEFT JOIN 
                storeitem si ON wi.sid = si.warehouse_item_sid AND si.status = 'ACTIVE'
            WHERE 
                p.sid = :product_sid
            GROUP BY 
                p.default_price
        )
        SELECT 
            ss.avg_daily_demand,
            ss.demand_std,
            ss.sale_days,
            ci.total_stock,
            ci.unit_cost
        FROM 
            sales_stats ss, current_inventory ci
    """)

    result = await db.execute(query, {"product_sid": product_sid})
    row = result.fetchone()

    if not row or row.sale_days < 7:
        raise HTTPException(
            status_code=400,
            detail="Not enough sales data for optimal purchase calculation"
        )

    avg_daily_demand = float(row.avg_daily_demand)
    demand_std = float(row.demand_std) if row.demand_std else avg_daily_demand * 0.2
    current_stock = float(row.total_stock)
    unit_cost = float(row.unit_cost)

    from scipy.stats import norm
    z_score = norm.ppf(service_level)

    safety_stock = z_score * demand_std * np.sqrt(lead_time_days)
    reorder_point = avg_daily_demand * lead_time_days + safety_stock

    holding_cost_rate = 0.2
    ordering_cost = 100

    eoq = np.sqrt((2 * ordering_cost * avg_daily_demand * 365) / (unit_cost * holding_cost_rate))

    days_until_reorder = max(0, (current_stock - reorder_point) / avg_daily_demand)
    next_order_date = datetime.now() + timedelta(days=int(days_until_reorder))

    stockout_risk = current_stock < reorder_point

    projection_days = 30
    stock_projection = []
    for day in range(projection_days):
        date = datetime.now() + timedelta(days=day)
        projected_stock = max(0, current_stock - avg_daily_demand * day)

        if day == int(days_until_reorder) and projected_stock < reorder_point:
            projected_stock += eoq

        stock_projection.append({
            "date": date.strftime("%Y-%m-%d"),
            "stock": round(projected_stock),
            "reorderPoint": round(reorder_point)
        })

    return {
        "reorderPoint": round(reorder_point),
        "safetyStock": round(safety_stock),
        "economicOrderQuantity": round(eoq),
        "leadTimeDays": lead_time_days,
        "currentStock": round(current_stock),
        "averageDailyDemand": round(avg_daily_demand, 2),
        "stockoutRisk": stockout_risk,
        "nextOrderDate": next_order_date.strftime("%Y-%m-%d"),
        "stockProjection": stock_projection
    }