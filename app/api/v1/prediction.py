from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
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