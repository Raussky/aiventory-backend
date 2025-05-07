# app/api/v1/prediction.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Prediction, Product, TimeFrame
from app.schemas.prediction import (
    PredictionResponse, PredictionCreate, PredictionRequest,
    PredictionStatResponse, ProductAnalyticsResponse
)
from app.core.dependencies import get_current_user
from app.services.prediction import PredictionService

router = APIRouter()


@router.get("/forecast/{product_sid}", response_model=List[PredictionResponse])
async def get_forecast(
        product_sid: str,
        refresh: bool = False,
        timeframe: TimeFrame = TimeFrame.MONTH,
        periods: int = 3,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get demand forecast for a product"""
    # Check product exists
    product_query = await db.execute(
        select(Product).where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # If no refresh requested, try to find existing forecasts
    if not refresh:
        # Find latest forecasts with this time frame
        predictions_query = await db.execute(
            select(Prediction)
            .where(
                Prediction.product_sid == product_sid,
                Prediction.timeframe == timeframe,
                Prediction.period_start >= datetime.now().date()
            )
            .order_by(Prediction.generated_at.desc())
            .limit(periods)
        )
        predictions = predictions_query.scalars().all()

        # If we have recent predictions, return them
        if predictions and len(predictions) == periods:
            return predictions

    # Generate new forecast
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

    # Save forecasts
    predictions = await prediction_service.save_forecast(forecasts)

    return predictions


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
    """Get sales statistics for visualization"""
    # Set default date range if not provided
    if not start_date:
        start_date = datetime.now() - timedelta(days=90)

    if not end_date:
        end_date = datetime.now()

    # Build query based on grouping option
    if group_by == "day":
        date_format = "DATE(s.sold_at)"
    elif group_by == "week":
        date_format = "DATE_TRUNC('week', s.sold_at)"
    else:  # month
        date_format = "DATE_TRUNC('month', s.sold_at)"

    # Start building the query
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

        # Convert to dataframe for easier manipulation
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

        # Format date properly
        df["date_str"] = df["date"].apply(
            lambda x: x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else str(x)
        )

        # Get unique dates, products and categories
        dates = df["date_str"].unique().tolist()
        products = df[["product_sid", "product_name"]].drop_duplicates().to_dict("records")
        categories = df[["category_sid", "category_name"]].drop_duplicates().to_dict("records")

        # Create pivot tables
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

        # Add category totals if not filtering by product
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


@router.get("/trends/{product_sid}", response_model=Dict[str, Any])
async def get_product_trends(
        product_sid: str,
        days_back: int = Query(90, ge=7, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get sales trends for a specific product"""
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


@router.get("/category-trends/{category_sid}", response_model=Dict[str, Any])
async def get_category_trends(
        category_sid: str,
        days_back: int = Query(90, ge=7, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get sales trends for a product category"""
    from app.models.inventory import Category

    category_query = await db.execute(
        select(Category).where(Category.sid == category_sid)
    )
    category = category_query.scalar_one_or_none()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    prediction_service = PredictionService(db)
    trends = await prediction_service.get_sales_trends(
        category_sid=category_sid,
        days_back=days_back
    )

    return trends


@router.get("/seasonality/{category_sid}", response_model=Dict[str, Any])
async def get_category_seasonality(
        category_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get seasonality patterns for a product category"""
    from app.models.inventory import Category

    category_query = await db.execute(
        select(Category).where(Category.sid == category_sid)
    )
    category = category_query.scalar_one_or_none()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    prediction_service = PredictionService(db)
    seasonality = await prediction_service.get_category_seasonality(category_sid)

    return seasonality


@router.get("/analytics/{product_sid}", response_model=Dict[str, Any])
async def get_product_analytics(
        product_sid: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get comprehensive analytics for a specific product"""
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


@router.get("/top-products", response_model=List[Dict[str, Any]])
async def get_top_products(
        limit: int = Query(10, ge=1, le=50),
        days_back: int = Query(30, ge=1, le=365),
        metric: str = Query("quantity", regex="^(quantity|revenue)$"),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get top-selling products by quantity or revenue"""
    query = text("""
        SELECT 
            p.sid as product_sid,
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
        WHERE 
            s.sold_at >= :min_date
        GROUP BY 
            p.sid, p.name, c.name
        ORDER BY 
            :order_field DESC
        LIMIT :limit
    """)

    min_date = datetime.now() - timedelta(days=days_back)

    # Use a literal to make the SQL order by work
    order_field = "revenue" if metric == "revenue" else "quantity"

    try:
        params = {"min_date": min_date, "limit": limit, "order_field": order_field}
        result = await db.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "product_sid": row.product_sid,
                "product_name": row.product_name,
                "category_name": row.category_name,
                "quantity": float(row.quantity),
                "revenue": float(row.revenue)
            }
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching top products: {str(e)}")


@router.get("/dashboard-metrics", response_model=Dict[str, Any])
async def get_dashboard_metrics(
        days_back: int = Query(30, ge=1, le=365),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Get summary metrics for dashboard"""
    query = text("""
        SELECT 
            COUNT(DISTINCT p.sid) as product_count,
            COUNT(DISTINCT c.sid) as category_count,
            SUM(s.sold_qty) as total_quantity,
            SUM(s.sold_qty * s.sold_price) as total_revenue,
            COUNT(DISTINCT DATE(s.sold_at)) as sales_days,
            COUNT(DISTINCT wi.sid) as warehouse_items_count,
            SUM(wi.quantity) as warehouse_total_quantity
        FROM 
            product p
        LEFT JOIN
            category c ON p.category_sid = c.sid
        LEFT JOIN
            warehouseitem wi ON wi.product_sid = p.sid AND wi.status = 'IN_STOCK'
        LEFT JOIN
            storeitem si ON si.warehouse_item_sid = wi.sid AND si.status = 'ACTIVE'
        LEFT JOIN
            sale s ON s.store_item_sid = si.sid AND s.sold_at >= :min_date
    """)

    min_date = datetime.now() - timedelta(days=days_back)

    try:
        result = await db.execute(query, {"min_date": min_date})
        row = result.fetchone()

        if not row:
            return {
                "product_count": 0,
                "category_count": 0,
                "total_quantity": 0,
                "total_revenue": 0,
                "avg_daily_sales": 0,
                "warehouse_items_count": 0,
                "warehouse_total_quantity": 0
            }

        # Calculate average daily sales
        avg_daily_sales = row.total_quantity / max(1, row.sales_days) if row.total_quantity else 0

        # Get top categories
        top_categories_query = text("""
            SELECT 
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
            WHERE 
                s.sold_at >= :min_date
            GROUP BY 
                c.name
            ORDER BY 
                revenue DESC
            LIMIT 5
        """)

        categories_result = await db.execute(top_categories_query, {"min_date": min_date})
        categories_rows = categories_result.fetchall()

        top_categories = [
            {
                "category_name": row.category_name,
                "quantity": float(row.quantity),
                "revenue": float(row.revenue)
            }
            for row in categories_rows
        ]

        return {
            "product_count": row.product_count or 0,
            "category_count": row.category_count or 0,
            "total_quantity": float(row.total_quantity) if row.total_quantity else 0,
            "total_revenue": float(row.total_revenue) if row.total_revenue else 0,
            "avg_daily_sales": float(avg_daily_sales),
            "warehouse_items_count": row.warehouse_items_count or 0,
            "warehouse_total_quantity": float(row.warehouse_total_quantity) if row.warehouse_total_quantity else 0,
            "top_categories": top_categories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard metrics: {str(e)}")