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
        refresh: bool = False,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    prediction_service = PredictionService(db, current_user.sid)

    if not refresh:
        existing_query = await db.execute(
            select(Prediction)
            .where(
                Prediction.product_sid == product_sid,
                Prediction.user_sid == current_user.sid,
                Prediction.period_start >= datetime.now().date()
            )
            .order_by(Prediction.period_start.asc())
            .limit(30)
        )
        existing = existing_query.scalars().all()

        if existing and len(existing) >= 7:
            return existing

    forecasts = await prediction_service.generate_forecast(
        product_sid=product_sid,
        timeframe=TimeFrame.DAY,
        periods_ahead=30
    )

    if not forecasts:
        raise HTTPException(
            status_code=400,
            detail="Not enough sales data to generate forecast"
        )

    saved_predictions = await prediction_service.save_forecast(forecasts)

    predictions = []
    for pred in saved_predictions:
        predictions.append(PredictionResponse(
            sid=pred["sid"],
            product_sid=pred["product_sid"],
            timeframe=pred["timeframe"],
            period_start=pred["period_start"],
            period_end=pred["period_end"],
            forecast_qty=pred["forecast_qty"],
            generated_at=pred["generated_at"],
            model_version=pred["model_version"],
            forecast_qty_lower=pred.get("forecast_qty_lower", pred["forecast_qty"] * 0.8),
            forecast_qty_upper=pred.get("forecast_qty_upper", pred["forecast_qty"] * 1.2)
        ))

    return predictions