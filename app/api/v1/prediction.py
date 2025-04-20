# app/api/v1/prediction.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from app.db.session import get_db
from app.models.users import User
from app.models.inventory import Prediction, Product, TimeFrame
from app.schemas.prediction import PredictionResponse, PredictionCreate
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
    """Получает прогноз для товара"""
    # Проверяем существование продукта
    product_query = await db.execute(
        select(Product).where(Product.sid == product_sid)
    )
    product = product_query.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Если не требуется обновление, пытаемся найти существующий прогноз
    if not refresh:
        # Находим последние прогнозы с этой гранулярностью времени
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

        # Если есть недавние прогнозы, возвращаем их
        if predictions and len(predictions) == periods:
            return predictions

    # Иначе генерируем новый прогноз
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

    # Сохраняем прогнозы в базу
    predictions = await prediction_service.save_forecast(forecasts)

    return predictions


@router.get("/stats", response_model=Dict[str, Any])
async def get_prediction_stats(
        product_sid: Optional[str] = None,
        start_date: datetime = Query(None),
        end_date: datetime = Query(None),
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
):
    """Получает статистику продаж для построения графиков"""
    # Определяем параметры запроса
    if not start_date:
        start_date = datetime.now() - timedelta(days=30)

    if not end_date:
        end_date = datetime.now()

    # Формируем SQL запрос
    query_text = """
        SELECT 
            DATE(s.sold_at) as date,
            p.name as product_name,
            p.sid as product_sid,
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
        WHERE 
            s.sold_at BETWEEN :start_date AND :end_date
    """

    params = {"start_date": start_date, "end_date": end_date}

    if product_sid:
        query_text += " AND p.sid = :product_sid"
        params["product_sid"] = product_sid

    query_text += """
        GROUP BY 
            DATE(s.sold_at), p.name, p.sid
        ORDER BY 
            date, product_name
    """

    result = await db.execute(text(query_text), params)
    rows = result.fetchall()

    # Преобразуем результаты в удобный формат для графиков
    df = pd.DataFrame(rows, columns=["date", "product_name", "product_sid", "quantity", "revenue"])

    if df.empty:
        return {
            "dates": [],
            "products": [],
            "data": []
        }

    # Преобразуем данные для построения графиков
    dates = df["date"].unique().tolist()
    products = df[["product_sid", "product_name"]].drop_duplicates().to_dict("records")

    # Пивот по товарам и датам
    quantity_pivot = df.pivot_table(
        index="date",
        columns="product_sid",
        values="quantity",
        fill_value=0
    ).reset_index().to_dict("records")

    revenue_pivot = df.pivot_table(
        index="date",
        columns="product_sid",
        values="revenue",
        fill_value=0
    ).reset_index().to_dict("records")

    return {
        "dates": dates,
        "products": products,
        "quantity_data": quantity_pivot,
        "revenue_data": revenue_pivot
    }