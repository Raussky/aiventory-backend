# app/services/prediction.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Импорт моделей
from app.models.inventory import Prediction, Product, TimeFrame
from app.models.base import Base  # Импортируем Base на уровне модуля

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_version = "stub_v1.0.0"

    async def get_sales_data(self, product_sid: str) -> pd.DataFrame:
        """Получает исторические данные продаж для продукта"""
        # Используем реальный SQL-запрос из оригинального кода
        query = text("""
            SELECT 
                DATE(s.sold_at) as ds,
                SUM(s.sold_qty) as y
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                p.sid = :product_sid
            GROUP BY 
                DATE(s.sold_at)
            ORDER BY 
                ds
        """)

        try:
            result = await self.db.execute(query, {"product_sid": product_sid})
            rows = result.fetchall()

            if not rows:
                logger.warning(f"No sales data found for product {product_sid}")
                return pd.DataFrame(columns=["ds", "y"])

            # Преобразуем результаты в DataFrame
            df = pd.DataFrame(rows, columns=["ds", "y"])
            return df
        except Exception as e:
            logger.error(f"Error getting sales data: {str(e)}")
            return pd.DataFrame(columns=["ds", "y"])

    async def train_prophet_model(self, product_sid: str) -> Optional[bool]:
        """Заглушка для метода обучения модели - для совместимости интерфейса"""
        df = await self.get_sales_data(product_sid)

        if df.empty or len(df) < 7:
            logger.warning(f"Not enough data to train model for product {product_sid}")
            return None

        # Для заглушки просто возвращаем True, если есть данные
        return True

    async def generate_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame,
            periods_ahead: int = 1
    ) -> List[Dict[str, Any]]:
        """Генерирует заглушку прогноза спроса для продукта"""
        # Проверяем наличие данных (опционально)
        model_trained = await self.train_prophet_model(product_sid)
        if not model_trained:
            # Если нет данных, возвращаем пустой список
            logger.warning(f"No data available for forecasting product {product_sid}")
            return []

        today = datetime.now().date()

        # Определяем период прогнозирования
        if timeframe == TimeFrame.DAY:
            days_per_period = 1
        elif timeframe == TimeFrame.WEEK:
            days_per_period = 7
        else:  # MONTH
            days_per_period = 30

        results = []
        for i in range(periods_ahead):
            period_start = today + timedelta(days=i * days_per_period)
            period_end = period_start + timedelta(days=days_per_period - 1)

            # Генерируем случайный прогноз
            forecast_qty = float(np.random.randint(50, 150))

            results.append({
                "product_sid": product_sid,
                "timeframe": timeframe,
                "period_start": period_start,
                "period_end": period_end,
                "forecast_qty": forecast_qty,
                "generated_at": datetime.now(),
                "model_version": f"{self.model_version} (stub)"
            })

        return results

    async def save_forecast(self, forecasts: List[Dict[str, Any]]) -> List[Prediction]:
        """Сохраняет прогнозы в базу данных"""
        prediction_objects = []

        for forecast in forecasts:
            prediction = Prediction(
                sid=Base.generate_sid(),
                product_sid=forecast["product_sid"],
                timeframe=forecast["timeframe"],
                period_start=forecast["period_start"],
                period_end=forecast["period_end"],
                forecast_qty=forecast["forecast_qty"],
                generated_at=forecast["generated_at"],
                model_version=forecast["model_version"]
            )
            self.db.add(prediction)
            prediction_objects.append(prediction)

        await self.db.commit()

        # Обновляем объекты после коммита
        for prediction in prediction_objects:
            await self.db.refresh(prediction)

        return prediction_objects