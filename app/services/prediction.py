import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy import text, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from prophet import Prophet
import warnings
import hashlib
import json

warnings.filterwarnings('ignore')

from app.models.inventory import Prediction, Product, TimeFrame
from app.models.base import Base

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: AsyncSession, user_sid: str):
        self.db = db
        self.user_sid = user_sid
        self.model_version = "prophet_v2.0"

    async def get_sales_hash(self, product_sid: str) -> str:
        """Получить хэш данных о продажах для проверки изменений"""
        query = text("""
            SELECT 
                DATE(s.sold_at) as date,
                SUM(s.sold_qty) as quantity
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                upload u ON wi.upload_sid = u.sid
            WHERE 
                wi.product_sid = :product_sid
                AND u.user_sid = :user_sid
            GROUP BY 
                DATE(s.sold_at)
            ORDER BY 
                date
        """)

        result = await self.db.execute(
            query,
            {"product_sid": product_sid, "user_sid": self.user_sid}
        )
        rows = result.fetchall()

        data_str = json.dumps([(str(row.date), float(row.quantity)) for row in rows])
        return hashlib.md5(data_str.encode()).hexdigest()

    async def check_if_forecast_needed(self, product_sid: str) -> bool:
        """Проверить, нужно ли обновлять прогноз"""
        current_hash = await self.get_sales_hash(product_sid)

        last_forecast_query = await self.db.execute(
            text("""
                SELECT 
                    meta_info::json->>'sales_hash' as sales_hash,
                    generated_at
                FROM prediction
                WHERE product_sid = :product_sid
                AND user_sid = :user_sid
                ORDER BY generated_at DESC
                LIMIT 1
            """),
            {"product_sid": product_sid, "user_sid": self.user_sid}
        )
        last_forecast = last_forecast_query.fetchone()

        # Обновляем если:
        # 1. Нет прогнозов вообще
        # 2. Изменились данные о продажах
        # 3. Прогноз старше 7 дней
        if not last_forecast:
            return True

        if last_forecast.sales_hash != current_hash:
            return True

        if (datetime.now() - last_forecast.generated_at).days > 7:
            return True

        return False

    async def get_sales_data(self, product_sid: str) -> pd.DataFrame:
        """Получить данные о продажах для обучения модели"""
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
                upload u ON wi.upload_sid = u.sid
            WHERE 
                wi.product_sid = :product_sid
                AND u.user_sid = :user_sid
            GROUP BY 
                DATE(s.sold_at)
            ORDER BY 
                ds
        """)

        result = await self.db.execute(
            query,
            {"product_sid": product_sid, "user_sid": self.user_sid}
        )
        rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([
            {"ds": row.ds, "y": float(row.y)}
            for row in rows
        ])

        df['ds'] = pd.to_datetime(df['ds'])

        # Заполняем пропущенные дни нулями
        date_range = pd.date_range(
            start=df['ds'].min(),
            end=datetime.now().date(),
            freq='D'
        )

        date_df = pd.DataFrame({"ds": date_range})
        merged_df = pd.merge(date_df, df, on='ds', how='left')
        merged_df['y'] = merged_df['y'].fillna(0)

        # Добавляем дополнительные признаки для лучшего прогнозирования
        merged_df['day_of_week'] = merged_df['ds'].dt.dayofweek
        merged_df['day_of_month'] = merged_df['ds'].dt.day
        merged_df['month'] = merged_df['ds'].dt.month

        return merged_df

    async def generate_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame = TimeFrame.DAY,
            periods_ahead: int = 90
    ) -> List[Dict[str, Any]]:
        """Генерировать прогноз на указанный период"""
        try:
            # Проверяем, нужен ли новый прогноз
            if not await self.check_if_forecast_needed(product_sid):
                logger.info(f"Forecast for product {product_sid} is up to date")
                return []

            # Получаем данные о продажах
            df = await self.get_sales_data(product_sid)

            if df.empty or len(df) < 14:
                logger.warning(f"Not enough data for product {product_sid}: {len(df)} days")
                return []

            # Настраиваем и обучаем модель Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                interval_width=0.95,
                growth='linear'
            )

            # Добавляем дополнительные регрессоры если есть достаточно данных
            if len(df) > 30:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            # Обучаем модель
            model.fit(df[['ds', 'y']])

            # Создаем датафрейм для прогноза
            future = model.make_future_dataframe(periods=periods_ahead)
            forecast = model.predict(future)

            # Получаем текущую дату
            today = datetime.now().date()
            results = []
            sales_hash = await self.get_sales_hash(product_sid)

            # Формируем результаты прогноза
            for i in range(periods_ahead):
                if timeframe == TimeFrame.DAY:
                    period_start = today + timedelta(days=i + 1)
                    period_end = period_start
                elif timeframe == TimeFrame.WEEK:
                    period_start = today + timedelta(weeks=i + 1)
                    period_end = period_start + timedelta(days=6)
                elif timeframe == TimeFrame.MONTH:
                    period_start = today + timedelta(days=30 * (i + 1))
                    period_end = period_start + timedelta(days=29)

                # Получаем прогноз на конкретную дату
                forecast_row = forecast[forecast['ds'].dt.date == period_start]

                if not forecast_row.empty:
                    forecast_qty = max(0, forecast_row['yhat'].iloc[0])
                    forecast_lower = max(0, forecast_row['yhat_lower'].iloc[0])
                    forecast_upper = max(0, forecast_row['yhat_upper'].iloc[0])
                else:
                    # Если нет точного прогноза, используем среднее
                    forecast_qty = df['y'].mean()
                    forecast_lower = forecast_qty * 0.8
                    forecast_upper = forecast_qty * 1.2

                results.append({
                    "product_sid": product_sid,
                    "timeframe": timeframe,
                    "period_start": period_start,
                    "period_end": period_end,
                    "forecast_qty": round(float(forecast_qty), 2),
                    "forecast_qty_lower": round(float(forecast_lower), 2),
                    "forecast_qty_upper": round(float(forecast_upper), 2),
                    "generated_at": datetime.now(),
                    "model_version": self.model_version,
                    "user_sid": self.user_sid,
                    "meta_info": {
                        "sales_hash": sales_hash,
                        "training_days": len(df),
                        "avg_sales": float(df['y'].mean()),
                        "std_sales": float(df['y'].std())
                    }
                })

            logger.info(f"Generated {len(results)} forecasts for product {product_sid}")
            return results

        except Exception as e:
            logger.error(f"Error generating forecast for product {product_sid}: {str(e)}")
            return []

    async def save_forecast(self, forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Сохранить прогнозы в базу данных"""
        if not forecasts:
            return []

        saved_predictions = []

        try:
            # Удаляем старые прогнозы для этого продукта
            await self.db.execute(
                delete(Prediction).where(
                    and_(
                        Prediction.product_sid == forecasts[0]["product_sid"],
                        Prediction.user_sid == self.user_sid,
                        Prediction.period_start >= datetime.now().date()
                    )
                )
            )

            # Сохраняем новые прогнозы
            for forecast in forecasts:
                prediction = Prediction(
                    sid=Base.generate_sid(),
                    product_sid=forecast["product_sid"],
                    user_sid=self.user_sid,
                    timeframe=forecast["timeframe"],
                    period_start=forecast["period_start"],
                    period_end=forecast["period_end"],
                    forecast_qty=forecast["forecast_qty"],
                    forecast_qty_lower=forecast.get("forecast_qty_lower"),
                    forecast_qty_upper=forecast.get("forecast_qty_upper"),
                    generated_at=forecast["generated_at"],
                    model_version=forecast["model_version"],
                    meta_info=json.dumps(forecast.get("meta_info", {}))
                )

                self.db.add(prediction)

                saved_predictions.append({
                    "sid": prediction.sid,
                    **forecast
                })

            await self.db.commit()
            logger.info(f"Saved {len(saved_predictions)} predictions")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error saving predictions: {str(e)}")
            raise

        return saved_predictions