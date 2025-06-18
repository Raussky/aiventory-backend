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

        if not last_forecast:
            logger.info(f"No forecast found for product {product_sid}")
            return True

        if last_forecast.sales_hash != current_hash:
            logger.info(f"Sales data changed for product {product_sid}")
            return True

        if (datetime.now() - last_forecast.generated_at).days > 7:
            logger.info(f"Forecast is outdated for product {product_sid}")
            return True

        return False

    async def get_sales_data(self, product_sid: str) -> pd.DataFrame:
        query = text("""
            WITH daily_sales AS (
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
                JOIN 
                    product p ON wi.product_sid = p.sid
                WHERE 
                    p.sid = :product_sid
                    AND u.user_sid = :user_sid
                    AND s.sold_at >= :min_date
                GROUP BY 
                    DATE(s.sold_at)
            )
            SELECT ds, y FROM daily_sales
            ORDER BY ds
        """)

        min_date = datetime.now() - timedelta(days=120)

        result = await self.db.execute(
            query,
            {"product_sid": product_sid, "user_sid": self.user_sid, "min_date": min_date}
        )
        rows = result.fetchall()

        if not rows:
            logger.warning(f"No sales data found for product {product_sid}")
            return pd.DataFrame()

        df = pd.DataFrame([
            {"ds": row.ds, "y": float(row.y)}
            for row in rows
        ])

        logger.info(f"Found {len(df)} days of sales data for product {product_sid}")

        unique_dates = df['ds'].nunique()
        if unique_dates < 7:
            logger.warning(f"Only {unique_dates} unique dates found for product {product_sid}")
            return pd.DataFrame()

        df['ds'] = pd.to_datetime(df['ds'])

        date_range = pd.date_range(
            start=df['ds'].min(),
            end=datetime.now().date(),
            freq='D'
        )

        date_df = pd.DataFrame({"ds": date_range})
        merged_df = pd.merge(date_df, df, on='ds', how='left')
        merged_df['y'] = merged_df['y'].fillna(0)

        merged_df['day_of_week'] = merged_df['ds'].dt.dayofweek
        merged_df['day_of_month'] = merged_df['ds'].dt.day
        merged_df['month'] = merged_df['ds'].dt.month

        logger.info(f"Final dataset has {len(merged_df)} days for product {product_sid}")
        return merged_df

    async def generate_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame = TimeFrame.DAY,
            periods_ahead: int = 90
    ) -> List[Dict[str, Any]]:
        try:
            df = await self.get_sales_data(product_sid)

            if df.empty:
                logger.error(f"No sales data available for product {product_sid}")
                return []

            if len(df) < 7:
                logger.error(f"Not enough data for product {product_sid}: only {len(df)} days")
                return []

            non_zero_sales = df[df['y'] > 0]
            if len(non_zero_sales) < 3:
                logger.error(f"Not enough non-zero sales for product {product_sid}: only {len(non_zero_sales)} days")
                return []

            logger.info(f"Training Prophet model for product {product_sid} with {len(df)} days of data")

            if len(df) < 30:
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.1,
                    interval_width=0.95,
                    growth='linear'
                )
            else:
                model = Prophet(
                    yearly_seasonality=len(df) > 365,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    interval_width=0.95,
                    growth='linear'
                )

                if len(df) > 60:
                    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            train_df = df[['ds', 'y']].copy()
            model.fit(train_df)

            future = model.make_future_dataframe(periods=periods_ahead)
            forecast = model.predict(future)

            sales_hash = await self.get_sales_hash(product_sid)

            today = datetime.now().date()
            results = []

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

                forecast_row = forecast[forecast['ds'].dt.date == period_start]

                if not forecast_row.empty:
                    forecast_qty = max(0, forecast_row['yhat'].iloc[0])
                    forecast_lower = max(0, forecast_row['yhat_lower'].iloc[0])
                    forecast_upper = max(0, forecast_row['yhat_upper'].iloc[0])
                else:
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
                        "std_sales": float(df['y'].std()) if len(df) > 1 else 0,
                        "non_zero_days": int((df['y'] > 0).sum())
                    }
                })

            logger.info(f"Successfully generated {len(results)} forecasts for product {product_sid}")
            return results

        except Exception as e:
            logger.error(f"Error generating forecast for product {product_sid}: {str(e)}", exc_info=True)
            return []

    async def save_forecast(self, forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not forecasts:
            return []

        saved_predictions = []

        try:
            await self.db.execute(
                delete(Prediction).where(
                    and_(
                        Prediction.product_sid == forecasts[0]["product_sid"],
                        Prediction.user_sid == self.user_sid,
                        Prediction.period_start >= datetime.now().date()
                    )
                )
            )

            for forecast in forecasts:
                meta_info_str = json.dumps(forecast.get("meta_info", {}))

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
                    created_at=forecast["generated_at"],
                    model_version=forecast["model_version"],
                    meta_info=meta_info_str
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
            logger.error(f"Error saving predictions: {str(e)}", exc_info=True)
            raise

        return saved_predictions