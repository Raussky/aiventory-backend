import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')

from app.models.inventory import Prediction, Product, TimeFrame, Category
from app.models.base import Base

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_version = "ensemble_v2.0.0"

    async def get_sales_data(self, product_sid: str, days_back: int = 180) -> pd.DataFrame:
        query = text("""
            SELECT 
                DATE(s.sold_at) as ds,
                SUM(s.sold_qty) as y,
                p.name as product_name,
                c.name as category_name,
                AVG(s.sold_price) as avg_price,
                COUNT(DISTINCT s.cashier_sid) as unique_customers,
                EXTRACT(DOW FROM s.sold_at) as day_of_week,
                EXTRACT(MONTH FROM s.sold_at) as month,
                CASE 
                    WHEN EXTRACT(DOW FROM s.sold_at) IN (0, 6) THEN 1 
                    ELSE 0 
                END as is_weekend
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
                p.sid = :product_sid
                AND s.sold_at >= :min_date
            GROUP BY 
                DATE(s.sold_at), p.name, c.name, EXTRACT(DOW FROM s.sold_at), EXTRACT(MONTH FROM s.sold_at)
            ORDER BY 
                ds
        """)

        min_date = datetime.now() - timedelta(days=days_back)

        try:
            result = await self.db.execute(
                query,
                {"product_sid": product_sid, "min_date": min_date}
            )
            rows = result.fetchall()

            if not rows:
                logger.warning(f"No sales data found for product {product_sid}")
                prod_query = text("""
                    SELECT p.name as product_name, c.name as category_name
                    FROM product p
                    JOIN category c ON p.category_sid = c.sid
                    WHERE p.sid = :product_sid
                """)
                prod_result = await self.db.execute(prod_query, {"product_sid": product_sid})
                prod_row = prod_result.fetchone()

                if prod_row:
                    product_name = prod_row.product_name
                    category_name = prod_row.category_name
                else:
                    product_name = "Unknown"
                    category_name = "Unknown"

                df = pd.DataFrame(columns=["ds", "y", "product_name", "category_name", "avg_price",
                                           "unique_customers", "day_of_week", "month", "is_weekend"])

                date_range = pd.date_range(
                    start=min_date,
                    end=datetime.now().date(),
                    freq='D'
                )

                empty_data = []
                for single_date in date_range:
                    empty_data.append({
                        "ds": single_date,
                        "y": 0.0,
                        "product_name": product_name,
                        "category_name": category_name,
                        "avg_price": 0.0,
                        "unique_customers": 0,
                        "day_of_week": single_date.dayofweek,
                        "month": single_date.month,
                        "is_weekend": 1 if single_date.dayofweek in [5, 6] else 0
                    })

                if empty_data:
                    df = pd.DataFrame(empty_data)

                return df

            df = pd.DataFrame([
                {
                    "ds": row.ds,
                    "y": float(row.y),
                    "product_name": row.product_name,
                    "category_name": row.category_name,
                    "avg_price": float(row.avg_price) if row.avg_price else 0,
                    "unique_customers": int(row.unique_customers),
                    "day_of_week": int(row.day_of_week),
                    "month": int(row.month),
                    "is_weekend": int(row.is_weekend)
                } for row in rows
            ])

            df['ds'] = pd.to_datetime(df['ds'])

            product_name = df['product_name'].iloc[0]
            category_name = df['category_name'].iloc[0]

            date_range = pd.date_range(
                start=df['ds'].min(),
                end=datetime.now().date(),
                freq='D'
            )

            date_df = pd.DataFrame({"ds": date_range})

            merged_df = pd.merge(date_df, df, on='ds', how='left')

            merged_df['y'] = merged_df['y'].fillna(0)
            merged_df['product_name'] = merged_df['product_name'].fillna(product_name)
            merged_df['category_name'] = merged_df['category_name'].fillna(category_name)
            merged_df['avg_price'] = merged_df['avg_price'].fillna(merged_df['avg_price'].mean())
            merged_df['unique_customers'] = merged_df['unique_customers'].fillna(0)
            merged_df['day_of_week'] = merged_df['ds'].dt.dayofweek
            merged_df['month'] = merged_df['ds'].dt.month
            merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)

            merged_df['lag_1'] = merged_df['y'].shift(1).fillna(0)
            merged_df['lag_7'] = merged_df['y'].shift(7).fillna(0)
            merged_df['rolling_mean_7'] = merged_df['y'].rolling(window=7, min_periods=1).mean()
            merged_df['rolling_std_7'] = merged_df['y'].rolling(window=7, min_periods=1).std().fillna(0)

            return merged_df

        except Exception as e:
            logger.error(f"Error getting sales data: {str(e)}")
            raise

    async def get_category_seasonality(self, category_sid: str) -> Dict[str, Any]:
        query = text("""
            SELECT 
                EXTRACT(DOW FROM s.sold_at) as day_of_week,
                EXTRACT(MONTH FROM s.sold_at) as month,
                AVG(s.sold_qty) as avg_quantity,
                STDDEV(s.sold_qty) as std_quantity
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                p.category_sid = :category_sid
                AND s.sold_at >= :min_date
            GROUP BY 
                day_of_week, month
            ORDER BY 
                month, day_of_week
        """)

        min_date = datetime.now() - timedelta(days=365)

        try:
            result = await self.db.execute(query, {"category_sid": category_sid, "min_date": min_date})
            rows = result.fetchall()

            if not rows:
                return {
                    "day_of_week": {},
                    "monthly": {},
                    "has_seasonality": False,
                    "seasonality_strength": 0
                }

            df = pd.DataFrame([
                {
                    "day_of_week": float(row.day_of_week),
                    "month": float(row.month),
                    "avg_quantity": float(row.avg_quantity),
                    "std_quantity": float(row.std_quantity) if row.std_quantity else 0
                } for row in rows
            ])

            daily_seasonality = df.groupby('day_of_week')['avg_quantity'].mean().to_dict()
            monthly_seasonality = df.groupby('month')['avg_quantity'].mean().to_dict()

            daily_values = list(daily_seasonality.values())
            monthly_values = list(monthly_seasonality.values())

            if daily_values and monthly_values:
                daily_variation = np.std(daily_values) / np.mean(daily_values)
                monthly_variation = np.std(monthly_values) / np.mean(monthly_values)
                seasonality_strength = max(daily_variation, monthly_variation)
            else:
                daily_variation = 0
                monthly_variation = 0
                seasonality_strength = 0

            has_seasonality = bool(daily_variation > 0.2 or monthly_variation > 0.3)

            return {
                "day_of_week": {int(k): float(v) for k, v in daily_seasonality.items()},
                "monthly": {int(k): float(v) for k, v in monthly_seasonality.items()},
                "has_seasonality": has_seasonality,
                "seasonality_strength": float(seasonality_strength)
            }

        except Exception as e:
            logger.error(f"Error getting category seasonality: {str(e)}")
            return {
                "day_of_week": {},
                "monthly": {},
                "has_seasonality": False,
                "seasonality_strength": 0
            }

    async def train_prophet_model(self,
                                  product_sid: str,
                                  seasonality_info: Dict[str, Any] = None) -> Optional[Prophet]:
        df = await self.get_sales_data(product_sid)

        if df.empty or len(df) < 7:
            logger.warning(f"Not enough data to train model for product {product_sid}")
            return None

        try:
            if not seasonality_info:
                prod_query = text("""
                    SELECT p.category_sid
                    FROM product p
                    WHERE p.sid = :product_sid
                """)
                prod_result = await self.db.execute(prod_query, {"product_sid": product_sid})
                category_sid = prod_result.scalar_one_or_none()

                if category_sid:
                    seasonality_info = await self.get_category_seasonality(category_sid)

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )

            has_seasonality = bool(seasonality_info.get('has_seasonality', False)) if seasonality_info else False
            if has_seasonality:
                model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )

            model.add_regressor('is_weekend')
            model.add_regressor('lag_7')
            model.add_regressor('rolling_mean_7')

            train_df = df[['ds', 'y', 'is_weekend', 'lag_7', 'rolling_mean_7']].copy()

            model.fit(train_df)

            return model

        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return None

    async def train_ensemble_models(self, product_sid: str) -> Dict[str, Any]:
        df = await self.get_sales_data(product_sid, days_back=365)

        if df.empty or len(df) < 30:
            return None

        features = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7',
                    'rolling_mean_7', 'rolling_std_7']

        X = df[features].fillna(0)
        y = df['y']

        if len(X) < 30:
            return None

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        return {
            'random_forest': rf_model,
            'linear_regression': lr_model,
            'features': features,
            'last_data': df.iloc[-1].to_dict()
        }

    async def generate_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame,
            periods_ahead: int = 1
    ) -> List[Dict[str, Any]]:
        try:
            prod_query = text("""
                SELECT p.name, c.name as category_name
                FROM product p
                JOIN category c ON p.category_sid = c.sid
                WHERE p.sid = :product_sid
            """)
            prod_result = await self.db.execute(prod_query, {"product_sid": product_sid})
            prod_row = prod_result.fetchone()

            if not prod_row:
                logger.warning(f"Product {product_sid} not found")
                return []

            product_name = prod_row.name
            category_name = prod_row.category_name

            prophet_model = await self.train_prophet_model(product_sid)
            ensemble_models = await self.train_ensemble_models(product_sid)

            if not prophet_model and not ensemble_models:
                return await self._generate_statistical_forecast(
                    product_sid, timeframe, periods_ahead, product_name, category_name
                )

            if timeframe == TimeFrame.DAY:
                days_per_period = 1
                freq = 'D'
            elif timeframe == TimeFrame.WEEK:
                days_per_period = 7
                freq = 'W'
            else:
                days_per_period = 30
                freq = 'M'

            results = []
            today = datetime.now().date()

            if prophet_model:
                future_periods = periods_ahead * days_per_period
                future = prophet_model.make_future_dataframe(periods=future_periods, freq='D')

                df = await self.get_sales_data(product_sid)
                last_data = df.iloc[-1] if not df.empty else None

                if last_data is not None:
                    future['is_weekend'] = future['ds'].dt.dayofweek.apply(lambda x: 1 if x in [5, 6] else 0)
                    future['lag_7'] = last_data['y']
                    future['rolling_mean_7'] = last_data['rolling_mean_7']
                else:
                    future['is_weekend'] = 0
                    future['lag_7'] = 0
                    future['rolling_mean_7'] = 0

                prophet_forecast = prophet_model.predict(future)

            for i in range(periods_ahead):
                period_start = today + timedelta(days=i * days_per_period)
                period_end = period_start + timedelta(days=days_per_period - 1)

                prophet_qty = 0
                prophet_lower = 0
                prophet_upper = 0

                if prophet_model:
                    period_forecast = prophet_forecast[
                        (prophet_forecast['ds'].dt.date >= period_start) &
                        (prophet_forecast['ds'].dt.date <= period_end)
                        ]

                    if not period_forecast.empty:
                        prophet_qty = period_forecast['yhat'].sum()
                        prophet_lower = period_forecast['yhat_lower'].sum()
                        prophet_upper = period_forecast['yhat_upper'].sum()

                ensemble_qty = 0
                if ensemble_models:
                    future_features = []
                    for day in range(days_per_period):
                        future_date = period_start + timedelta(days=day)
                        features = {
                            'day_of_week': future_date.weekday(),
                            'month': future_date.month,
                            'is_weekend': 1 if future_date.weekday() in [5, 6] else 0,
                            'lag_1': ensemble_models['last_data']['y'],
                            'lag_7': ensemble_models['last_data']['y'],
                            'rolling_mean_7': ensemble_models['last_data']['rolling_mean_7'],
                            'rolling_std_7': ensemble_models['last_data']['rolling_std_7']
                        }
                        future_features.append(features)

                    X_future = pd.DataFrame(future_features)
                    rf_pred = ensemble_models['random_forest'].predict(X_future).sum()
                    lr_pred = ensemble_models['linear_regression'].predict(X_future).sum()
                    ensemble_qty = (rf_pred + lr_pred) / 2

                if prophet_model and ensemble_models:
                    forecast_qty = (prophet_qty * 0.7 + ensemble_qty * 0.3)
                    forecast_qty_lower = prophet_lower * 0.8
                    forecast_qty_upper = prophet_upper * 1.2
                elif prophet_model:
                    forecast_qty = prophet_qty
                    forecast_qty_lower = prophet_lower
                    forecast_qty_upper = prophet_upper
                else:
                    forecast_qty = ensemble_qty
                    forecast_qty_lower = ensemble_qty * 0.7
                    forecast_qty_upper = ensemble_qty * 1.3

                results.append({
                    "product_sid": product_sid,
                    "product_name": product_name,
                    "category_name": category_name,
                    "timeframe": timeframe,
                    "period_start": period_start,
                    "period_end": period_end,
                    "forecast_qty": max(0, round(float(forecast_qty), 2)),
                    "forecast_qty_lower": max(0, round(float(forecast_qty_lower), 2)),
                    "forecast_qty_upper": max(0, round(float(forecast_qty_upper), 2)),
                    "generated_at": datetime.now(),
                    "model_version": f"{self.model_version}"
                })

            return results

        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return await self._generate_statistical_forecast(
                product_sid, timeframe, periods_ahead
            )

    async def _generate_statistical_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame,
            periods_ahead: int = 1,
            product_name: str = None,
            category_name: str = None
    ) -> List[Dict[str, Any]]:
        try:
            df = await self.get_sales_data(product_sid)

            if df.empty:
                if not product_name or not category_name:
                    prod_query = text("""
                        SELECT p.name, c.name as category_name
                        FROM product p
                        JOIN category c ON p.category_sid = c.sid
                        WHERE p.sid = :product_sid
                    """)
                    prod_result = await self.db.execute(prod_query, {"product_sid": product_sid})
                    prod_row = prod_result.fetchone()

                    if prod_row:
                        product_name = prod_row.name
                        category_name = prod_row.category_name
                    else:
                        product_name = "Unknown"
                        category_name = "Unknown"

                return self._generate_placeholder_forecast(
                    product_sid, timeframe, periods_ahead, product_name, category_name
                )

            if not product_name:
                product_name = df['product_name'].iloc[0]
            if not category_name:
                category_name = df['category_name'].iloc[0]

            if len(df) >= 14:
                y_values = df['y'].values

                if len(y_values) >= 30:
                    window_size = 14
                else:
                    window_size = max(3, len(y_values) // 3)

                recent_avg = np.mean(y_values[-window_size:])

                if len(y_values) >= window_size * 2:
                    prev_avg = np.mean(y_values[-(window_size * 2):-window_size])
                    trend = (recent_avg - prev_avg) / window_size
                else:
                    trend = 0

                today = datetime.now().date()
                days_per_period = 1 if timeframe == TimeFrame.DAY else 7 if timeframe == TimeFrame.WEEK else 30

                results = []
                for i in range(periods_ahead):
                    period_start = today + timedelta(days=i * days_per_period)
                    period_end = period_start + timedelta(days=days_per_period - 1)

                    forecast_qty = recent_avg + trend * (i + 1) * days_per_period
                    forecast_qty = max(0, forecast_qty * days_per_period)

                    std_dev = np.std(y_values[-min(30, len(y_values)):])
                    forecast_qty_lower = max(0, forecast_qty - 1.96 * std_dev * days_per_period)
                    forecast_qty_upper = forecast_qty + 1.96 * std_dev * days_per_period

                    results.append({
                        "product_sid": product_sid,
                        "product_name": product_name,
                        "category_name": category_name,
                        "timeframe": timeframe,
                        "period_start": period_start,
                        "period_end": period_end,
                        "forecast_qty": round(float(forecast_qty), 2),
                        "forecast_qty_lower": round(float(forecast_qty_lower), 2),
                        "forecast_qty_upper": round(float(forecast_qty_upper), 2),
                        "generated_at": datetime.now(),
                        "model_version": f"{self.model_version} (statistical)"
                    })

                return results
            else:
                return self._generate_placeholder_forecast(
                    product_sid, timeframe, periods_ahead, product_name, category_name
                )

        except Exception as e:
            logger.error(f"Error generating statistical forecast: {str(e)}")
            return self._generate_placeholder_forecast(
                product_sid, timeframe, periods_ahead
            )

    def _generate_placeholder_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame,
            periods_ahead: int = 1,
            product_name: str = "Unknown",
            category_name: str = "Unknown"
    ) -> List[Dict[str, Any]]:
        today = datetime.now().date()

        if timeframe == TimeFrame.DAY:
            days_per_period = 1
        elif timeframe == TimeFrame.WEEK:
            days_per_period = 7
        else:
            days_per_period = 30

        results = []
        for i in range(periods_ahead):
            period_start = today + timedelta(days=i * days_per_period)
            period_end = period_start + timedelta(days=days_per_period - 1)

            base_qty = float(np.random.randint(50, 150))

            forecast_qty_lower = base_qty * 0.7
            forecast_qty_upper = base_qty * 1.3

            results.append({
                "product_sid": product_sid,
                "product_name": product_name,
                "category_name": category_name,
                "timeframe": timeframe,
                "period_start": period_start,
                "period_end": period_end,
                "forecast_qty": base_qty,
                "forecast_qty_lower": forecast_qty_lower,
                "forecast_qty_upper": forecast_qty_upper,
                "generated_at": datetime.now(),
                "model_version": f"{self.model_version} (placeholder)"
            })

        return results

    async def get_sales_trends(self,
                               product_sid: Optional[str] = None,
                               category_sid: Optional[str] = None,
                               days_back: int = 90) -> Dict[str, Any]:
        filters = []
        params = {"min_date": datetime.now() - timedelta(days=days_back)}

        if product_sid:
            filters.append("p.sid = :product_sid")
            params["product_sid"] = product_sid

        if category_sid:
            filters.append("p.category_sid = :category_sid")
            params["category_sid"] = category_sid

        where_clause = "AND " + " AND ".join(filters) if filters else ""

        query = text(f"""
            SELECT 
                DATE(s.sold_at) as date,
                SUM(s.sold_qty) as quantity,
                SUM(s.sold_qty * s.sold_price) as revenue,
                COUNT(DISTINCT p.sid) as product_count,
                COUNT(DISTINCT s.cashier_sid) as unique_customers
            FROM 
                sale s
            JOIN 
                storeitem si ON s.store_item_sid = si.sid
            JOIN 
                warehouseitem wi ON si.warehouse_item_sid = wi.sid
            JOIN 
                product p ON wi.product_sid = p.sid
            WHERE 
                s.sold_at >= :min_date
                {where_clause}
            GROUP BY 
                date
            ORDER BY 
                date
        """)

        try:
            result = await self.db.execute(query, params)
            rows = result.fetchall()

            if not rows:
                return {
                    "dates": [],
                    "quantities": [],
                    "revenues": [],
                    "growth": {
                        "quantity": 0,
                        "revenue": 0
                    },
                    "trend": "stable",
                    "insights": []
                }

            df = pd.DataFrame([
                {
                    "date": row.date,
                    "quantity": float(row.quantity),
                    "revenue": float(row.revenue),
                    "product_count": int(row.product_count),
                    "unique_customers": int(row.unique_customers)
                } for row in rows
            ])

            dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
            quantities = df["quantity"].tolist()
            revenues = df["revenue"].tolist()

            if len(df) >= 2:
                half_point = len(df) // 2
                first_half_qty = df["quantity"].iloc[:half_point].mean()
                second_half_qty = df["quantity"].iloc[half_point:].mean()

                first_half_rev = df["revenue"].iloc[:half_point].mean()
                second_half_rev = df["revenue"].iloc[half_point:].mean()

                if first_half_qty > 0:
                    qty_growth = (second_half_qty - first_half_qty) / first_half_qty * 100
                else:
                    qty_growth = 0

                if first_half_rev > 0:
                    rev_growth = (second_half_rev - first_half_rev) / first_half_rev * 100
                else:
                    rev_growth = 0

                if qty_growth > 10 and rev_growth > 10:
                    trend = "strong_growth"
                elif qty_growth > 5 and rev_growth > 5:
                    trend = "growth"
                elif qty_growth < -10 and rev_growth < -10:
                    trend = "strong_decline"
                elif qty_growth < -5 and rev_growth < -5:
                    trend = "decline"
                else:
                    trend = "stable"
            else:
                qty_growth = 0
                rev_growth = 0
                trend = "insufficient_data"

            insights = []

            if trend == "strong_growth":
                insights.append({
                    "type": "positive",
                    "message": f"Продажи демонстрируют сильный рост: +{qty_growth:.1f}% по количеству"
                })
            elif trend == "decline":
                insights.append({
                    "type": "negative",
                    "message": f"Наблюдается снижение продаж: {qty_growth:.1f}% по количеству"
                })

            avg_daily_sales = df["quantity"].mean()
            if avg_daily_sales < 5:
                insights.append({
                    "type": "warning",
                    "message": "Низкий уровень среднедневных продаж"
                })

            qty_growth_val = float(qty_growth) if 'qty_growth' in locals() else 0
            rev_growth_val = float(rev_growth) if 'rev_growth' in locals() else 0

            return {
                "dates": dates,
                "quantities": quantities,
                "revenues": revenues,
                "growth": {
                    "quantity": round(qty_growth_val, 2),
                    "revenue": round(rev_growth_val, 2)
                },
                "trend": trend,
                "insights": insights
            }

        except Exception as e:
            logger.error(f"Error getting sales trends: {str(e)}")
            return {
                "dates": [],
                "quantities": [],
                "revenues": [],
                "growth": {
                    "quantity": 0,
                    "revenue": 0
                },
                "trend": "error",
                "insights": []
            }

    async def save_forecast(self, forecasts: List[Dict[str, Any]]) -> List[Prediction]:
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
            prediction_objects.append({
                "db_obj": prediction,
                "forecast_qty_lower": forecast.get("forecast_qty_lower"),
                "forecast_qty_upper": forecast.get("forecast_qty_upper")
            })

        await self.db.commit()

        result_predictions = []
        for pred_info in prediction_objects:
            pred = pred_info["db_obj"]
            await self.db.refresh(pred)

            prediction_dict = {
                "sid": pred.sid,
                "product_sid": pred.product_sid,
                "timeframe": pred.timeframe,
                "period_start": pred.period_start,
                "period_end": pred.period_end,
                "forecast_qty": pred.forecast_qty,
                "generated_at": pred.generated_at,
                "model_version": pred.model_version,
                "product": None,
                "forecast_qty_lower": pred_info["forecast_qty_lower"],
                "forecast_qty_upper": pred_info["forecast_qty_upper"]
            }

            result_predictions.append(prediction_dict)

        return result_predictions

    async def get_product_analytics(self, product_sid: str) -> Dict[str, Any]:
        try:
            prod_query = text("""
                SELECT 
                    p.name as product_name,
                    p.barcode,
                    p.default_price,
                    c.name as category_name,
                    c.sid as category_sid
                FROM 
                    product p
                JOIN 
                    category c ON p.category_sid = c.sid
                WHERE 
                    p.sid = :product_sid
            """)

            prod_result = await self.db.execute(prod_query, {"product_sid": product_sid})
            prod_data = prod_result.fetchone()

            if not prod_data:
                return {"error": "Product not found"}

            sales_query = text("""
                SELECT 
                    DATE_TRUNC('month', s.sold_at) as month,
                    SUM(s.sold_qty) as quantity,
                    SUM(s.sold_qty * s.sold_price) as revenue,
                    AVG(s.sold_price) as avg_price,
                    COUNT(s.id) as transaction_count,
                    COUNT(DISTINCT s.cashier_sid) as unique_customers
                FROM 
                    sale s
                JOIN 
                    storeitem si ON s.store_item_sid = si.sid
                JOIN 
                    warehouseitem wi ON si.warehouse_item_sid = wi.sid
                WHERE 
                    wi.product_sid = :product_sid
                    AND s.sold_at >= :min_date
                GROUP BY 
                    month
                ORDER BY 
                    month
            """)

            min_date = datetime.now() - timedelta(days=365)
            sales_result = await self.db.execute(sales_query,
                                                 {"product_sid": product_sid, "min_date": min_date})
            sales_rows = sales_result.fetchall()

            inventory_query = text("""
                SELECT 
                    SUM(wi.quantity) as warehouse_quantity,
                    COUNT(wi.id) as batch_count,
                    MIN(wi.expire_date) as nearest_expiry,
                    AVG(CASE WHEN wi.expire_date IS NOT NULL 
                        THEN wi.expire_date - CURRENT_DATE 
                        ELSE NULL END) as avg_days_to_expiry
                FROM 
                    warehouseitem wi
                WHERE 
                    wi.product_sid = :product_sid
                    AND wi.status = 'IN_STOCK'
            """)

            inventory_result = await self.db.execute(inventory_query, {"product_sid": product_sid})
            inventory_data = inventory_result.fetchone()

            store_query = text("""
                SELECT 
                    SUM(si.quantity) as store_quantity,
                    AVG(si.price) as current_price,
                    COUNT(d.id) as active_discounts,
                    MIN(si.price) as min_price,
                    MAX(si.price) as max_price
                FROM 
                    storeitem si
                JOIN 
                    warehouseitem wi ON si.warehouse_item_sid = wi.sid
                LEFT JOIN 
                    discount d ON si.sid = d.store_item_sid 
                    AND d.starts_at <= NOW() 
                    AND d.ends_at >= NOW()
                WHERE 
                    wi.product_sid = :product_sid
                    AND si.status = 'ACTIVE'
                GROUP BY 
                    wi.product_sid
            """)

            store_result = await self.db.execute(store_query, {"product_sid": product_sid})
            store_data = store_result.fetchone()

            trends = await self.get_sales_trends(product_sid=product_sid)

            forecasts = await self.generate_forecast(
                product_sid=product_sid,
                timeframe=TimeFrame.MONTH,
                periods_ahead=3
            )

            if sales_rows:
                sales_data = []
                total_quantity = 0
                total_revenue = 0
                total_customers = 0

                for row in sales_rows:
                    month_str = row.month.strftime("%Y-%m")
                    sales_data.append({
                        "month": month_str,
                        "quantity": float(row.quantity),
                        "revenue": float(row.revenue),
                        "avg_price": float(row.avg_price),
                        "transaction_count": int(row.transaction_count),
                        "unique_customers": int(row.unique_customers)
                    })
                    total_quantity += row.quantity
                    total_revenue += row.revenue
                    total_customers += row.unique_customers

                avg_monthly_sales = total_quantity / len(sales_rows)
                avg_monthly_revenue = total_revenue / len(sales_rows)
                avg_customers_per_month = total_customers / len(sales_rows)

                if store_data and store_data.store_quantity:
                    turnover_rate = avg_monthly_sales / float(store_data.store_quantity)
                    days_of_supply = float(store_data.store_quantity) / (
                            avg_monthly_sales / 30) if avg_monthly_sales > 0 else 0
                else:
                    turnover_rate = 0
                    days_of_supply = 0

                if inventory_data and inventory_data.warehouse_quantity:
                    warehouse_days_supply = float(inventory_data.warehouse_quantity) / (
                            avg_monthly_sales / 30) if avg_monthly_sales > 0 else 0
                else:
                    warehouse_days_supply = 0

                price_volatility = 0
                if store_data and store_data.min_price and store_data.max_price:
                    price_volatility = (store_data.max_price - store_data.min_price) / store_data.min_price * 100
            else:
                sales_data = []
                avg_monthly_sales = 0
                avg_monthly_revenue = 0
                avg_customers_per_month = 0
                turnover_rate = 0
                days_of_supply = 0
                warehouse_days_supply = 0
                price_volatility = 0

            category_query = text("""
                SELECT 
                    p.sid as product_sid,
                    p.name as product_name,
                    SUM(s.sold_qty) as quantity,
                    SUM(s.sold_qty * s.sold_price) as revenue,
                    AVG(s.sold_price) as avg_price
                FROM 
                    sale s
                JOIN 
                    storeitem si ON s.store_item_sid = si.sid
                JOIN 
                    warehouseitem wi ON si.warehouse_item_sid = wi.sid
                JOIN 
                    product p ON wi.product_sid = p.sid
                WHERE 
                    p.category_sid = :category_sid
                    AND s.sold_at >= :min_date
                GROUP BY 
                    p.sid, p.name
                ORDER BY 
                    revenue DESC
                LIMIT 5
            """)

            category_result = await self.db.execute(
                category_query,
                {"category_sid": prod_data.category_sid, "min_date": min_date}
            )
            category_rows = category_result.fetchall()

            category_comparison = []
            for row in category_rows:
                is_current = bool(row.product_sid == product_sid)

                category_comparison.append({
                    "product_sid": row.product_sid,
                    "product_name": row.product_name,
                    "quantity": float(row.quantity),
                    "revenue": float(row.revenue),
                    "avg_price": float(row.avg_price),
                    "is_current": is_current
                })

            default_price = float(prod_data.default_price) if prod_data.default_price else 0
            warehouse_quantity = float(
                inventory_data.warehouse_quantity) if inventory_data and inventory_data.warehouse_quantity else 0
            store_quantity = float(store_data.store_quantity) if store_data and store_data.store_quantity else 0
            current_price = float(store_data.current_price) if store_data and store_data.current_price else 0
            active_discounts = int(store_data.active_discounts) if store_data and store_data.active_discounts else 0

            insights = []

            if turnover_rate < 0.5 and turnover_rate > 0:
                insights.append({
                    "type": "warning",
                    "title": "Низкая оборачиваемость товара",
                    "description": f"Текущая оборачиваемость {turnover_rate:.1%} ниже оптимальной",
                    "recommendation": "Рассмотрите возможность снижения цены или проведения акции"
                })

            if days_of_supply > 60:
                insights.append({
                    "type": "warning",
                    "title": "Избыточные запасы в магазине",
                    "description": f"Запасов в магазине хватит на {days_of_supply:.0f} дней",
                    "recommendation": "Приостановите перемещение товаров со склада"
                })
            elif days_of_supply < 7 and days_of_supply > 0:
                insights.append({
                    "type": "critical",
                    "title": "Низкий уровень запасов",
                    "description": f"Запасов осталось на {days_of_supply:.0f} дней",
                    "recommendation": "Срочно переместите товар со склада в магазин"
                })

            if price_volatility > 20:
                insights.append({
                    "type": "info",
                    "title": "Высокая волатильность цен",
                    "description": f"Разброс цен составляет {price_volatility:.1f}%",
                    "recommendation": "Стабилизируйте ценовую политику для данного товара"
                })

            if inventory_data and inventory_data.avg_days_to_expiry:
                avg_days = float(inventory_data.avg_days_to_expiry)
                if avg_days < 30:
                    insights.append({
                        "type": "warning",
                        "title": "Товары с коротким сроком годности",
                        "description": f"Средний срок до истечения: {avg_days:.0f} дней",
                        "recommendation": "Ускорьте реализацию или примените скидки"
                    })

            return {
                "product_info": {
                    "name": prod_data.product_name,
                    "barcode": prod_data.barcode,
                    "default_price": default_price,
                    "category": prod_data.category_name
                },
                "inventory": {
                    "warehouse_quantity": warehouse_quantity,
                    "store_quantity": store_quantity,
                    "current_price": current_price,
                    "active_discounts": active_discounts,
                    "nearest_expiry": inventory_data.nearest_expiry.strftime(
                        "%Y-%m-%d") if inventory_data and inventory_data.nearest_expiry else None,
                    "avg_days_to_expiry": float(
                        inventory_data.avg_days_to_expiry) if inventory_data and inventory_data.avg_days_to_expiry else None
                },
                "sales_data": sales_data,
                "trends": trends,
                "forecasts": [
                    {
                        "period_start": f.get("period_start").strftime("%Y-%m-%d"),
                        "period_end": f.get("period_end").strftime("%Y-%m-%d"),
                        "forecast_qty": float(f.get("forecast_qty")),
                        "forecast_qty_lower": float(f.get("forecast_qty_lower", f.get("forecast_qty") * 0.7)),
                        "forecast_qty_upper": float(f.get("forecast_qty_upper", f.get("forecast_qty") * 1.3))
                    }
                    for f in forecasts
                ],
                "kpis": {
                    "avg_monthly_sales": float(avg_monthly_sales),
                    "avg_monthly_revenue": float(avg_monthly_revenue),
                    "avg_customers_per_month": float(avg_customers_per_month),
                    "turnover_rate": float(turnover_rate),
                    "days_of_supply": float(days_of_supply),
                    "warehouse_days_supply": float(warehouse_days_supply),
                    "price_volatility": float(price_volatility)
                },
                "category_comparison": category_comparison,
                "insights": insights
            }

        except Exception as e:
            logger.error(f"Error getting product analytics: {str(e)}")
            return {"error": f"Failed to get analytics: {str(e)}"}