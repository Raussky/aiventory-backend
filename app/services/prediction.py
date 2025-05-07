# app/services/prediction.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from app.models.inventory import Prediction, Product, TimeFrame, Category
from app.models.base import Base

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_version = "prophet_v1.0.0"

    async def get_sales_data(self, product_sid: str, days_back: int = 180) -> pd.DataFrame:
        """Retrieves historical sales data for a product"""
        query = text("""
            SELECT 
                DATE(s.sold_at) as ds,
                SUM(s.sold_qty) as y,
                p.name as product_name,
                c.name as category_name
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
                DATE(s.sold_at), p.name, c.name
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
                return pd.DataFrame(columns=["ds", "y", "product_name", "category_name"])

            df = pd.DataFrame(rows, columns=["ds", "y", "product_name", "category_name"])
            df['ds'] = pd.to_datetime(df['ds'])

            # Fill missing dates with zeros
            date_range = pd.date_range(
                start=df['ds'].min() if not df.empty else min_date,
                end=datetime.now().date(),
                freq='D'
            )

            all_dates = pd.DataFrame({'ds': date_range})
            if not df.empty:
                product_name = df['product_name'].iloc[0]
                category_name = df['category_name'].iloc[0]
            else:
                # Get product and category names from the database
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

            all_dates['product_name'] = product_name
            all_dates['category_name'] = category_name

            merged_df = pd.merge(all_dates, df, on='ds', how='left')
            merged_df['y'] = merged_df['y_y'].fillna(0)
            merged_df.drop(['product_name_y', 'category_name_y', 'y_y'], axis=1, errors='ignore', inplace=True)
            merged_df.rename(columns={'product_name_x': 'product_name', 'category_name_x': 'category_name'},
                             inplace=True)

            return merged_df

        except Exception as e:
            logger.error(f"Error getting sales data: {str(e)}")
            return pd.DataFrame(columns=["ds", "y", "product_name", "category_name"])

    async def get_category_seasonality(self, category_sid: str) -> Dict[str, Any]:
        """Get seasonal patterns for a category"""
        query = text("""
            SELECT 
                EXTRACT(DOW FROM s.sold_at) as day_of_week,
                EXTRACT(MONTH FROM s.sold_at) as month,
                AVG(s.sold_qty) as avg_quantity
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
                    "has_seasonality": False
                }

            # Convert to dataframe
            df = pd.DataFrame(rows, columns=["day_of_week", "month", "avg_quantity"])

            # Calculate daily seasonality
            daily_seasonality = df.groupby('day_of_week')['avg_quantity'].mean().to_dict()

            # Calculate monthly seasonality
            monthly_seasonality = df.groupby('month')['avg_quantity'].mean().to_dict()

            # Determine if product has strong seasonality
            daily_variation = np.std(list(daily_seasonality.values())) / np.mean(list(daily_seasonality.values()))
            monthly_variation = np.std(list(monthly_seasonality.values())) / np.mean(list(monthly_seasonality.values()))

            return {
                "day_of_week": {int(k): float(v) for k, v in daily_seasonality.items()},
                "monthly": {int(k): float(v) for k, v in monthly_seasonality.items()},
                "has_seasonality": daily_variation > 0.2 or monthly_variation > 0.3
            }

        except Exception as e:
            logger.error(f"Error getting category seasonality: {str(e)}")
            return {
                "day_of_week": {},
                "monthly": {},
                "has_seasonality": False
            }

    async def train_prophet_model(self,
                                  product_sid: str,
                                  seasonality_info: Dict[str, Any] = None) -> Optional[Prophet]:
        """Train a Prophet model for time series forecasting"""
        df = await self.get_sales_data(product_sid)

        if df.empty or len(df) < 7:
            logger.warning(f"Not enough data to train model for product {product_sid}")
            return None

        try:
            # Get product category for seasonality information
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

            # Configure Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05
            )

            # Add custom seasonality if needed
            has_seasonality = seasonality_info.get('has_seasonality', False) if seasonality_info else False
            if has_seasonality:
                # Add monthly seasonality
                model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )

            # Training data needs 'ds' and 'y' columns
            train_df = df[['ds', 'y']].copy()

            # Train model
            model.fit(train_df)

            return model

        except Exception as e:
            logger.error(f"Error training Prophet model: {str(e)}")
            return None

    async def generate_forecast(
            self,
            product_sid: str,
            timeframe: TimeFrame,
            periods_ahead: int = 1
    ) -> List[Dict[str, Any]]:
        """Generates forecast for a product using Prophet"""
        try:
            # Get product details
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

            # Train model
            model = await self.train_prophet_model(product_sid)

            # If model training failed, fallback to simple statistical forecast
            if model is None:
                return await self._generate_statistical_forecast(
                    product_sid, timeframe, periods_ahead, product_name, category_name
                )

            # Determine period length in days
            if timeframe == TimeFrame.DAY:
                days_per_period = 1
                freq = 'D'
            elif timeframe == TimeFrame.WEEK:
                days_per_period = 7
                freq = 'W'
            else:  # MONTH
                days_per_period = 30
                freq = 'M'

            # Create future dataframe
            future_periods = periods_ahead
            if timeframe == TimeFrame.DAY:
                future_periods *= 1
            elif timeframe == TimeFrame.WEEK:
                future_periods *= 7
            else:  # MONTH
                future_periods *= 30

            future = model.make_future_dataframe(periods=future_periods, freq='D')

            # Generate forecast
            forecast = model.predict(future)

            # Extract results for the requested periods
            results = []
            today = datetime.now().date()

            for i in range(periods_ahead):
                period_start = today + timedelta(days=i * days_per_period)
                period_end = period_start + timedelta(days=days_per_period - 1)

                # Filter forecast for this period
                period_forecast = forecast[
                    (forecast['ds'].dt.date >= period_start) &
                    (forecast['ds'].dt.date <= period_end)
                    ]

                # Calculate total forecast for the period
                if not period_forecast.empty:
                    forecast_qty = period_forecast['yhat'].sum()
                    forecast_qty_lower = period_forecast['yhat_lower'].sum()
                    forecast_qty_upper = period_forecast['yhat_upper'].sum()
                else:
                    # Fallback if no forecast available
                    forecast_qty = 0
                    forecast_qty_lower = 0
                    forecast_qty_upper = 0

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
                    "model_version": f"{self.model_version} (prophet)"
                })

            return results

        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {str(e)}")
            # Fallback to statistical forecast
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
        """Fallback method using simple statistical forecasting"""
        try:
            # Get sales data
            df = await self.get_sales_data(product_sid)

            if df.empty:
                # No data available, use placeholder values
                if not product_name or not category_name:
                    # Get product details if not provided
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

                # Generate placeholder forecasts
                return self._generate_placeholder_forecast(
                    product_sid, timeframe, periods_ahead, product_name, category_name
                )

            # Use product name and category from data if not provided
            if not product_name:
                product_name = df['product_name'].iloc[0]
            if not category_name:
                category_name = df['category_name'].iloc[0]

            # Calculate simple moving average or exponential smoothing
            if len(df) >= 14:
                # Enough data for moving average
                y_values = df['y'].values

                # Use different window sizes based on data availability
                if len(y_values) >= 30:
                    window_size = 14
                else:
                    window_size = max(3, len(y_values) // 3)

                # Calculate moving average for recent values
                recent_avg = np.mean(y_values[-window_size:])

                # Calculate trend
                if len(y_values) >= window_size * 2:
                    prev_avg = np.mean(y_values[-(window_size * 2):-window_size])
                    trend = (recent_avg - prev_avg) / window_size
                else:
                    trend = 0

                # Generate forecasts
                today = datetime.now().date()
                days_per_period = 1 if timeframe == TimeFrame.DAY else 7 if timeframe == TimeFrame.WEEK else 30

                results = []
                for i in range(periods_ahead):
                    period_start = today + timedelta(days=i * days_per_period)
                    period_end = period_start + timedelta(days=days_per_period - 1)

                    # Calculate forecast with trend
                    forecast_qty = recent_avg + trend * (i + 1) * days_per_period
                    forecast_qty = max(0, forecast_qty * days_per_period)

                    # Add some variability for prediction intervals
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
                # Not enough data, use placeholder values
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
        """Generate placeholder forecasts when no data is available"""
        today = datetime.now().date()

        # Determine period length in days
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

            # Generate random forecast values
            base_qty = float(np.random.randint(50, 150))

            results.append({
                "product_sid": product_sid,
                "product_name": product_name,
                "category_name": category_name,
                "timeframe": timeframe,
                "period_start": period_start,
                "period_end": period_end,
                "forecast_qty": base_qty,
                "forecast_qty_lower": max(0, base_qty * 0.7),
                "forecast_qty_upper": base_qty * 1.3,
                "generated_at": datetime.now(),
                "model_version": f"{self.model_version} (placeholder)"
            })

        return results

    async def get_sales_trends(self,
                               product_sid: Optional[str] = None,
                               category_sid: Optional[str] = None,
                               days_back: int = 90) -> Dict[str, Any]:
        """Get sales trends over time with growth indicators"""
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
                COUNT(DISTINCT p.sid) as product_count
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
                    "trend": "stable"
                }

            df = pd.DataFrame(rows, columns=["date", "quantity", "revenue", "product_count"])

            # Convert date to string format for JSON
            dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
            quantities = [float(q) for q in df["quantity"]]
            revenues = [float(r) for r in df["revenue"]]

            # Calculate growth rates
            if len(df) >= 2:
                # Split data into two halves to compare
                half_point = len(df) // 2
                first_half_qty = df["quantity"].iloc[:half_point].mean()
                second_half_qty = df["quantity"].iloc[half_point:].mean()

                first_half_rev = df["revenue"].iloc[:half_point].mean()
                second_half_rev = df["revenue"].iloc[half_point:].mean()

                # Calculate growth as percentage
                if first_half_qty > 0:
                    qty_growth = (second_half_qty - first_half_qty) / first_half_qty * 100
                else:
                    qty_growth = 0

                if first_half_rev > 0:
                    rev_growth = (second_half_rev - first_half_rev) / first_half_rev * 100
                else:
                    rev_growth = 0

                # Determine trend
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

            return {
                "dates": dates,
                "quantities": quantities,
                "revenues": revenues,
                "growth": {
                    "quantity": round(float(qty_growth), 2),
                    "revenue": round(float(rev_growth), 2)
                },
                "trend": trend
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
                "trend": "error"
            }

    async def save_forecast(self, forecasts: List[Dict[str, Any]]) -> List[Prediction]:
        """Saves forecasts to the database"""
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

        # Update objects after commit
        for prediction in prediction_objects:
            await self.db.refresh(prediction)

        return prediction_objects

    async def get_product_analytics(self, product_sid: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a specific product"""
        try:
            # Get basic product info
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

            # Get sales data
            sales_query = text("""
                SELECT 
                    DATE_TRUNC('month', s.sold_at) as month,
                    SUM(s.sold_qty) as quantity,
                    SUM(s.sold_qty * s.sold_price) as revenue,
                    AVG(s.sold_price) as avg_price,
                    COUNT(s.id) as transaction_count
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

            # Get inventory data
            inventory_query = text("""
                SELECT 
                    SUM(wi.quantity) as warehouse_quantity,
                    COUNT(wi.id) as batch_count,
                    MIN(wi.expire_date) as nearest_expiry
                FROM 
                    warehouseitem wi
                WHERE 
                    wi.product_sid = :product_sid
                    AND wi.status = 'IN_STOCK'
            """)

            inventory_result = await self.db.execute(inventory_query, {"product_sid": product_sid})
            inventory_data = inventory_result.fetchone()

            # Get store data
            store_query = text("""
                SELECT 
                    SUM(si.quantity) as store_quantity,
                    AVG(si.price) as current_price,
                    COUNT(d.id) as active_discounts
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

            # Get sales trends
            trends = await self.get_sales_trends(product_sid=product_sid)

            # Generate future forecast
            forecasts = await self.generate_forecast(
                product_sid=product_sid,
                timeframe=TimeFrame.MONTH,
                periods_ahead=3
            )

            # Calculate additional KPIs
            if sales_rows:
                sales_data = []
                total_quantity = 0
                total_revenue = 0

                for row in sales_rows:
                    month_str = row.month.strftime("%Y-%m")
                    sales_data.append({
                        "month": month_str,
                        "quantity": float(row.quantity),
                        "revenue": float(row.revenue),
                        "avg_price": float(row.avg_price),
                        "transaction_count": int(row.transaction_count)
                    })
                    total_quantity += row.quantity
                    total_revenue += row.revenue

                avg_monthly_sales = total_quantity / len(sales_rows)
                avg_monthly_revenue = total_revenue / len(sales_rows)

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
            else:
                sales_data = []
                avg_monthly_sales = 0
                avg_monthly_revenue = 0
                turnover_rate = 0
                days_of_supply = 0
                warehouse_days_supply = 0

            # Get top selling products in same category
            category_query = text("""
                SELECT 
                    p.sid as product_sid,
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
                    product p ON wi.product_sid = p.sid
                WHERE 
                    p.category_sid = :category_sid
                    AND s.sold_at >= :min_date
                GROUP BY 
                    p.sid, p.name
                ORDER BY 
                    quantity DESC
                LIMIT 5
            """)

            category_result = await self.db.execute(
                category_query,
                {"category_sid": prod_data.category_sid, "min_date": min_date}
            )
            category_rows = category_result.fetchall()

            category_comparison = []
            for row in category_rows:
                category_comparison.append({
                    "product_sid": row.product_sid,
                    "product_name": row.product_name,
                    "quantity": float(row.quantity),
                    "revenue": float(row.revenue),
                    "is_current": row.product_sid == product_sid
                })

            # Return comprehensive analytics
            return {
                "product_info": {
                    "name": prod_data.product_name,
                    "barcode": prod_data.barcode,
                    "default_price": float(prod_data.default_price) if prod_data.default_price else 0,
                    "category": prod_data.category_name
                },
                "inventory": {
                    "warehouse_quantity": float(
                        inventory_data.warehouse_quantity) if inventory_data and inventory_data.warehouse_quantity else 0,
                    "store_quantity": float(
                        store_data.store_quantity) if store_data and store_data.store_quantity else 0,
                    "current_price": float(store_data.current_price) if store_data and store_data.current_price else 0,
                    "active_discounts": int(
                        store_data.active_discounts) if store_data and store_data.active_discounts else 0,
                    "nearest_expiry": inventory_data.nearest_expiry.strftime(
                        "%Y-%m-%d") if inventory_data and inventory_data.nearest_expiry else None
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
                    "turnover_rate": float(turnover_rate),
                    "days_of_supply": float(days_of_supply),
                    "warehouse_days_supply": float(warehouse_days_supply)
                },
                "category_comparison": category_comparison
            }

        except Exception as e:
            logger.error(f"Error getting product analytics: {str(e)}")
            return {"error": f"Failed to get analytics: {str(e)}"}