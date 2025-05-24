from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime
from enum import Enum

from app.schemas.inventory import ProductResponse

class TimeFrame(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

class PredictionBase(BaseModel):
    product_sid: str
    timeframe: TimeFrame
    period_start: date
    period_end: date
    forecast_qty: float = Field(..., ge=0)
    model_version: str

    class Config:
        protected_namespaces = ()

class PredictionCreate(PredictionBase):
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        protected_namespaces = ()

class PredictionResponse(PredictionBase):
    sid: str
    generated_at: datetime
    product: Optional[ProductResponse] = None
    forecast_qty_lower: Optional[float] = None
    forecast_qty_upper: Optional[float] = None

    class Config:
        from_attributes = True
        protected_namespaces = ()

class PredictionRequest(BaseModel):
    product_sid: str
    timeframe: TimeFrame = TimeFrame.MONTH
    periods: int = Field(3, ge=1, le=12)
    refresh: bool = False

class PredictionStatFilter(BaseModel):
    product_sid: Optional[str] = None
    category_sid: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    group_by: str = "day"

class SalesTrend(BaseModel):
    dates: List[str]
    quantities: List[float]
    revenues: List[float]
    growth: Dict[str, float]
    trend: str

class SeasonalityInfo(BaseModel):
    day_of_week: Dict[int, float]
    monthly: Dict[int, float]
    has_seasonality: bool

class ProductMetrics(BaseModel):
    product_sid: str
    product_name: str
    quantity: float
    revenue: float
    is_current: Optional[bool] = False

class ForecastPoint(BaseModel):
    period_start: str
    period_end: str
    forecast_qty: float
    forecast_qty_lower: float
    forecast_qty_upper: float

class ProductAnalyticsResponse(BaseModel):
    product_info: Dict[str, Any]
    inventory: Dict[str, Any]
    sales_data: List[Dict[str, Any]]
    trends: Dict[str, Any]
    forecasts: List[Dict[str, Any]]
    kpis: Dict[str, float]
    category_comparison: List[Dict[str, Any]]

class PredictionStatResponse(BaseModel):
    dates: List[str]
    products: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    quantity_data: List[Dict[str, Any]]
    revenue_data: List[Dict[str, Any]]
    category_quantity_data: Optional[List[Dict[str, Any]]] = None
    category_revenue_data: Optional[List[Dict[str, Any]]] = None