from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

class PredictionCreate(PredictionBase):
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class PredictionResponse(PredictionBase):
    sid: str
    generated_at: datetime
    product: Optional[ProductResponse] = None

    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    product_sid: str
    timeframe: TimeFrame = TimeFrame.MONTH
    periods: int = Field(3, ge=1, le=12)
    refresh: bool = False

class PredictionStatFilter(BaseModel):
    product_sid: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class PredictionStatResponse(BaseModel):
    dates: List[str]
    products: List[Dict[str, Any]]
    quantity_data: List[Dict[str, Any]]
    revenue_data: List[Dict[str, Any]]