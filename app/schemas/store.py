from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from app.schemas.inventory import ProductResponse

class StoreItemStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REMOVED = "removed"

class StoreItemBase(BaseModel):
    warehouse_item_sid: str
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)

    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

class StoreItemCreate(StoreItemBase):
    pass

class StoreItemResponse(StoreItemBase):
    sid: str
    moved_at: datetime
    status: StoreItemStatus
    product: Optional[ProductResponse] = None
    expire_date: Optional[datetime] = None
    current_discounts: Optional[List[Dict[str, Any]]] = []

    class Config:
        from_attributes = True

class DiscountBase(BaseModel):
    store_item_sid: str
    percentage: float = Field(..., ge=0, le=100)
    starts_at: datetime
    ends_at: datetime

    @validator('ends_at')
    def end_date_after_start_date(cls, v, values):
        if 'starts_at' in values and v <= values['starts_at']:
            raise ValueError('End date must be after start date')
        return v

class DiscountCreate(DiscountBase):
    pass

class DiscountResponse(DiscountBase):
    sid: str
    created_by_sid: str

    class Config:
        from_attributes = True

class SaleBase(BaseModel):
    store_item_sid: str
    sold_qty: int = Field(..., gt=0)
    sold_price: float = Field(..., ge=0)

class SaleCreate(SaleBase):
    pass

class SaleResponse(SaleBase):
    sid: str
    sold_at: datetime
    cashier_sid: str

    class Config:
        from_attributes = True

class StoreItemFilter(BaseModel):
    status: Optional[StoreItemStatus] = None
    expired_only: bool = False
    search: Optional[str] = None
    category_sid: Optional[str] = None