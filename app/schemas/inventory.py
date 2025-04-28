from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union
from datetime import date, datetime
from enum import Enum

class Currency(str, Enum):
    KZT = "kzt"
    USD = "usd"
    EUR = "eur"
    RUB = "rub"

class StorageDurationType(str, Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"

class CategoryBase(BaseModel):
    name: str

class CategoryCreate(CategoryBase):
    pass

class CategoryResponse(CategoryBase):
    sid: str

    class Config:
        from_attributes = True

class ProductBase(BaseModel):
    name: str
    category_sid: str
    barcode: Optional[str] = None
    default_unit: Optional[str] = None
    default_price: Optional[float] = None
    currency: Optional[Currency] = Currency.KZT
    storage_duration: Optional[int] = 30
    storage_duration_type: Optional[StorageDurationType] = StorageDurationType.DAY

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    sid: str
    category: Optional[CategoryResponse] = None

    class Config:
        from_attributes = True

class WarehouseItemStatus(str, Enum):
    IN_STOCK = "in_stock"
    MOVED = "moved"
    DISCARDED = "discarded"

class WarehouseItemBase(BaseModel):
    product_sid: str
    batch_code: Optional[str] = None
    quantity: int = Field(..., gt=0)
    expire_date: Optional[date] = None
    received_at: date

    @validator('quantity')
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v

class WarehouseItemCreate(WarehouseItemBase):
    pass

class WarehouseItemResponse(WarehouseItemBase):
    sid: str
    upload_sid: str
    status: WarehouseItemStatus
    product: ProductResponse

    class Config:
        from_attributes = True

class UploadBase(BaseModel):
    file_name: str

class UploadCreate(UploadBase):
    pass

class UploadResponse(UploadBase):
    sid: str
    uploaded_at: datetime
    rows_imported: int

    class Config:
        from_attributes = True

class WarehouseItemFilter(BaseModel):
    upload_sid: Optional[str] = None
    expire_soon: Optional[bool] = False
    product_sid: Optional[str] = None
    status: Optional[WarehouseItemStatus] = None
    search: Optional[str] = None