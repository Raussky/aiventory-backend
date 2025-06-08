from sqlalchemy import Column, String, Integer, Float, DateTime, Date, ForeignKey, Enum, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.models.base import Base
from datetime import datetime, timedelta, timezone


class Currency(str, enum.Enum):
    KZT = "kzt"
    USD = "usd"
    EUR = "eur"
    RUB = "rub"


class StorageDurationType(str, enum.Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class UrgencyLevel(str, enum.Enum):
    NORMAL = "normal"
    URGENT = "urgent"
    CRITICAL = "critical"


class Category(Base):
    name = Column(String, unique=True, nullable=False)
    products = relationship("Product", back_populates="category")


class Product(Base):
    category_sid = Column(String(22), ForeignKey("category.sid"), nullable=False)
    category = relationship("Category", back_populates="products")
    name = Column(String, nullable=False)
    barcode = Column(String, unique=True)
    default_unit = Column(String)
    default_price = Column(Float)
    currency = Column(Enum(Currency), default=Currency.KZT)
    storage_duration = Column(Integer, default=30)
    storage_duration_type = Column(Enum(StorageDurationType), default=StorageDurationType.DAY)

    warehouse_items = relationship("WarehouseItem", back_populates="product")
    predictions = relationship("Prediction", back_populates="product")


class WarehouseItemStatus(str, enum.Enum):
    IN_STOCK = "in_stock"
    MOVED = "moved"
    DISCARDED = "discarded"


class Upload(Base):
    user_sid = Column(String(22), ForeignKey("user.sid"), nullable=False)
    user = relationship("User")
    file_name = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    rows_imported = Column(Integer, default=0)

    warehouse_items = relationship("WarehouseItem", back_populates="upload")


class WarehouseItem(Base):
    upload_sid = Column(String(22), ForeignKey("upload.sid"), nullable=False)
    upload = relationship("Upload", back_populates="warehouse_items")
    product_sid = Column(String(22), ForeignKey("product.sid"), nullable=False)
    product = relationship("Product", back_populates="warehouse_items")
    batch_code = Column(String)
    quantity = Column(Integer, default=0)
    expire_date = Column(Date)
    received_at = Column(Date, nullable=False)
    status = Column(Enum(WarehouseItemStatus), default=WarehouseItemStatus.IN_STOCK)
    urgency_level = Column(Enum(UrgencyLevel), default=UrgencyLevel.NORMAL)

    store_items = relationship("StoreItem", back_populates="warehouse_item")


class StoreItemStatus(str, enum.Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REMOVED = "removed"


class StoreItem(Base):
    warehouse_item_sid = Column(String(22), ForeignKey("warehouseitem.sid"), nullable=False)
    warehouse_item = relationship("WarehouseItem", back_populates="store_items")
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    moved_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    status = Column(Enum(StoreItemStatus), default=StoreItemStatus.ACTIVE)

    sales = relationship("Sale", back_populates="store_item")
    discounts = relationship("Discount", back_populates="store_item")
    cart_items = relationship("CartItem", back_populates="store_item")


class Discount(Base):
    store_item_sid = Column(String(22), ForeignKey("storeitem.sid"), nullable=False)
    store_item = relationship("StoreItem", back_populates="discounts")
    percentage = Column(Float, nullable=False)
    starts_at = Column(DateTime(timezone=True), nullable=False)
    ends_at = Column(DateTime(timezone=True), nullable=False)
    created_by_sid = Column(String(22), ForeignKey("user.sid"), nullable=False)
    created_by = relationship("User")


class CartItem(Base):
    store_item_sid = Column(String(22), ForeignKey("storeitem.sid"), nullable=False)
    store_item = relationship("StoreItem", back_populates="cart_items")
    user_sid = Column(String(22), ForeignKey("user.sid"), nullable=False)
    user = relationship("User")
    quantity = Column(Integer, nullable=False)
    price_per_unit = Column(Float, nullable=False)
    added_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Sale(Base):
    store_item_sid = Column(String(22), ForeignKey("storeitem.sid"), nullable=False)
    store_item = relationship("StoreItem", back_populates="sales")
    sold_qty = Column(Integer, nullable=False)
    sold_price = Column(Float, nullable=False)
    sold_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    cashier_sid = Column(String(22), ForeignKey("user.sid"), nullable=False)
    cashier = relationship("User")


class TimeFrame(str, enum.Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class Prediction(Base):
    product_sid = Column(String(22), ForeignKey("product.sid"), nullable=False)
    product = relationship("Product", back_populates="predictions")
    user_sid = Column(String(22), ForeignKey("user.sid"), nullable=False)
    user = relationship("User")
    timeframe = Column(Enum(TimeFrame), nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    forecast_qty = Column(Float, nullable=False)
    forecast_qty_lower = Column(Float, nullable=True)
    forecast_qty_upper = Column(Float, nullable=True)
    generated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    model_version = Column(String, nullable=False)