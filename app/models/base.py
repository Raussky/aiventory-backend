# app/models/base.py
from sqlalchemy import Column, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase
import uuid
import nanoid

class Base(DeclarativeBase):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sid = Column(String(22), unique=True, nullable=False, index=True)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    @staticmethod
    def generate_sid():
        """Генерирует короткий ID для внешнего API"""
        return nanoid.generate(size=22)