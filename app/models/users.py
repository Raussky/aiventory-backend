# app/models/users.py
from sqlalchemy import Column, String, Boolean, Text, Enum, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID  # Добавить этот импорт
from app.models.base import Base
import enum

class UserRole(str, enum.Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MANAGER = "manager"

class User(Base):
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    is_verified = Column(Boolean, default=False)
    role = Column(Enum(UserRole), default=UserRole.OWNER)

class VerificationToken(Base):
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    token = Column(String(6), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)