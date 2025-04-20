# app/schemas/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from enum import Enum

class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MANAGER = "manager"

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserVerify(BaseModel):
    email: EmailStr
    code: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    sid: str
    is_verified: bool
    role: UserRole

    class Config:
        from_attributes = True