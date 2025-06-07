from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
import random
import string
from sqlalchemy import select, delete
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from typing import Dict
import uuid

from app.core.security import create_access_token, verify_password, get_password_hash
from app.core.config import settings
from app.db.session import get_db
from app.db.redis import get_redis
from app.models.users import User, VerificationToken
from app.schemas.user import UserCreate, UserResponse, UserVerify, UserLogin
from app.services.email import send_verification_email
from app.models.base import Base

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/auth/login")


async def get_current_user(
        db: AsyncSession = Depends(get_db),
        token: str = Depends(oauth2_scheme),
        redis: Redis = Depends(get_redis),
) -> User:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        token_jti = payload.get("jti")
        user_sid = payload.get("sub")

        if await redis.get(f"blacklist:{token_jti}"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

        if user_sid is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await db.execute(select(User).where(User.sid == user_sid))
    user = user.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/register", response_model=UserResponse)
async def register(
        user_in: UserCreate,
        db: AsyncSession = Depends(get_db),
):
    user_exists = await db.execute(select(User).where(User.email == user_in.email))
    if user_exists.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        sid=Base.generate_sid(),
        email=user_in.email,
        password_hash=get_password_hash(user_in.password),
        is_verified=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    verification_code = ''.join(random.choice(string.digits) for _ in range(6))
    expires_at = datetime.utcnow() + timedelta(hours=24)

    verification = VerificationToken(
        sid=Base.generate_sid(),
        user_id=user.id,
        token=verification_code,
        expires_at=expires_at,
    )
    db.add(verification)
    await db.commit()

    await send_verification_email(user.email, verification_code)

    return user


@router.post("/verify", response_model=UserResponse)
async def verify_email(
        verification_data: UserVerify,
        db: AsyncSession = Depends(get_db),
):
    user = await db.execute(select(User).where(User.email == verification_data.email))
    user = user.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    verification = await db.execute(
        select(VerificationToken)
        .where(
            VerificationToken.user_id == user.id,
            VerificationToken.token == verification_data.code,
            VerificationToken.expires_at > datetime.utcnow()
        )
    )
    verification = verification.scalar_one_or_none()

    if not verification:
        raise HTTPException(status_code=400, detail="Invalid or expired verification code")

    user.is_verified = True
    await db.delete(verification)
    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login")
async def login(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: AsyncSession = Depends(get_db),
):
    user = await db.execute(select(User).where(User.email == form_data.username))
    user = user.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")

    access_token = create_access_token(subject=user.sid)

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout(
        current_user: User = Depends(get_current_user),
        token: str = Depends(oauth2_scheme),
        redis: Redis = Depends(get_redis)
):
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    token_jti = payload.get("jti")

    exp_timestamp = payload.get("exp")
    current_timestamp = datetime.utcnow().timestamp()
    ttl = max(int(exp_timestamp - current_timestamp), 0)

    await redis.set(f"blacklist:{token_jti}", "1", ex=ttl)

    return {"message": "Logged out successfully"}


@router.post("/resend-verification", response_model=Dict[str, str])
async def resend_verification(
        email: str,
        db: AsyncSession = Depends(get_db),
):
    user = await db.execute(select(User).where(User.email == email))
    user = user.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_verified:
        raise HTTPException(status_code=400, detail="User already verified")

    await db.execute(
        delete(VerificationToken).where(VerificationToken.user_id == user.id)
    )

    verification_code = ''.join(random.choice(string.digits) for _ in range(6))
    expires_at = datetime.utcnow() + timedelta(hours=24)

    verification = VerificationToken(
        sid=Base.generate_sid(),
        user_id=user.id,
        token=verification_code,
        expires_at=expires_at,
    )
    db.add(verification)
    await db.commit()

    await send_verification_email(user.email, verification_code)

    return {"message": "Verification code sent successfully"}