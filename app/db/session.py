# app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,
    future=True,
)

AsyncSessionLocal = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()