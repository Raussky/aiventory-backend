import pytest
from httpx import AsyncClient
from datetime import datetime

from app.core.config import settings
from app.models.users import User, UserRole


@pytest.mark.asyncio
async def test_login_success(client, test_user):
    """Test successful login"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": test_user["email"],
            "password": test_user["password"]
        }
    )
    assert response.status_code == 200
    json = response.json()
    assert "access_token" in json
    assert json["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_incorrect_password(client, test_user):
    """Test login with incorrect password"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": test_user["email"],
            "password": "wrong_password"
        }
    )
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    """Test login with nonexistent user"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": "nonexistent@example.com",
            "password": "password123"
        }
    )
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_login_unverified_user(client, db_session):
    """Test login with unverified user"""
    # Create unverified user
    user = User(
        email="unverified@example.com",
        password_hash="$2b$12$CiJAbAuTGhDKQITsYUNiLehimzXqKdAvxaQRGu6u8Z9eL85Bsu9Sq",  # "password123"
        is_verified=False,
        role=UserRole.MANAGER
    )
    db_session.add(user)
    await db_session.commit()

    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": "unverified@example.com",
            "password": "password123"
        }
    )
    assert response.status_code == 403
    assert "not verified" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_success(client, mock_email_service):
    """Test successful registration"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/register",
        json={
            "email": "new_user@example.com",
            "password": "secure_password123"
        }
    )
    assert response.status_code == 200
    json = response.json()
    assert json["email"] == "new_user@example.com"
    assert json["is_verified"] is False
    assert "sid" in json

    # Check that email service was called
    mock_email_service.assert_called_once()


@pytest.mark.asyncio
async def test_register_duplicate_email(client, test_user, mock_email_service):
    """Test registration with existing email"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/register",
        json={
            "email": test_user["email"],
            "password": "another_password"
        }
    )
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"].lower()

    # Check that email service was not called
    mock_email_service.assert_not_called()


@pytest.mark.asyncio
async def test_register_weak_password(client, mock_email_service):
    """Test registration with weak password"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/register",
        json={
            "email": "weak_password@example.com",
            "password": "short"
        }
    )
    # Assuming validation happens at Pydantic level
    assert response.status_code == 422

    # Check that email service was not called
    mock_email_service.assert_not_called()


@pytest.mark.asyncio
async def test_verify_email_success(client, db_session, verification_token):
    """Test successful email verification"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/verify",
        json={
            "email": "test@example.com",
            "code": "123456"
        }
    )
    assert response.status_code == 200
    json = response.json()
    assert json["is_verified"] is True

    # Verify that token was deleted
    result = await db_session.execute(
        "SELECT COUNT(*) FROM verificationtoken WHERE token = '123456'"
    )
    count = result.scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_verify_email_invalid_code(client, verification_token):
    """Test email verification with invalid code"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/verify",
        json={
            "email": "test@example.com",
            "code": "654321"  # Wrong code
        }
    )
    assert response.status_code == 400
    assert "invalid" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_verify_email_expired_code(client, db_session, verification_token):
    """Test email verification with expired code"""
    # Update token to be expired
    result = await db_session.execute(
        "UPDATE verificationtoken SET expires_at = :expired_at WHERE token = '123456'",
        {"expired_at": datetime.utcnow().isoformat()}
    )
    await db_session.commit()

    response = await client.post(
        f"{settings.API_V1_STR}/auth/verify",
        json={
            "email": "test@example.com",
            "code": "123456"
        }
    )
    assert response.status_code == 400
    assert "expired" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_logout_success(authorized_client, mock_redis):
    """Test successful logout"""
    response = await authorized_client.post(f"{settings.API_V1_STR}/auth/logout")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "logged out" in response.json()["message"].lower()

    # Check that token was blacklisted
    mock_redis.set.assert_called_once()
    assert "blacklist:" in mock_redis.set.call_args[0][0]


@pytest.mark.asyncio
async def test_logout_unauthorized(client):
    """Test logout without authentication"""
    response = await client.post(f"{settings.API_V1_STR}/auth/logout")
    assert response.status_code == 401