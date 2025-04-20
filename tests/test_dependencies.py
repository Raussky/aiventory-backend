import pytest
from fastapi import HTTPException
from jose import jwt
import uuid
from datetime import datetime, timedelta

from app.core.dependencies import (
    get_current_user, get_current_active_user,
    get_admin_user, rate_limit_dependency
)
from app.core.config import settings
from app.models.users import User, UserRole


@pytest.mark.asyncio
async def test_get_current_user_valid_token(db_session, test_user, mock_redis):
    """Test extracting current user from valid token"""
    # Create a token for test_user
    payload = {
        "sub": test_user["sid"],
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "jti": str(uuid.uuid4())
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    # Configure redis mock to indicate token not blacklisted
    mock_redis.get.return_value = None

    # Call the dependency
    user = await get_current_user(db_session, token, mock_redis)

    assert user is not None
    assert user.email == test_user["email"]
    assert user.sid == test_user["sid"]

    # Verify redis was checked for blacklist
    mock_redis.get.assert_called_once()
    assert "blacklist:" in mock_redis.get.call_args[0][0]


@pytest.mark.asyncio
async def test_get_current_user_blacklisted_token(db_session, test_user, mock_redis):
    """Test extracting current user from blacklisted token"""
    # Create a token for test_user
    token_jti = str(uuid.uuid4())
    payload = {
        "sub": test_user["sid"],
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "jti": token_jti
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    # Configure redis mock to indicate token is blacklisted
    mock_redis.get.return_value = "1"  # Token in blacklist

    # Call the dependency - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(db_session, token, mock_redis)

    assert excinfo.value.status_code == 401
    assert "revoked" in excinfo.value.detail.lower()

    # Verify redis was checked for blacklist
    mock_redis.get.assert_called_once()
    assert f"blacklist:{token_jti}" == mock_redis.get.call_args[0][0]


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(db_session, mock_redis):
    """Test extracting current user from invalid token"""
    # Create an invalid token
    token = "invalid.token.string"

    # Call the dependency - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(db_session, token, mock_redis)

    assert excinfo.value.status_code == 401
    assert "credentials" in excinfo.value.detail.lower()

    # Redis should not be called for invalid tokens
    mock_redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_user_expired_token(db_session, test_user, mock_redis):
    """Test extracting current user from expired token"""
    # Create an expired token
    payload = {
        "sub": test_user["sid"],
        "exp": datetime.utcnow() - timedelta(minutes=30),  # Expired
        "jti": str(uuid.uuid4())
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    # Call the dependency - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(db_session, token, mock_redis)

    assert excinfo.value.status_code == 401

    # Redis should not be called for expired tokens
    mock_redis.get.assert_not_called()


@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(db_session, mock_redis):
    """Test extracting nonexistent user from token"""
    # Create token for non-existent user
    payload = {
        "sub": "nonexistent_user_sid",
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "jti": str(uuid.uuid4())
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    # Configure redis mock to indicate token not blacklisted
    mock_redis.get.return_value = None

    # Call the dependency - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(db_session, token, mock_redis)

    assert excinfo.value.status_code == 401
    assert "credentials" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_get_current_active_user(db_session):
    """Test get_current_active_user dependency"""
    # Create verified user
    verified_user = User(
        email="verified@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.MANAGER
    )

    # Call dependency with verified user
    result = await get_current_active_user(verified_user)
    assert result is verified_user

    # Create unverified user
    unverified_user = User(
        email="unverified@example.com",
        password_hash="hashed_password",
        is_verified=False,
        role=UserRole.MANAGER
    )

    # Call dependency with unverified user - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_current_active_user(unverified_user)

    assert excinfo.value.status_code == 403
    assert "not verified" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_get_admin_user(db_session):
    """Test get_admin_user dependency"""
    # Create admin user
    admin_user = User(
        email="admin@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.ADMIN
    )

    # Call dependency with admin user
    result = await get_admin_user(admin_user)
    assert result is admin_user

    # Create owner user
    owner_user = User(
        email="owner@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.OWNER
    )

    # Call dependency with owner user (should also have admin privileges)
    result = await get_admin_user(owner_user)
    assert result is owner_user

    # Create manager user (not an admin)
    manager_user = User(
        email="manager@example.com",
        password_hash="hashed_password",
        is_verified=True,
        role=UserRole.MANAGER
    )

    # Call dependency with manager user - should raise exception
    with pytest.raises(HTTPException) as excinfo:
        await get_admin_user(manager_user)

    assert excinfo.value.status_code == 403
    assert "permissions" in excinfo.value.detail.lower()


@pytest.mark.asyncio
async def test_rate_limit_dependency(mock_redis):
    """Test rate limit dependency"""
    # Create a rate limit dependency with small limits for testing
    rate_limit = rate_limit_dependency(requests_limit=5, time_window=60)

    # Mock request object
    class MockRequest:
        class MockClient:
            host = "127.0.0.1"

        client = MockClient()

    request = MockRequest()

    # Test normal usage (under limit)
    mock_redis.incr.return_value = 1  # First request
    await rate_limit(request, mock_redis)

    # Verify Redis interaction
    mock_redis.incr.assert_called_once()
    assert "rate_limit:127.0.0.1" == mock_redis.incr.call_args[0][0]

    mock_redis.expire.assert_called_once()
    assert "rate_limit:127.0.0.1" == mock_redis.expire.call_args[0][0]
    assert 60 == mock_redis.expire.call_args[0][1]  # time_window

    # Reset mocks for next test
    mock_redis.reset_mock()

    # Test limit reached
    mock_redis.incr.return_value = 6  # Over the limit

    # Should raise HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await rate_limit(request, mock_redis)

    assert excinfo.value.status_code == 429
    assert "too many requests" in excinfo.value.detail.lower()

    # Verify Redis interaction
    mock_redis.incr.assert_called_once()
    assert "rate_limit:127.0.0.1" == mock_redis.incr.call_args[0][0]

    # expire should not be called for existing keys
    mock_redis.expire.assert_not_called()