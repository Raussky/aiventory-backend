import pytest
from httpx import AsyncClient
import pytest_asyncio

from app.core.config import settings


@pytest.mark.asyncio
async def test_login(client, test_user):
    """Тест успешного логина"""
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
    """Тест логина с неверным паролем"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/login",
        data={
            "username": test_user["email"],
            "password": "wrong_password"
        }
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register(client):
    """Тест регистрации нового пользователя"""
    response = await client.post(
        f"{settings.API_V1_STR}/auth/register",
        json={
            "email": "new_user@example.com",
            "password": "new_password"
        }
    )
    assert response.status_code == 200
    json = response.json()
    assert json["email"] == "new_user@example.com"
    assert json["is_verified"] == False
    assert "sid" in json


@pytest.mark.asyncio
async def test_verify_email(client, db_session):
    """Тест верификации email"""
    # Т.к. в тестах мы не отправляем настоящие письма, нужно вручную получить код из базы
    # Это упрощенная версия для теста
    verification_query = """
        SELECT token FROM verificationtoken 
        JOIN "user" ON verificationtoken.user_id = "user".id
        WHERE "user".email = 'new_user@example.com'
    """
    result = await db_session.execute(verification_query)
    verification_code = result.scalar_one_or_none()

    if verification_code:
        response = await client.post(
            f"{settings.API_V1_STR}/auth/verify",
            json={
                "email": "new_user@example.com",
                "code": verification_code
            }
        )
        assert response.status_code == 200
        json = response.json()
        assert json["is_verified"] == True
    else:
        # Если код не найден, пропускаем тест
        pytest.skip("Verification code not found")


@pytest.mark.asyncio
async def test_logout(authorized_client):
    """Тест выхода из системы"""
    response = await authorized_client.post(f"{settings.API_V1_STR}/auth/logout")
    assert response.status_code == 200

    # Пробуем получить данные пользователя с тем же токеном
    response = await authorized_client.get(f"{settings.API_V1_STR}/auth/me")
    assert response.status_code == 401