from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.core.config import settings
from typing import List, Dict, Any
from loguru import logger
from celery import shared_task
import os

DEV_MODE = os.getenv("DEV_MODE", "0") == "1" or settings.SMTP_HOST == "smtp.example.com"

conf = ConnectionConfig(
    MAIL_USERNAME=settings.SMTP_USER,
    MAIL_PASSWORD=settings.SMTP_PASSWORD,
    MAIL_FROM=settings.EMAILS_FROM_EMAIL,
    MAIL_PORT=settings.SMTP_PORT,
    MAIL_SERVER=settings.SMTP_HOST,
    MAIL_FROM_NAME=settings.EMAILS_FROM_NAME,
    MAIL_STARTTLS=settings.SMTP_TLS,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)


async def send_email(
        email_to: str,
        subject: str,
        body: str,
        template_name: str = None,
        template_body: Dict[str, Any] = None
):
    if DEV_MODE:
        logger.info(f"DEV MODE: Email to {email_to}, subject: {subject}")
        logger.info(f"DEV MODE: Email body: {body[:100]}...")
        return True

    try:
        message = MessageSchema(
            subject=subject,
            recipients=[email_to],
            body=body,
            subtype="html"
        )

        fm = FastMail(conf)

        await fm.send_message(message)
        logger.info(f"Email sent to {email_to}, subject: {subject}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {email_to}: {str(e)}")
        return False


async def send_verification_email(email_to: str, verification_code: str):
    subject = "Подтверждение регистрации"
    body = f"""
    <html>
    <body>
        <h2>Подтверждение регистрации</h2>
        <p>Спасибо за регистрацию в системе управления запасами!</p>
        <p>Ваш код подтверждения: <strong>{verification_code}</strong></p>
        <p>Введите его на странице верификации для активации аккаунта.</p>
    </body>
    </html>
    """

    return await send_email(email_to=email_to, subject=subject, body=body)


@shared_task
def send_expiry_notification(user_email: str, warehouse_items: List[Dict], store_items: List[Dict]):
    subject = "Товары с истекающим сроком годности"

    warehouse_table = ""
    if warehouse_items:
        warehouse_table = """
        <h3>Товары на складе:</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Название</th>
                <th>Количество</th>
                <th>Срок годности</th>
            </tr>
        """

        for item in warehouse_items:
            warehouse_table += f"""
            <tr>
                <td>{item['product_name']}</td>
                <td>{item['quantity']}</td>
                <td>{item['expire_date']}</td>
            </tr>
            """

        warehouse_table += "</table>"

    store_table = ""
    if store_items:
        store_table = """
        <h3>Товары в магазине:</h3>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Название</th>
                <th>Количество</th>
                <th>Срок годности</th>
                <th>Цена</th>
            </tr>
        """

        for item in store_items:
            store_table += f"""
            <tr>
                <td>{item['product_name']}</td>
                <td>{item['quantity']}</td>
                <td>{item['expire_date']}</td>
                <td>{item['price']}</td>
            </tr>
            """

        store_table += "</table>"

    body = f"""
    <html>
    <body>
        <h2>Уведомление о товарах с истекающим сроком годности</h2>
        <p>Система обнаружила товары, у которых скоро истекает срок годности:</p>

        {warehouse_table}

        {store_table}

        <p>Рекомендуется принять меры по утилизации или уценке данных товаров.</p>
    </body>
    </html>
    """

    return send_email(email_to=user_email, subject=subject, body=body)