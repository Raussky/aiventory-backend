from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.core.config import settings
from typing import List, Dict, Any
from loguru import logger
from celery import shared_task
import os

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
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h1 style="color: #6322FE; text-align: center; margin-bottom: 30px;">Подтверждение регистрации</h1>
            <p style="color: #333; font-size: 16px; line-height: 1.5;">Спасибо за регистрацию в системе управления запасами!</p>
            <p style="color: #333; font-size: 16px; line-height: 1.5;">Ваш код подтверждения:</p>
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; text-align: center; margin: 20px 0;">
                <h2 style="color: #6322FE; font-size: 32px; letter-spacing: 5px; margin: 0;">{verification_code}</h2>
            </div>
            <p style="color: #666; font-size: 14px; line-height: 1.5;">Этот код действителен в течение 24 часов.</p>
            <p style="color: #666; font-size: 14px; line-height: 1.5;">Если вы не регистрировались в нашей системе, просто проигнорируйте это письмо.</p>
        </div>
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