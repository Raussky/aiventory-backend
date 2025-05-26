from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.core.config import settings
from typing import List, Dict, Any
from loguru import logger
from celery import shared_task
import os


def is_smtp_configured():
    return (
            settings.SMTP_HOST and
            settings.SMTP_HOST != "smtp.gmail.com" and
            settings.SMTP_USER and
            settings.SMTP_PASSWORD and
            settings.EMAILS_FROM_EMAIL
    )


def get_mail_config():
    if not is_smtp_configured():
        return None

    return ConnectionConfig(
        MAIL_USERNAME=settings.SMTP_USER,
        MAIL_PASSWORD=settings.SMTP_PASSWORD,
        MAIL_FROM=settings.EMAILS_FROM_EMAIL,
        MAIL_PORT=settings.SMTP_PORT or 587,
        MAIL_SERVER=settings.SMTP_HOST,
        MAIL_FROM_NAME=settings.EMAILS_FROM_NAME or "Inventory System",
        MAIL_STARTTLS=settings.SMTP_TLS if settings.SMTP_TLS is not None else True,
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
        conf = get_mail_config()

        if not conf:
            logger.warning(f"SMTP not configured. Email to {email_to} with subject '{subject}' not sent.")
            logger.info(f"Configure SMTP settings in .env file: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAILS_FROM_EMAIL")
            return False

        message = MessageSchema(
            subject=subject,
            recipients=[email_to],
            body=body,
            subtype="html"
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        logger.info(f"Email sent successfully to {email_to}, subject: {subject}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email to {email_to}: {str(e)}")
        if "smtp.example.com" in str(e):
            logger.error("SMTP server is not configured properly. Please update SMTP_HOST in .env file.")
        return False


async def send_verification_email(email_to: str, verification_code: str):
    if not is_smtp_configured():
        logger.warning(f"SMTP not configured. Verification code for {email_to}: {verification_code}")
        logger.info("In development mode, use this code to verify the account.")
        logger.info("To enable email sending, configure SMTP settings in .env file.")
        return True

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
            <hr style="margin: 30px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p style="color: #999; font-size: 12px; text-align: center;">
                Это автоматическое сообщение. Пожалуйста, не отвечайте на него.
            </p>
        </div>
    </body>
    </html>
    """

    return await send_email(email_to=email_to, subject=subject, body=body)


@shared_task
def send_expiry_notification(user_email: str, warehouse_items: List[Dict], store_items: List[Dict]):
    if not is_smtp_configured():
        logger.warning(f"SMTP not configured. Expiry notification for {user_email} not sent.")
        logger.info(f"Expiring warehouse items: {len(warehouse_items)}")
        logger.info(f"Expiring store items: {len(store_items)}")
        return True

    subject = "Товары с истекающим сроком годности"

    warehouse_table = ""
    if warehouse_items:
        warehouse_table = """
        <h3 style="color: #333; margin-top: 20px;">Товары на складе:</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Название</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Количество</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Срок годности</th>
            </tr>
        """

        for item in warehouse_items:
            warehouse_table += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{item['product_name']}</td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;">{item['quantity']}</td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6; color: #dc3545; font-weight: bold;">{item['expire_date']}</td>
            </tr>
            """

        warehouse_table += "</table>"

    store_table = ""
    if store_items:
        store_table = """
        <h3 style="color: #333; margin-top: 20px;">Товары в магазине:</h3>
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Название</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Количество</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Срок годности</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Цена</th>
            </tr>
        """

        for item in store_items:
            store_table += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{item['product_name']}</td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;">{item['quantity']}</td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6; color: #dc3545; font-weight: bold;">{item['expire_date']}</td>
                <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;">{item['price']} ₸</td>
            </tr>
            """

        store_table += "</table>"

    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
        <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h1 style="color: #dc3545; text-align: center; margin-bottom: 30px;">⚠️ Уведомление о товарах с истекающим сроком годности</h1>
            <p style="color: #333; font-size: 16px; line-height: 1.5;">Система обнаружила товары, у которых скоро истекает срок годности:</p>

            {warehouse_table}

            {store_table}

            <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin-top: 20px;">
                <h4 style="color: #856404; margin-top: 0;">Рекомендации:</h4>
                <ul style="color: #856404; margin-bottom: 0;">
                    <li>Переместите товары со склада в магазин для быстрой реализации</li>
                    <li>Установите скидки на товары с коротким сроком годности</li>
                    <li>Рассмотрите возможность утилизации просроченных товаров</li>
                </ul>
            </div>

            <hr style="margin: 30px 0; border: none; border-top: 1px solid #e0e0e0;">
            <p style="color: #999; font-size: 12px; text-align: center;">
                Это автоматическое уведомление системы управления запасами.
            </p>
        </div>
    </body>
    </html>
    """

    return send_email(email_to=user_email, subject=subject, body=body)


async def send_test_email(email_to: str):
    if not is_smtp_configured():
        logger.warning(f"SMTP not configured. Cannot send test email to {email_to}")
        return False

    subject = "Тестовое сообщение"
    body = """
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Тестовое сообщение</h2>
        <p>Если вы видите это сообщение, значит настройки SMTP работают корректно.</p>
    </body>
    </html>
    """

    return await send_email(email_to=email_to, subject=subject, body=body)