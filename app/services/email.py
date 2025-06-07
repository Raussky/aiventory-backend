from app.core.config import settings
from typing import List, Dict, Any
from loguru import logger
import asyncio
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


async def send_email_smtp(email_to: str, subject: str, body: str) -> bool:
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{settings.EMAILS_FROM_NAME} <{settings.EMAILS_FROM_EMAIL}>"
        message["To"] = email_to

        html_part = MIMEText(body, "html")
        message.attach(html_part)

        await aiosmtplib.send(
            message,
            hostname=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            start_tls=settings.SMTP_TLS,
        )

        logger.info(f"Email sent successfully to {email_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return False


async def send_verification_email(email_to: str, verification_code: str) -> bool:
    subject = "Подтверждение регистрации - AIventory"
    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color: #f5f5f5; padding: 20px 0;">
            <tr>
                <td align="center">
                    <table cellpadding="0" cellspacing="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <tr>
                            <td style="padding: 40px 30px; text-align: center; background: linear-gradient(135deg, #6322FE 0%, #8B5CF6 100%); border-radius: 8px 8px 0 0;">
                                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600;">AIventory</h1>
                                <p style="margin: 10px 0 0 0; color: #E9D5FF; font-size: 16px;">Система управления запасами</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 40px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #1F2937; font-size: 24px; font-weight: 600; text-align: center;">Подтверждение регистрации</h2>
                                <p style="margin: 0 0 30px 0; color: #4B5563; font-size: 16px; line-height: 24px; text-align: center;">
                                    Спасибо за регистрацию! Используйте код ниже для подтверждения вашего email адреса.
                                </p>
                                <div style="background-color: #F3F4F6; border-radius: 8px; padding: 30px; text-align: center; margin: 0 0 30px 0;">
                                    <p style="margin: 0 0 10px 0; color: #6B7280; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Ваш код подтверждения</p>
                                    <div style="font-size: 36px; font-weight: 700; color: #6322FE; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                        {verification_code}
                                    </div>
                                </div>
                                <p style="margin: 0 0 20px 0; color: #6B7280; font-size: 14px; line-height: 20px; text-align: center;">
                                    Этот код действителен в течение 24 часов. Если вы не регистрировались в нашей системе, просто проигнорируйте это письмо.
                                </p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 30px; background-color: #F9FAFB; border-radius: 0 0 8px 8px; text-align: center;">
                                <p style="margin: 0; color: #9CA3AF; font-size: 12px; line-height: 18px;">
                                    © 2024 AIventory. Все права защищены.<br>
                                    Это автоматическое сообщение, пожалуйста, не отвечайте на него.
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    return await send_email_smtp(email_to, subject, body)


async def send_password_reset_email(email_to: str, reset_code: str) -> bool:
    subject = "Восстановление пароля - AIventory"
    body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f5f5;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background-color: #f5f5f5; padding: 20px 0;">
            <tr>
                <td align="center">
                    <table cellpadding="0" cellspacing="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <tr>
                            <td style="padding: 40px 30px; text-align: center; background: linear-gradient(135deg, #6322FE 0%, #8B5CF6 100%); border-radius: 8px 8px 0 0;">
                                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600;">AIventory</h1>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 40px 30px;">
                                <h2 style="margin: 0 0 20px 0; color: #1F2937; font-size: 24px; font-weight: 600; text-align: center;">Восстановление пароля</h2>
                                <div style="background-color: #F3F4F6; border-radius: 8px; padding: 30px; text-align: center; margin: 0 0 30px 0;">
                                    <p style="margin: 0 0 10px 0; color: #6B7280; font-size: 14px;">Код для сброса пароля:</p>
                                    <div style="font-size: 36px; font-weight: 700; color: #6322FE; letter-spacing: 8px; font-family: 'Courier New', monospace;">
                                        {reset_code}
                                    </div>
                                </div>
                                <p style="margin: 0; color: #6B7280; font-size: 14px; text-align: center;">
                                    Код действителен в течение 1 часа.
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    return await send_email_smtp(email_to, subject, body)


async def send_expiry_notification(user_email: str, warehouse_items: List[Dict], store_items: List[Dict]) -> bool:
    subject = "Товары с истекающим сроком годности - AIventory"

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
                Это автоматическое уведомление системы управления запасами AIventory.
            </p>
        </div>
    </body>
    </html>
    """

    return await send_email_smtp(user_email, subject, body)


async def send_test_email(email_to: str) -> bool:
    subject = "Тестовое сообщение - AIventory"
    body = """
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2>Тестовое сообщение</h2>
        <p>Если вы видите это сообщение, значит настройки SMTP работают корректно.</p>
    </body>
    </html>
    """

    return await send_email_smtp(email_to, subject, body)