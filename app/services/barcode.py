import cv2
from pyzbar.pyzbar import decode
import base64
import numpy as np
from fastapi import HTTPException
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any


async def decode_barcode_from_base64(base64_image: str) -> str:
    """Декодирует штрих-код из base64-изображения"""
    try:
        # Декодируем base64 в изображение
        image_data = base64.b64decode(base64_image.split(',')[1] if ',' in base64_image else base64_image)
        image = Image.open(BytesIO(image_data))

        # Конвертируем PIL Image в numpy array для OpenCV
        image_np = np.array(image)

        # Для черно-белого изображения
        if len(image_np.shape) == 2:
            gray = image_np
        # Для цветного изображения
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Декодируем штрих-код
        decoded_objects = decode(gray)

        if not decoded_objects:
            raise HTTPException(status_code=400, detail="No barcode found in the image")

        # Возвращаем первый найденный штрих-код
        return decoded_objects[0].data.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding barcode: {str(e)}")


async def generate_barcode(barcode_type: str, data: str) -> Dict[str, Any]:
    """Генерирует штрих-код указанного типа"""
    try:
        if barcode_type.lower() not in ["qr", "code39", "code128", "ean13", "ean8"]:
            raise HTTPException(status_code=400, detail=f"Unsupported barcode type: {barcode_type}")

        if barcode_type.lower() == "qr":
            # Генерируем QR-код
            qr = cv2.QRCodeDetector()
            # Размер QR-кода
            size = 300
            # Создаем изображение
            img = np.zeros((size, size), np.uint8)
            # Заполняем белым цветом
            img.fill(255)
            # Генерируем QR-код
            qr_code = qr.encode(img, data)[0]

            # Кодируем в base64
            _, buffer = cv2.imencode('.png', qr_code)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "barcode_type": "qr",
                "data": data,
                "image": f"data:image/png;base64,{img_base64}"
            }

        elif barcode_type.lower() in ["code39", "code128", "ean13", "ean8"]:
            # Для других типов штрих-кодов можно использовать библиотеку python-barcode
            # Здесь добавить код для других типов штрих-кодов
            # Пока возвращаем заглушку
            return {
                "barcode_type": barcode_type,
                "data": data,
                "error": "Generation not implemented for this barcode type"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating barcode: {str(e)}")


async def verify_barcode(barcode: str, product_barcode: Optional[str] = None) -> bool:
    """Проверяет корректность штрих-кода и его соответствие продукту"""
    # Для EAN-13 можно проверить контрольную сумму
    if len(barcode) == 13 and barcode.isdigit():
        # Проверка контрольной суммы EAN-13
        sum_odd = sum(int(barcode[i]) for i in range(0, 12, 2))
        sum_even = sum(int(barcode[i]) * 3 for i in range(1, 12, 2))
        check_sum = (10 - ((sum_odd + sum_even) % 10)) % 10

        if int(barcode[12]) != check_sum:
            return False

    # Если указан штрих-код продукта, проверяем соответствие
    if product_barcode and barcode != product_barcode:
        return False

    return True