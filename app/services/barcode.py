import cv2
from pyzbar.pyzbar import decode
import base64
import numpy as np
from fastapi import HTTPException
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any
import logging

# Настройка логирования для отладки
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def decode_barcode_from_base64(base64_image: str) -> str:
    """Декодирует штрих-код из base64-изображения"""
    try:
        # Проверяем, что base64_image не пустой
        if not base64_image:
            logger.error("Пустое base64-изображение")
            raise HTTPException(status_code=400, detail="Empty base64 image provided")

        # Логируем размер изображения
        logger.info(f"Размер base64-строки: {len(base64_image)} символов")

        try:
            # Декодируем base64 в изображение
            if ',' in base64_image:
                logger.info("Найден разделитель в base64")
                base64_part = base64_image.split(',')[1]
            else:
                base64_part = base64_image

            image_data = base64.b64decode(base64_part)
            logger.info(f"Размер декодированных данных: {len(image_data)} байт")

            image = Image.open(BytesIO(image_data))
            logger.info(f"Изображение открыто. Размер: {image.size}, формат: {image.format}")
        except Exception as e:
            logger.error(f"Ошибка при декодировании base64: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

        # Конвертируем PIL Image в numpy array для OpenCV
        image_np = np.array(image)
        logger.info(f"Форма массива изображения: {image_np.shape}")

        # Для черно-белого изображения
        if len(image_np.shape) == 2:
            logger.info("Обрабатываем черно-белое изображение")
            gray = image_np
        # Для цветного изображения
        else:
            logger.info("Конвертируем цветное изображение в оттенки серого")
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Пробуем улучшить контраст изображения
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Декодируем штрих-код
        logger.info("Пытаемся декодировать штрих-код")
        decoded_objects = decode(gray)
        logger.info(f"Найдено штрих-кодов: {len(decoded_objects)}")

        if not decoded_objects:
            # Попробуем изменить параметры изображения
            logger.info("Штрих-код не найден, пробуем обработать изображение")

            # Изменяем размер изображения
            resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            decoded_objects = decode(resized)

            if not decoded_objects:
                # Применяем фильтр для улучшения границ
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                decoded_objects = decode(thresh)

                if not decoded_objects:
                    logger.error("Штрих-код не обнаружен даже после обработки")
                    raise HTTPException(status_code=400,
                                        detail="No barcode found in the image. Please make sure the barcode is clear and try again.")

        # Возвращаем первый найденный штрих-код
        barcode_data = decoded_objects[0].data.decode('utf-8')
        barcode_type = decoded_objects[0].type
        logger.info(f"Успешно декодирован штрих-код типа {barcode_type}: {barcode_data}")

        return barcode_data
    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        logger.error(f"Необработанная ошибка: {str(e)}")
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