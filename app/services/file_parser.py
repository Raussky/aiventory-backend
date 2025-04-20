# app/services/file_parser.py
import pandas as pd
from fastapi import UploadFile
from typing import List, Dict, Any
import io


async def parse_excel(file: UploadFile) -> List[Dict[str, Any]]:
    contents = await file.read()
    file_obj = io.BytesIO(contents)

    # Проверка расширения файла и выбор правильной библиотеки
    if file.filename.endswith('.xlsx'):
        df = pd.read_excel(file_obj, engine='openpyxl')
    elif file.filename.endswith('.xls'):
        df = pd.read_excel(file_obj, engine='xlrd')
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .xls")

    # Конвертируем DataFrame в список словарей
    records = df.to_dict('records')
    return records


async def parse_csv(file: UploadFile, delimiter: str = ',') -> List[Dict[str, Any]]:
    contents = await file.read()
    file_obj = io.BytesIO(contents)

    df = pd.read_csv(file_obj, delimiter=delimiter)
    records = df.to_dict('records')
    return records


async def detect_and_parse_file(file: UploadFile) -> List[Dict[str, Any]]:
    """Определяет тип файла и применяет соответствующий парсер"""
    if file.filename.endswith(('.xlsx', '.xls')):
        return await parse_excel(file)
    elif file.filename.endswith('.csv'):
        return await parse_csv(file)
    else:
        raise ValueError("Unsupported file format. Use .xlsx, .xls, or .csv")