"""API endpoints"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import tempfile
import logging

import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
from core.correctors import AutoCorrectionSystem
from api.schemas import (
    CorrectionRequest, CorrectionResponse, ProcessingResult,
    BatchProcessingResult, CorrectionsDBResponse
)
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()
pipeline = DocumentPipeline()
corrector = AutoCorrectionSystem()


@router.post("/process", response_model=ProcessingResult)
async def process_document(
    file: UploadFile = File(...),
    template: Optional[str] = Form(None),
    required_fields: Optional[str] = Form(None)
):
    """
    Основной endpoint обработки документа
    
    Args:
        file: Загружаемый файл
        template: Тип шаблона документа (опционально)
        required_fields: Список обязательных полей через запятую
        
    Returns:
        Результаты обработки
    """
    # Сохранение файла во временную директорию
    temp_file = None
    try:
        # Создание временного файла
        file_ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name
        
        # Парсинг обязательных полей
        fields_list = None
        if required_fields:
            fields_list = [f.strip() for f in required_fields.split(',')]
        
        # Обработка документа
        result = pipeline.process(temp_file, template, fields_list)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
    finally:
        # Удаление временного файла
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass


@router.post("/batch_process", response_model=BatchProcessingResult)
async def batch_process(files: List[UploadFile] = File(...), template: Optional[str] = Form(None)):
    """
    Пакетная обработка документов с выявлением новых ошибок
    
    Args:
        files: Список загружаемых файлов
        template: Тип шаблона (опционально)
        
    Returns:
        Результаты обработки всех документов
    """
    temp_files = []
    try:
        # Сохранение всех файлов
        for file in files:
            file_ext = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_files.append(tmp.name)
        
        # Пакетная обработка
        result = pipeline.batch_process(temp_files, template)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Ошибка при пакетной обработке: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
    finally:
        # Удаление временных файлов
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass


@router.post("/confirm_correction", response_model=CorrectionResponse)
async def confirm_correction(correction_data: CorrectionRequest):
    """
    Подтверждение исправления оператором
    
    Args:
        correction_data: Данные исправления
        
    Returns:
        Результат подтверждения
    """
    try:
        if correction_data.add_to_db:
            corrector.add_correction(
                correction_data.original,
                correction_data.corrected,
                confirm=False
            )
        
        return CorrectionResponse(
            success=True,
            message="Исправление подтверждено и добавлено в базу",
            correction_id=f"corr_{hash(correction_data.original)}"
        )
        
    except Exception as e:
        logger.error(f"Ошибка при подтверждении исправления: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/corrections_db", response_model=CorrectionsDBResponse)
async def get_corrections():
    """
    Получение текущей базы автозамен
    
    Returns:
        База исправлений
    """
    try:
        corrections = corrector.corrections_db
        
        # Получение времени последнего обновления
        last_updated = None
        if os.path.exists(settings.CORRECTIONS_DB):
            from datetime import datetime
            last_updated = datetime.fromtimestamp(
                os.path.getmtime(settings.CORRECTIONS_DB)
            ).isoformat()
        
        return CorrectionsDBResponse(
            corrections=corrections,
            total_count=len(corrections),
            last_updated=last_updated
        )
        
    except Exception as e:
        logger.error(f"Ошибка при получении базы исправлений: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {"status": "healthy", "service": "OCR Document Processing"}
