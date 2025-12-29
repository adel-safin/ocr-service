"""API endpoints для обратной связи и активного обучения"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging

import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from services.active_learning import ActiveLearningSystem
from api.schemas import CorrectionRequest

logger = logging.getLogger(__name__)

router = APIRouter()
active_learning = ActiveLearningSystem()


@router.post("/feedback/correction")
async def submit_correction_feedback(feedback_data: Dict[str, Any]):
    """
    Отправка обратной связи по исправлению
    
    Body:
        {
            "correction": {
                "original": "ошибочный текст",
                "corrected": "исправленный текст",
                "document_id": "doc_123",
                "context": "контекст (опционально)",
                "user_id": "user_123",
                "confidence": 1.0
            }
        }
    """
    try:
        result = active_learning.process_feedback(feedback_data)
        return JSONResponse(content={
            "success": True,
            "message": "Обратная связь по исправлению принята",
            "result": result
        })
    except Exception as e:
        logger.error(f"Ошибка при обработке feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/feedback/quality")
async def submit_quality_feedback(feedback_data: Dict[str, Any]):
    """
    Отправка оценки качества обработки
    
    Body:
        {
            "quality": {
                "document_id": "doc_123",
                "rating": 0.85,
                "issues": ["проблема1", "проблема2"],
                "user_id": "user_123"
            }
        }
    """
    try:
        result = active_learning.process_feedback(feedback_data)
        return JSONResponse(content={
            "success": True,
            "message": "Оценка качества принята",
            "result": result
        })
    except Exception as e:
        logger.error(f"Ошибка при обработке feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/feedback/document_type")
async def submit_document_type_feedback(feedback_data: Dict[str, Any]):
    """
    Отправка обратной связи по типу документа
    
    Body:
        {
            "document_type": {
                "document_id": "doc_123",
                "predicted_type": "Акт АОСР",
                "actual_type": "Акт АОСР",
                "user_id": "user_123"
            }
        }
    """
    try:
        result = active_learning.process_feedback(feedback_data)
        return JSONResponse(content={
            "success": True,
            "message": "Обратная связь по типу документа принята",
            "result": result
        })
    except Exception as e:
        logger.error(f"Ошибка при обработке feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/feedback/statistics")
async def get_feedback_statistics():
    """Получение статистики по обратной связи"""
    try:
        stats = active_learning.get_learning_statistics()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/feedback/analysis")
async def get_feedback_analysis():
    """Получение анализа паттернов в обратной связи"""
    try:
        analysis = active_learning.analyze_feedback_patterns()
        return JSONResponse(content=analysis)
    except Exception as e:
        logger.error(f"Ошибка при анализе: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.post("/feedback/auto_update")
async def trigger_auto_update():
    """Принудительный запуск автоматического обновления базы исправлений"""
    try:
        active_learning._auto_update_corrections()
        stats = active_learning.feedback_collector.get_statistics()
        return JSONResponse(content={
            "success": True,
            "message": "Автоматическое обновление выполнено",
            "statistics": stats
        })
    except Exception as e:
        logger.error(f"Ошибка при автоматическом обновлении: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@router.get("/feedback/export")
async def export_training_data():
    """Экспорт данных для обучения моделей"""
    try:
        export_path = active_learning.export_training_data()
        return JSONResponse(content={
            "success": True,
            "message": "Данные экспортированы",
            "export_path": export_path
        })
    except Exception as e:
        logger.error(f"Ошибка при экспорте: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")
