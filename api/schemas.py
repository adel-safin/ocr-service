"""Pydantic схемы для API"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class CorrectionRequest(BaseModel):
    """Запрос на подтверждение исправления"""
    original: str = Field(..., description="Оригинальный текст")
    corrected: str = Field(..., description="Исправленный текст")
    context: Optional[str] = Field(None, description="Контекст исправления")
    add_to_db: bool = Field(True, description="Добавить в базу автозамен")


class CorrectionResponse(BaseModel):
    """Ответ на подтверждение исправления"""
    success: bool
    message: str
    correction_id: Optional[str] = None


class FieldValidation(BaseModel):
    """Результат валидации поля"""
    value: str
    confidence: float
    valid: bool
    message: Optional[str] = None
    suggested_correction: Optional[str] = None


class QualityReport(BaseModel):
    """Отчет о качестве"""
    overall_quality: float
    image_quality: Dict[str, Any]
    ocr_confidence: float
    issues: List[Dict[str, Any]]
    needs_review: bool


class ExtractedData(BaseModel):
    """Извлеченные данные"""
    critical_fields: Dict[str, FieldValidation]
    full_text: str
    raw_text: Optional[str] = None


class ProcessingResult(BaseModel):
    """Результат обработки документа"""
    document_id: str
    processing_date: str
    file_path: str
    file_type: str
    template: Optional[str] = None
    quality_report: QualityReport
    extracted_data: ExtractedData
    corrections_applied: List[Dict[str, Any]]
    needs_review: bool
    new_corrections_suggested: List[Dict[str, Any]] = []


class BatchProcessingResult(BaseModel):
    """Результат пакетной обработки"""
    documents: List[ProcessingResult]
    new_corrections_suggested: List[Dict[str, Any]]
    needs_human_review: bool
    total_processed: int
    successful: int


class CorrectionsDBResponse(BaseModel):
    """Ответ с базой исправлений"""
    corrections: Dict[str, str]
    total_count: int
    last_updated: Optional[str] = None
