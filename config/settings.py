"""Конфигурация приложения"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Настройки приложения"""
    
    # API настройки
    API_TITLE: str = "OCR Document Processing Service"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # OCR настройки (Vision Framework)
    VISION_RECOGNITION_LEVEL: str = "accurate"  # "accurate" или "fast"
    VISION_LANGUAGE_CORRECTION: bool = True  # Использовать языковую коррекцию
    
    # Пути к данным
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    TEMPLATES_DIR: str = os.path.join(DATA_DIR, "templates")
    CORRECTIONS_DB: str = os.path.join(DATA_DIR, "corrections.json")
    OUTPUTS_DIR: str = os.path.join(DATA_DIR, "outputs")
    
    # Настройки валидации
    CONFIDENCE_THRESHOLD: float = 0.7
    MIN_FIELD_CONFIDENCE: float = 0.6
    
    # Настройки автокоррекции
    SIMILARITY_THRESHOLD: float = 0.8
    MAX_CORRECTION_DISTANCE: int = 2
    
    # Настройки качества
    QUALITY_CHECK_ENABLED: bool = True
    HANDWRITTEN_DETECTION_ENABLED: bool = True
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
