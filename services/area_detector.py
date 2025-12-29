"""Детектор областей документа (базовая версия)"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AreaDetector:
    """Класс для детекции областей документа"""
    
    def __init__(self):
        """Инициализация детектора"""
        pass
    
    def detect_areas(self, image: np.ndarray, template: Optional[str] = None) -> List[Dict[str, int]]:
        """
        Детекция областей документа
        
        Args:
            image: Изображение
            template: Тип шаблона (опционально)
            
        Returns:
            Список областей с координатами
        """
        # Базовая реализация: возвращает весь документ как одну область
        # В Фазе 2 здесь будет ML модель для детекции областей
        
        height, width = image.shape[:2]
        
        # Можно добавить простую детекцию на основе контуров
        areas = self._detect_by_contours(image)
        
        if not areas:
            # Если не найдено областей, возвращаем весь документ
            areas = [{
                'x1': 0,
                'y1': 0,
                'x2': width,
                'y2': height,
                'type': 'full_document'
            }]
        
        return areas
    
    def _detect_by_contours(self, image: np.ndarray) -> List[Dict[str, int]]:
        """
        Детекция областей на основе контуров
        
        Args:
            image: Изображение
            
        Returns:
            Список областей
        """
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Поиск контуров
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        areas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Фильтрация маленьких областей
            if w > 50 and h > 50:
                areas.append({
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h,
                    'type': 'text_block'
                })
        
        return areas
