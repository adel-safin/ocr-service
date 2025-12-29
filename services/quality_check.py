"""Проверка качества распознавания"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QualityChecker:
    """Класс для проверки качества изображения и распознавания"""
    
    def __init__(self):
        """Инициализация проверки качества"""
        pass
    
    def check_image_quality(self, image: np.ndarray) -> Dict[str, any]:
        """
        Проверка качества изображения
        
        Args:
            image: Изображение для проверки
            
        Returns:
            Словарь с метриками качества
        """
        # Конвертация в grayscale если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Резкость (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 100.0, 1.0)  # Нормализация
        
        # Контраст
        contrast = gray.std()
        contrast_score = min(contrast / 50.0, 1.0)  # Нормализация
        
        # Яркость
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # Идеальная яркость ~127
        
        # Общее качество
        overall_quality = (sharpness_score + contrast_score + brightness_score) / 3.0
        
        return {
            'sharpness': sharpness_score,
            'contrast': contrast_score,
            'brightness': brightness_score,
            'overall_quality': overall_quality,
            'issues': self._detect_issues(sharpness_score, contrast_score, brightness_score)
        }
    
    def _detect_issues(self, sharpness: float, contrast: float, brightness: float) -> List[Dict[str, any]]:
        """
        Обнаружение проблем с качеством
        
        Args:
            sharpness: Оценка резкости
            contrast: Оценка контраста
            brightness: Оценка яркости
            
        Returns:
            Список обнаруженных проблем
        """
        issues = []
        
        if sharpness < 0.5:
            issues.append({
                'type': 'blur',
                'severity': 'high' if sharpness < 0.3 else 'medium',
                'message': 'Изображение размыто, может снизить точность OCR'
            })
        
        if contrast < 0.5:
            issues.append({
                'type': 'low_contrast',
                'severity': 'medium',
                'message': 'Низкий контраст изображения'
            })
        
        if brightness < 0.5:
            issues.append({
                'type': 'brightness',
                'severity': 'medium',
                'message': 'Неоптимальная яркость изображения'
            })
        
        return issues
    
    def detect_handwritten_text(self, image: np.ndarray, ocr_data: Dict) -> List[Dict[str, any]]:
        """
        Детекция рукописного текста (базовая версия)
        
        Args:
            image: Изображение
            ocr_data: Данные OCR с координатами
            
        Returns:
            Список областей с рукописным текстом
        """
        handwritten_areas = []
        
        # Базовая эвристика: проверка регулярности символов
        # В реальной реализации здесь должна быть ML модель
        
        detailed_data = ocr_data.get('detailed_data', {})
        if not detailed_data:
            return handwritten_areas
        
        # Проверяем новые данные с text_regions
        text_regions = detailed_data.get('text_regions', [])
        
        if text_regions:
            # Используем новые данные с координатами
            for region in text_regions:
                text = region.get('text', '')
                confidence = region.get('confidence', 0.0)
                bbox = region.get('bbox', {})
                
                if text.strip() and confidence < 0.5:  # Низкая уверенность
                    handwritten_areas.append({
                        'type': 'handwritten_text',
                        'area': {
                            'x1': bbox.get('x1', 0),
                            'y1': bbox.get('y1', 0),
                            'x2': bbox.get('x2', 0),
                            'y2': bbox.get('y2', 0),
                            'width': bbox.get('width', 0),
                            'height': bbox.get('height', 0)
                        },
                        'message': 'Рукописный текст может быть распознан неточно - перепроверьте',
                        'confidence': confidence,
                        'text': text,
                        'page_number': ocr_data.get('page_number', 1)
                    })
        else:
            # Старый формат для обратной совместимости
            confidences = detailed_data.get('conf', [])
            texts = detailed_data.get('text', [])
            
            for i, (text, conf) in enumerate(zip(texts, confidences)):
                if int(conf) > 0 and text.strip():
                    # Низкая уверенность может указывать на рукописный текст
                    if int(conf) < 50:
                        # Получение координат
                        left = detailed_data.get('left', [0])[i] if i < len(detailed_data.get('left', [])) else 0
                        top = detailed_data.get('top', [0])[i] if i < len(detailed_data.get('top', [])) else 0
                        width = detailed_data.get('width', [0])[i] if i < len(detailed_data.get('width', [])) else 0
                        height = detailed_data.get('height', [0])[i] if i < len(detailed_data.get('height', [])) else 0
                        
                        handwritten_areas.append({
                            'type': 'handwritten_text',
                            'area': {
                                'x1': left,
                                'y1': top,
                                'x2': left + width,
                                'y2': top + height,
                                'width': width,
                                'height': height
                            },
                            'message': 'Рукописный текст может быть распознан неточно - перепроверьте',
                            'confidence': int(conf) / 100.0,
                            'text': text,
                            'page_number': ocr_data.get('page_number', 1)
                        })
        
        return handwritten_areas
    
    def check_quality(self, image: np.ndarray, ocr_result: Dict, corrected_text: str) -> Dict[str, any]:
        """
        Комплексная проверка качества
        
        Args:
            image: Изображение
            ocr_result: Результаты OCR
            corrected_text: Исправленный текст
            
        Returns:
            Отчет о качестве
        """
        # Проверка качества изображения
        image_quality = self.check_image_quality(image)
        
        # Проверка уверенности OCR
        ocr_confidence = ocr_result.get('confidence', 0.0)
        
        # Детекция рукописного текста (всегда включена)
        handwritten_areas = self.detect_handwritten_text(image, ocr_result)
        
        # Общая оценка качества
        overall_quality = (
            image_quality['overall_quality'] * 0.3 +
            ocr_confidence * 0.5 +
            (1.0 - len(handwritten_areas) * 0.1) * 0.2
        )
        overall_quality = max(0.0, min(1.0, overall_quality))
        
        # Сбор всех проблем
        all_issues = image_quality.get('issues', []) + handwritten_areas
        
        # Формирование предупреждений
        warnings = []
        if image_quality['overall_quality'] < 0.7:
            warnings.append('Уведомление о качестве исходной документации: низкое качество изображения может не позволить на 100% распознать текст')
        if handwritten_areas:
            warnings.append(f'Обнаружено {len(handwritten_areas)} областей с возможным рукописным текстом - перепроверьте')
        if any(issue.get('type') == 'text_overlay' for issue in all_issues):
            warnings.append('Печатный текст из-за наслоения может распознаться некорректно')
        
        return {
            'overall_quality': overall_quality,
            'image_quality': image_quality,
            'ocr_confidence': ocr_confidence,
            'text_quality': ocr_confidence,  # Для совместимости
            'issues': all_issues,
            'warnings': warnings,
            'handwritten_areas': handwritten_areas,
            'text_regions': ocr_result.get('detailed_data', {}).get('text_regions', []),
            'needs_review': overall_quality < 0.7 or len(all_issues) > 0
        }
