"""ML-улучшенная проверка качества изображения"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MLQualityChecker:
    """ML модель для оценки качества изображения"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Инициализация ML проверки качества
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device
        self.model = self._create_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    def _create_model(self) -> nn.Module:
        """Создание модели для оценки качества"""
        # Использование предобученной ResNet18
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Замена последнего слоя для регрессии (оценка качества 0-1)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Выход в диапазоне [0, 1]
        )
        
        return model.to(self.device)
    
    def predict_quality(self, image: np.ndarray) -> float:
        """
        Предсказание качества изображения
        
        Args:
            image: Изображение (numpy array)
            
        Returns:
            Оценка качества (0.0 - 1.0)
        """
        try:
            # Конвертация в PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Трансформация
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Предсказание
            self.model.eval()
            with torch.no_grad():
                quality_score = self.model(image_tensor).item()
            
            return quality_score
        
        except Exception as e:
            logger.error(f"Ошибка при оценке качества: {str(e)}")
            return 0.5  # Среднее значение при ошибке
    
    def detect_handwritten_regions(self, image: np.ndarray, ocr_data: Dict) -> List[Dict]:
        """
        Детекция рукописных областей с помощью ML
        
        Args:
            image: Изображение
            ocr_data: Данные OCR с координатами
            
        Returns:
            Список областей с рукописным текстом
        """
        handwritten_regions = []
        
        # Базовая реализация: анализ уверенности OCR
        # В полной версии здесь должна быть отдельная ML модель
        
        detailed_data = ocr_data.get('detailed_data', {})
        if not detailed_data:
            return handwritten_regions
        
        confidences = detailed_data.get('conf', [])
        texts = detailed_data.get('text', [])
        lefts = detailed_data.get('left', [])
        tops = detailed_data.get('top', [])
        widths = detailed_data.get('width', [])
        heights = detailed_data.get('height', [])
        
        for i, (text, conf) in enumerate(zip(texts, confidences)):
            if int(conf) > 0 and text.strip():
                # Низкая уверенность + короткие слова могут быть рукописными
                if int(conf) < 60 and len(text.strip()) < 10:
                    if i < len(lefts) and i < len(tops):
                        handwritten_regions.append({
                            'type': 'handwritten_text',
                            'area': {
                                'x1': lefts[i],
                                'y1': tops[i],
                                'x2': lefts[i] + widths[i] if i < len(widths) else lefts[i] + 50,
                                'y2': tops[i] + heights[i] if i < len(heights) else tops[i] + 20
                            },
                            'message': 'Возможный рукописный текст',
                            'confidence': int(conf) / 100.0,
                            'text': text
                        })
        
        return handwritten_regions
    
    def check_quality_ml(self, image: np.ndarray, ocr_result: Dict, 
                        corrected_text: str) -> Dict:
        """
        Комплексная ML проверка качества
        
        Args:
            image: Изображение
            ocr_result: Результаты OCR
            corrected_text: Исправленный текст
            
        Returns:
            Отчет о качестве с ML оценками
        """
        # ML оценка качества изображения
        ml_quality_score = self.predict_quality(image)
        
        # Детекция рукописных областей
        handwritten_regions = self.detect_handwritten_regions(image, ocr_result)
        
        # OCR уверенность
        ocr_confidence = ocr_result.get('confidence', 0.0)
        
        # Комбинированная оценка
        # ML оценка может быть неточной для необученной модели,
        # поэтому больше веса даем OCR уверенности
        overall_quality = (
            ml_quality_score * 0.2 +  # Меньший вес для ML (модель не обучена)
            ocr_confidence * 0.6 +     # Больший вес для OCR уверенности
            (1.0 - len(handwritten_regions) * 0.05) * 0.2
        )
        overall_quality = max(0.0, min(1.0, overall_quality))
        
        # Если ML оценка сильно отличается от OCR уверенности,
        # используем более консервативный подход
        if abs(ml_quality_score - ocr_confidence) > 0.3:
            logger.debug(f"Большая разница между ML ({ml_quality_score:.2f}) и OCR ({ocr_confidence:.2f}) оценками")
            # Используем среднее между базовой оценкой и OCR уверенностью
            overall_quality = (ocr_confidence * 0.8 + ml_quality_score * 0.2)
        
        return {
            'overall_quality': overall_quality,
            'ml_quality_score': ml_quality_score,
            'ocr_confidence': ocr_confidence,
            'handwritten_regions': handwritten_regions,
            'needs_review': overall_quality < 0.7 or len(handwritten_regions) > 3,
            'issues': [
                {
                    'type': 'low_ml_quality',
                    'message': f'ML оценка качества низкая: {ml_quality_score:.2f}',
                    'severity': 'medium' if ml_quality_score < 0.6 else 'low'
                }
            ] if ml_quality_score < 0.6 else []
        }
