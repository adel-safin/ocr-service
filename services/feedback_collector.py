"""Система сбора обратной связи от пользователей"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Класс для сбора и хранения обратной связи"""
    
    def __init__(self, feedback_db_path: Optional[str] = None):
        """
        Инициализация сборщика обратной связи
        
        Args:
            feedback_db_path: Путь к файлу базы данных feedback
        """
        if feedback_db_path is None:
            feedback_db_path = os.path.join(settings.DATA_DIR, "feedback.json")
        
        self.feedback_db_path = Path(feedback_db_path)
        self.feedback_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Загрузка существующих данных
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self) -> Dict[str, Any]:
        """Загрузка данных обратной связи"""
        if not self.feedback_db_path.exists():
            return {
                'corrections': [],
                'document_types': [],
                'quality_ratings': [],
                'statistics': {
                    'total_feedback': 0,
                    'corrections_count': 0,
                    'last_updated': None
                }
            }
        
        try:
            with open(self.feedback_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке feedback: {str(e)}")
            return {
                'corrections': [],
                'document_types': [],
                'quality_ratings': [],
                'statistics': {
                    'total_feedback': 0,
                    'corrections_count': 0,
                    'last_updated': None
                }
            }
    
    def _save_feedback(self):
        """Сохранение данных обратной связи"""
        try:
            self.feedback_data['statistics']['last_updated'] = datetime.now().isoformat()
            with open(self.feedback_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении feedback: {str(e)}")
    
    def add_correction_feedback(self, original: str, corrected: str, 
                               document_id: str, context: Optional[str] = None,
                               user_id: Optional[str] = None,
                               confidence: float = 1.0) -> str:
        """
        Добавление обратной связи по исправлению
        
        Args:
            original: Оригинальный (ошибочный) текст
            corrected: Исправленный текст
            document_id: ID документа
            context: Контекст исправления
            user_id: ID пользователя
            confidence: Уверенность пользователя (0.0-1.0)
            
        Returns:
            ID записи feedback
        """
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data['corrections'])}"
        
        feedback_entry = {
            'id': feedback_id,
            'type': 'correction',
            'original': original,
            'corrected': corrected,
            'document_id': document_id,
            'context': context,
            'user_id': user_id,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'applied': False
        }
        
        self.feedback_data['corrections'].append(feedback_entry)
        self.feedback_data['statistics']['total_feedback'] += 1
        self.feedback_data['statistics']['corrections_count'] += 1
        
        self._save_feedback()
        logger.info(f"Добавлен feedback по исправлению: {feedback_id}")
        
        return feedback_id
    
    def add_quality_rating(self, document_id: str, rating: float,
                          issues: Optional[List[str]] = None,
                          user_id: Optional[str] = None) -> str:
        """
        Добавление оценки качества обработки
        
        Args:
            document_id: ID документа
            rating: Оценка качества (0.0-1.0)
            issues: Список проблем
            user_id: ID пользователя
            
        Returns:
            ID записи feedback
        """
        feedback_id = f"quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data['quality_ratings'])}"
        
        feedback_entry = {
            'id': feedback_id,
            'type': 'quality',
            'document_id': document_id,
            'rating': rating,
            'issues': issues or [],
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_data['quality_ratings'].append(feedback_entry)
        self.feedback_data['statistics']['total_feedback'] += 1
        
        self._save_feedback()
        logger.info(f"Добавлена оценка качества: {feedback_id}")
        
        return feedback_id
    
    def add_document_type_feedback(self, document_id: str, 
                                  predicted_type: str, actual_type: str,
                                  user_id: Optional[str] = None) -> str:
        """
        Добавление обратной связи по типу документа
        
        Args:
            document_id: ID документа
            predicted_type: Предсказанный тип
            actual_type: Реальный тип
            user_id: ID пользователя
            
        Returns:
            ID записи feedback
        """
        feedback_id = f"doctype_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_data['document_types'])}"
        
        feedback_entry = {
            'id': feedback_id,
            'type': 'document_type',
            'document_id': document_id,
            'predicted_type': predicted_type,
            'actual_type': actual_type,
            'correct': predicted_type == actual_type,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_data['document_types'].append(feedback_entry)
        self.feedback_data['statistics']['total_feedback'] += 1
        
        self._save_feedback()
        logger.info(f"Добавлен feedback по типу документа: {feedback_id}")
        
        return feedback_id
    
    def get_unapplied_corrections(self, min_confidence: float = 0.7,
                                 min_occurrences: int = 2) -> List[Dict]:
        """
        Получение не примененных исправлений для автоматического добавления
        
        Args:
            min_confidence: Минимальная уверенность
            min_occurrences: Минимальное количество вхождений
            
        Returns:
            Список исправлений для применения
        """
        # Группировка по парам (original, corrected)
        correction_groups = {}
        
        for correction in self.feedback_data['corrections']:
            if correction.get('applied', False):
                continue
            
            key = (correction['original'], correction['corrected'])
            if key not in correction_groups:
                correction_groups[key] = {
                    'original': correction['original'],
                    'corrected': correction['corrected'],
                    'occurrences': [],
                    'total_confidence': 0.0
                }
            
            correction_groups[key]['occurrences'].append(correction)
            correction_groups[key]['total_confidence'] += correction.get('confidence', 1.0)
        
        # Фильтрация по критериям
        candidates = []
        for key, group in correction_groups.items():
            count = len(group['occurrences'])
            avg_confidence = group['total_confidence'] / count if count > 0 else 0.0
            
            if count >= min_occurrences and avg_confidence >= min_confidence:
                candidates.append({
                    'original': group['original'],
                    'corrected': group['corrected'],
                    'occurrences': count,
                    'avg_confidence': avg_confidence,
                    'feedback_ids': [c['id'] for c in group['occurrences']]
                })
        
        # Сортировка по количеству вхождений и уверенности
        candidates.sort(key=lambda x: (x['occurrences'], x['avg_confidence']), reverse=True)
        
        return candidates
    
    def mark_corrections_applied(self, feedback_ids: List[str]):
        """
        Отметка исправлений как примененных
        
        Args:
            feedback_ids: Список ID feedback записей
        """
        for correction in self.feedback_data['corrections']:
            if correction['id'] in feedback_ids:
                correction['applied'] = True
                correction['applied_at'] = datetime.now().isoformat()
        
        self._save_feedback()
        logger.info(f"Отмечено {len(feedback_ids)} исправлений как примененных")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по feedback"""
        stats = self.feedback_data.get('statistics', {})
        
        # Дополнительная статистика
        total_corrections = len(self.feedback_data.get('corrections', []))
        applied_corrections = sum(1 for c in self.feedback_data.get('corrections', []) 
                                 if c.get('applied', False))
        
        return {
            **stats,
            'total_corrections': total_corrections,
            'applied_corrections': applied_corrections,
            'pending_corrections': total_corrections - applied_corrections,
            'quality_ratings_count': len(self.feedback_data.get('quality_ratings', [])),
            'document_types_feedback_count': len(self.feedback_data.get('document_types', []))
        }
