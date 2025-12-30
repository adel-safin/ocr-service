"""Основной пайплайн обработки документов"""
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.ocr_engine import OCREngine
from core.validators import FieldValidator
from core.correctors import AutoCorrectionSystem
from services.quality_check import QualityChecker
from config.settings import settings

logger = logging.getLogger(__name__)

# Импорт ML компонентов (опционально, если доступны)
try:
    import torch
    from models.document_classifier import DocumentClassifier
    from models.spell_corrector import SpellCorrector, SimpleSpellCorrector
    from services.ml_quality_check import MLQualityChecker
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    torch = None
    logger.warning(f"ML компоненты недоступны, используется базовый режим: {str(e)}")


class DocumentPipeline:
    """Основной пайплайн обработки документов"""
    
    def __init__(self, use_ml: bool = True, use_active_learning: bool = True):
        """
        Инициализация пайплайна
        
        Args:
            use_ml: Использовать ML компоненты (если доступны)
            use_active_learning: Использовать активное обучение
        """
        self.ocr_engine = OCREngine()
        self.validator = FieldValidator()
        self.corrector = AutoCorrectionSystem()
        self.quality_checker = QualityChecker()
        
        # Инициализация активного обучения
        self.use_active_learning = use_active_learning
        self.active_learning = None
        if self.use_active_learning:
            try:
                from services.active_learning import ActiveLearningSystem
                self.active_learning = ActiveLearningSystem()
                logger.info("Система активного обучения инициализирована")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать активное обучение: {str(e)}")
                self.use_active_learning = False
        
        # Инициализация ML компонентов
        self.use_ml = use_ml and ML_AVAILABLE
        self.document_classifier = None
        self.spell_corrector = None
        self.ml_quality_checker = None
        
        if self.use_ml:
            try:
                # Загрузка классификатора документов (если обучен)
                self._load_classifier()
                
                # Инициализация исправителя опечаток
                device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
                # Попытка использовать локальную модель
                from pathlib import Path
                app_root = Path(__file__).parent.parent
                local_rut5 = app_root / "rut5-base"
                local_path = str(local_rut5) if local_rut5.exists() else None
                
                self.spell_corrector = SpellCorrector(device=device, local_path=local_path)
                if self.spell_corrector.model is None:
                    # Fallback на простой исправитель
                    self.spell_corrector = SimpleSpellCorrector()
                
                # Инициализация ML проверки качества
                self.ml_quality_checker = MLQualityChecker(device=device)
                
                logger.info("ML компоненты инициализированы")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать ML компоненты: {str(e)}")
                self.use_ml = False
    
    def process(self, file_path: str, template: Optional[str] = None, 
                required_fields: Optional[List[str]] = None,
                selected_areas: Optional[List[Dict[str, int]]] = None) -> Dict[str, Any]:
        """
        Полная обработка документа
        
        Args:
            file_path: Путь к файлу
            template: Тип шаблона документа (опционально)
            required_fields: Список обязательных полей для валидации
            
        Returns:
            Результаты обработки
        """
        document_id = f"{os.path.basename(file_path)}_{uuid.uuid4().hex[:8]}"
        
        try:
            # 1. Обработка всех страниц PDF отдельно
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Обработка выделенных областей с повышенным DPI (900)
            selected_areas_text = ""
            selected_areas_data = []  # Список данных о каждой выделенной области
            if selected_areas and len(selected_areas) > 0:
                logger.info(f"Обработка {len(selected_areas)} выделенных областей с DPI 900")
                try:
                    # Загружаем изображение с высоким DPI для выделенных областей
                    if file_ext == '.pdf':
                        # Для PDF обрабатываем первую страницу (можно расширить для всех страниц)
                        high_dpi_image, _ = self.ocr_engine.load_image(file_path, page=1, dpi=900)
                    else:
                        # Для изображений загружаем с высоким DPI (масштабируем)
                        high_dpi_image, _ = self.ocr_engine.load_image(file_path, dpi=900)
                    
                    # Координаты приходят для изображения с DPI 300 (реальные размеры изображения)
                    # Нужно масштабировать их для изображения с DPI 900
                    if file_ext == '.pdf':
                        # Для PDF получаем размеры первой страницы с DPI 300
                        normal_dpi_image, _ = self.ocr_engine.load_image(file_path, page=1, dpi=300)
                    else:
                        normal_dpi_image, _ = self.ocr_engine.load_image(file_path, dpi=300)
                    
                    # Вычисляем масштаб на основе реальных размеров изображений
                    normal_height, normal_width = normal_dpi_image.shape[:2]
                    high_height, high_width = high_dpi_image.shape[:2]
                    
                    # Масштаб: DPI 900 в 3 раза больше DPI 300 (900/300 = 3.0)
                    actual_scale_x = high_width / normal_width if normal_width > 0 else 3.0
                    actual_scale_y = high_height / normal_height if normal_height > 0 else 3.0
                    
                    logger.info(f"Масштабирование координат: DPI 300 размер {normal_width}x{normal_height}, DPI 900 размер {high_width}x{high_height}, масштаб {actual_scale_x:.2f}x{actual_scale_y:.2f}")
                    
                    # Обрабатываем каждую выделенную область
                    area_texts = []
                    for i, area in enumerate(selected_areas):
                        try:
                            # Координаты уже в размерах изображения с DPI 300, масштабируем для DPI 900
                            original_x1 = area.get('x1', 0)
                            original_y1 = area.get('y1', 0)
                            original_x2 = area.get('x2', 0)
                            original_y2 = area.get('y2', 0)
                            
                            scaled_area = {
                                'x1': int(original_x1 * actual_scale_x),
                                'y1': int(original_y1 * actual_scale_y),
                                'x2': int(original_x2 * actual_scale_x),
                                'y2': int(original_y2 * actual_scale_y)
                            }
                            
                            logger.info(f"Область {i+1}: исходные координаты (DPI 300) x1={original_x1}, y1={original_y1}, x2={original_x2}, y2={original_y2}")
                            logger.info(f"Область {i+1}: масштабированные координаты (DPI 900) x1={scaled_area['x1']}, y1={scaled_area['y1']}, x2={scaled_area['x2']}, y2={scaled_area['y2']}")
                            
                            area_result = self.ocr_engine.extract_text_by_area(high_dpi_image, scaled_area, dpi=900)
                            if area_result.get('text'):
                                area_text = area_result['text']
                                area_texts.append(f"[Область {i+1}]: {area_text}")
                                
                                # Сохраняем данные о каждой области отдельно (исходные координаты)
                                selected_areas_data.append({
                                    'area_number': i + 1,
                                    'coordinates': {
                                        'x1': area.get('x1', 0),
                                        'y1': area.get('y1', 0),
                                        'x2': area.get('x2', 0),
                                        'y2': area.get('y2', 0)
                                    },
                                    'text': area_text,
                                    'confidence': area_result.get('confidence', 0.0),
                                    'word_count': area_result.get('word_count', 0),
                                    'dpi': 900
                                })
                                
                                logger.info(f"Область {i+1}: распознано {len(area_text)} символов, уверенность {area_result.get('confidence', 0):.2%}")
                        except Exception as area_e:
                            logger.error(f"Ошибка при обработке области {i+1}: {str(area_e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                            continue
                    
                    if area_texts:
                        selected_areas_text = "\n\n--- ВЫДЕЛЕННЫЕ ОБЛАСТИ (DPI 900) ---\n\n" + "\n\n".join(area_texts)
                except Exception as e:
                    logger.error(f"Ошибка при обработке выделенных областей: {str(e)}")
                    # Продолжаем обработку без выделенных областей
            
            if file_ext == '.pdf':
                # Обрабатываем все страницы
                page_results = self.ocr_engine.process_file_all_pages(file_path)
                
                # Объединяем тексты со всех страниц
                all_texts = []
                all_confidences = []
                pages_data = []
                
                for page_result in page_results:
                    all_texts.append(page_result['text'])
                    all_confidences.append(page_result['confidence'])
                    pages_data.append({
                        'page_number': page_result.get('page_number', 1),
                        'text': page_result['text'],
                        'confidence': page_result['confidence'],
                        'word_count': page_result.get('word_count', 0)
                    })
                
                raw_text = '\n\n--- Страница ---\n\n'.join(all_texts)
                
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
                
                ocr_result = {
                    'text': raw_text,
                    'confidence': avg_confidence,
                    'detailed_data': {},
                    'word_count': sum(r.get('word_count', 0) for r in page_results),
                    'pages': pages_data,
                    'total_pages': len(page_results)
                }
                file_type = 'pdf'
            else:
                # Для не-PDF файлов обрабатываем как обычно
                image, file_type = self.ocr_engine.load_image(file_path)
                ocr_result = self.ocr_engine.extract_text(image)
                raw_text = ocr_result['text']
                raw_text = ocr_result['text']
                ocr_result['pages'] = [{
                    'page_number': 1,
                    'text': raw_text,
                    'confidence': ocr_result.get('confidence', 0.0),
                    'word_count': ocr_result.get('word_count', 0)
                }]
                ocr_result['total_pages'] = 1
            
            # 2.5. Добавляем текст из выделенных областей в начало
            if selected_areas_text:
                raw_text = selected_areas_text + "\n\n--- ОСНОВНОЙ ТЕКСТ ---\n\n" + raw_text
            
            # 3. Автокоррекция (базовая)
            corrected_text, corrections_applied = self.corrector.correct_text(raw_text)
            
            # 3.5. ML исправление опечаток (если доступно)
            # ВАЖНО: ML исправление отключено по умолчанию, так как T5 модель
            # может портить текст при обработке OCR ошибок
            # Используйте только для коротких, хорошо распознанных фрагментов
            use_ml_correction = False  # Отключено по умолчанию
            
            if use_ml_correction and self.use_ml and self.spell_corrector:
                try:
                    # Применяем ML исправление только к коротким фрагментам
                    # или отключено полностью для безопасности
                    pass
                    # ml_corrected = self.spell_corrector.correct_text(corrected_text)
                    # if ml_corrected != corrected_text and '<extra_id' not in ml_corrected:
                    #     corrections_applied.append({
                    #         'from': corrected_text[:50] + '...' if len(corrected_text) > 50 else corrected_text,
                    #         'to': ml_corrected[:50] + '...' if len(ml_corrected) > 50 else ml_corrected,
                    #         'confidence': 0.8,
                    #         'method': 'ml_transformer'
                    #     })
                    #     corrected_text = ml_corrected
                except Exception as e:
                    logger.warning(f"Ошибка ML исправления: {str(e)}")
            
            # 4. Валидация критических полей
            validation_results = self.validator.validate_critical_fields(
                corrected_text, 
                required_fields
            )
            
            # 5. Проверка качества
            # Используем базовую проверку качества, так как ML модель не обучена
            # ML проверку можно включить после обучения модели
            use_ml_quality = False  # Отключено до обучения модели
            
            # Загружаем изображение для проверки качества (первая страница для PDF)
            if file_ext == '.pdf' and 'pages' in ocr_result and len(ocr_result['pages']) > 0:
                # Для PDF используем первую страницу
                quality_image, _ = self.ocr_engine.load_image(file_path, page=1)
            else:
                # Для не-PDF файлов
                if file_ext != '.pdf':
                    quality_image, _ = self.ocr_engine.load_image(file_path)
                else:
                    # Fallback для PDF
                    quality_image, _ = self.ocr_engine.load_image(file_path, page=1)
            
            if use_ml_quality and self.use_ml and self.ml_quality_checker:
                try:
                    quality_report = self.ml_quality_checker.check_quality_ml(
                        quality_image,
                        ocr_result,
                        corrected_text
                    )
                except Exception as e:
                    logger.warning(f"Ошибка ML проверки качества: {str(e)}")
                    quality_report = self.quality_checker.check_quality(
                        quality_image, 
                        ocr_result, 
                        corrected_text
                    )
            else:
                quality_report = self.quality_checker.check_quality(
                    quality_image, 
                    ocr_result, 
                    corrected_text
                )
            
            # Добавляем информацию о координатах областей из OCR результата
            if 'detailed_data' in ocr_result and 'text_regions' in ocr_result['detailed_data']:
                quality_report['text_regions'] = ocr_result['detailed_data']['text_regions']
            
            # Добавляем информацию о страницах для многостраничных документов
            if 'pages' in ocr_result:
                quality_report['pages_info'] = ocr_result['pages']
            
            # 5.5. Автоматическое определение типа документа (если не указан)
            if not template and self.use_ml and self.document_classifier:
                try:
                    device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
                    predicted_idx, confidence = self.document_classifier.predict(file_path, device=device)
                    if confidence > 0.7 and hasattr(self, 'class_mapping'):
                        predicted_type = self.class_mapping['idx_to_class'].get(predicted_idx, 'unknown')
                        template = predicted_type
                        logger.info(f"Автоматически определен тип документа: {template} (уверенность: {confidence:.2f})")
                except Exception as e:
                    logger.warning(f"Ошибка классификации документа: {str(e)}")
            
            # 5.6. Извлечение важных данных
            important_data = self.validator.extract_important_data(corrected_text)
            
            # 6. Формирование результата
            result = {
                'document_id': document_id,
                'processing_date': datetime.now().isoformat(),
                'file_path': file_path,
                'file_type': file_type,
                'template': template,
                'quality_report': quality_report,
                'important_data': important_data,
                'extracted_data': {
                    'critical_fields': {
                        field: {
                            'value': result.value,
                            'confidence': result.confidence,
                            'valid': result.valid,
                            'message': result.message,
                            'suggested_correction': result.suggested_correction
                        }
                        for field, result in validation_results.items()
                    },
                    'full_text': corrected_text,
                    'raw_text': raw_text,
                    'pages': ocr_result.get('pages', []),
                    'total_pages': ocr_result.get('total_pages', 1),
                    'selected_areas': selected_areas_data  # Добавляем данные о выделенных областях
                },
                'total_pages': ocr_result.get('total_pages', 1),
                'corrections_applied': corrections_applied,
                'needs_review': quality_report.get('needs_review', False) or 
                               any(not r.valid for r in validation_results.values()),
                'new_corrections_suggested': []
            }
            
            # 6.5. Сбор данных для активного обучения (если включено)
            if self.use_active_learning and self.active_learning:
                try:
                    # Автоматический сбор данных о примененных исправлениях
                    for correction in corrections_applied:
                        if correction.get('method') != 'ml_transformer':  # Не дублируем ML исправления
                            self.active_learning.feedback_collector.add_correction_feedback(
                                original=correction.get('from', ''),
                                corrected=correction.get('to', ''),
                                document_id=document_id,
                                context=corrected_text[:200] if len(corrected_text) > 200 else corrected_text,
                                confidence=correction.get('confidence', 0.8)
                            )
                except Exception as e:
                    logger.warning(f"Ошибка при сборе данных для активного обучения: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при обработке документа {file_path}: {str(e)}")
            raise
    
    def batch_process(self, file_paths: List[str], 
                     template: Optional[str] = None) -> Dict[str, Any]:
        """
        Пакетная обработка документов
        
        Args:
            file_paths: Список путей к файлам
            template: Тип шаблона (опционально)
            
        Returns:
            Результаты обработки всех документов
        """
        results = []
        unknown_errors = []
        
        for file_path in file_paths:
            try:
                processed = self.process(file_path, template)
                
                # Поиск потенциальных ошибок
                if processed.get('needs_review'):
                    # Сбор примеров для обучения
                    for correction in processed.get('corrections_applied', []):
                        if correction.get('confidence', 0) < 0.8:
                            unknown_errors.append({
                                'file': file_path,
                                'correction': correction
                            })
                
                results.append(processed)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке {file_path}: {str(e)}")
                results.append({
                    'document_id': f"error_{uuid.uuid4().hex[:8]}",
                    'file_path': file_path,
                    'error': str(e),
                    'success': False
                })
        
        # Предложение новых автозамен
        new_suggestions = []
        if unknown_errors:
            # Группировка похожих ошибок
            error_groups = {}
            for error in unknown_errors:
                correction = error['correction']
                key = correction.get('from', '')
                if key not in error_groups:
                    error_groups[key] = []
                error_groups[key].append(correction.get('to', ''))
            
            # Предложения на основе частоты
            for original, corrections in error_groups.items():
                if len(set(corrections)) == 1:  # Все исправления одинаковые
                    new_suggestions.append({
                        'from': original,
                        'to': corrections[0],
                        'confidence': 0.7,
                        'occurrences': len(corrections)
                    })
        
        return {
            'documents': results,
            'new_corrections_suggested': new_suggestions,
            'needs_human_review': len(new_suggestions) > 0,
            'total_processed': len(results),
            'successful': len([r for r in results if r.get('success', True)])
        }
    
    def _load_classifier(self):
        """Загрузка обученного классификатора документов"""
        try:
            import pickle
            import torch
            
            classifier_path = Path('models/document_classifier')
            model_path = classifier_path / 'best_model_epoch_*.pth'
            
            # Поиск последней обученной модели
            model_files = list(classifier_path.glob('best_model_epoch_*.pth'))
            if not model_files:
                logger.info("Обученная модель классификатора не найдена")
                return
            
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            
            # Загрузка словаря классов
            mapping_path = classifier_path / 'class_mapping.pkl'
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    mapping = pickle.load(f)
                    self.class_mapping = mapping
                    num_classes = len(mapping['class_to_idx'])
            else:
                logger.warning("Словарь классов не найден")
                return
            
            # Загрузка модели
            self.document_classifier = DocumentClassifier(num_classes=num_classes, pretrained=False)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.document_classifier.load_state_dict(torch.load(latest_model, map_location=device))
            self.document_classifier.eval()
            
            logger.info(f"Классификатор документов загружен: {latest_model}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить классификатор: {str(e)}")
    
    def format_output(self, result: Dict[str, Any], format: str = 'json') -> str:
        """
        Форматирование результата
        
        Args:
            result: Результат обработки
            format: Формат вывода ('json', 'text')
            
        Returns:
            Отформатированная строка
        """
        if format == 'json':
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        elif format == 'text':
            lines = [
                f"Document ID: {result.get('document_id')}",
                f"Processing Date: {result.get('processing_date')}",
                f"Quality: {result.get('quality_report', {}).get('overall_quality', 0):.2f}",
                "\nCritical Fields:",
            ]
            for field, data in result.get('extracted_data', {}).get('critical_fields', {}).items():
                lines.append(f"  {field}: {data.get('value')} (valid: {data.get('valid')})")
            return "\n".join(lines)
        else:
            raise ValueError(f"Неизвестный формат: {format}")
