"""OCR движок на основе Apple Vision Framework"""
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
from typing import Dict, List, Tuple, Optional
import os
import logging
import sys
from pathlib import Path

# Vision Framework через PyObjC
try:
    from Vision import (
        VNRecognizeTextRequest,
        VNImageRequestHandler,
        VNRequestTextRecognitionLevelAccurate,
        VNRequestTextRecognitionLevelFast
    )
    from Cocoa import NSImage
    from Foundation import NSURL
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Vision Framework недоступен: {str(e)}. Требуется PyObjC: pip install pyobjc-framework-Vision")

app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from config.settings import settings

logger = logging.getLogger(__name__)


class OCREngine:
    """Класс для выполнения OCR распознавания с использованием Vision Framework"""
    
    def __init__(self):
        """Инициализация OCR движка"""
        if not VISION_AVAILABLE:
            raise RuntimeError(
                "Vision Framework недоступен. Установите: pip install pyobjc-framework-Vision"
            )
        
        self.recognition_level = VNRequestTextRecognitionLevelAccurate
        self.uses_language_correction = True
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Препроцессинг изображения для улучшения качества OCR
        
        Args:
            image: Входное изображение
            
        Returns:
            Обработанное изображение
        """
        # Конвертация в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Масштабирование для улучшения качества
        height, width = gray.shape
        if height < 1500 or width < 1500:
            scale = max(1500 / height, 1500 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Улучшение контраста (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Легкое удаление шума
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 5, 7, 21)
        
        return denoised
    
    def load_image(self, file_path: str, page: Optional[int] = None) -> Tuple[np.ndarray, str]:
        """
        Загрузка изображения из файла (PDF, JPG, PNG)
        
        Args:
            file_path: Путь к файлу
            page: Номер страницы для PDF (если None - первая страница)
            
        Returns:
            Кортеж (изображение, тип файла)
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Конвертация PDF в изображение
            if page is not None:
                images = convert_from_path(file_path, dpi=300, first_page=page, last_page=page)
            else:
                images = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
            
            if images:
                img_array = np.array(images[0])
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array, 'pdf'
            else:
                raise ValueError(f"Не удалось конвертировать PDF: {file_path}")
        
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {file_path}")
            return image, 'image'
        
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
    
    def get_pdf_page_count(self, file_path: str) -> int:
        """
        Получение количества страниц в PDF
        
        Args:
            file_path: Путь к PDF файлу
            
        Returns:
            Количество страниц
        """
        try:
            info = pdfinfo_from_path(file_path)
            return info.get('Pages', 1)
        except:
            return 1
    
    def process_file_all_pages(self, file_path: str) -> List[Dict[str, any]]:
        """
        Обработка всех страниц PDF файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Список результатов для каждой страницы
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            # Получаем количество страниц
            page_count = self.get_pdf_page_count(file_path)
            results = []
            
            for page_num in range(1, page_count + 1):
                try:
                    image, file_type = self.load_image(file_path, page=page_num)
                    result = self.extract_text(image)
                    result['file_type'] = file_type
                    result['file_path'] = file_path
                    result['page_number'] = page_num
                    result['total_pages'] = page_count
                    results.append(result)
                except Exception as e:
                    logger.error(f"Ошибка при обработке страницы {page_num} файла {file_path}: {str(e)}")
                    continue
            
            return results
        else:
            # Для не-PDF файлов обрабатываем как обычно
            image, file_type = self.load_image(file_path)
            result = self.extract_text(image)
            result['file_type'] = file_type
            result['file_path'] = file_path
            result['page_number'] = 1
            result['total_pages'] = 1
            return [result]
    
    def extract_text_vision(self, image_path: str) -> Dict[str, any]:
        """
        Извлечение текста с использованием Vision Framework
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Словарь с результатами распознавания
        """
        try:
            from Cocoa import NSImage
            from Foundation import NSURL
            
            # Создаем NSURL из пути
            file_url = NSURL.fileURLWithPath_(image_path)
            
            # Загружаем изображение через NSImage
            ns_image = NSImage.alloc().initWithContentsOfURL_(file_url)
            if not ns_image:
                raise ValueError("Не удалось загрузить изображение")
            
            # Создаем request для распознавания текста с правильными параметрами
            request = VNRecognizeTextRequest.alloc().init()
            # Устанавливаем параметры через правильные методы
            if self.recognition_level == VNRequestTextRecognitionLevelAccurate:
                request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
            else:
                request.setRecognitionLevel_(VNRequestTextRecognitionLevelFast)
            request.setUsesLanguageCorrection_(self.uses_language_correction)
            
            # Создаем handler для изображения
            handler = VNImageRequestHandler.alloc().initWithData_options_(
                ns_image.TIFFRepresentation(), {}
            )
            
            # Выполняем распознавание
            error_ref = None
            success = handler.performRequests_error_([request], error_ref)
            
            if not success:
                logger.warning("Vision Framework не смог обработать изображение")
                return {
                    'text': '',
                    'confidence': 0.0,
                    'detailed_data': {},
                    'word_count': 0
                }
            
            # Извлекаем результаты
            observations = request.results()
            texts = []
            confidences = []
            text_regions = []  # Координаты областей текста
            
            if observations:
                for observation in observations:
                    # Получаем топ кандидатов
                    candidates = observation.topCandidates_(1)
                    if candidates and len(candidates) > 0:
                        candidate = candidates[0]
                        text = candidate.string()
                        confidence = float(candidate.confidence())
                        texts.append(text)
                        confidences.append(confidence)
                        
                        # Получаем координаты области (bounding box)
                        try:
                            bounding_box = observation.boundingBox()
                            if bounding_box:
                                # Координаты в формате CGRect (x, y, width, height)
                                x = float(bounding_box.origin.x)
                                y = float(bounding_box.origin.y)
                                width = float(bounding_box.size.width)
                                height = float(bounding_box.size.height)
                                
                                text_regions.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': {
                                        'x': x,
                                        'y': y,
                                        'width': width,
                                        'height': height,
                                        'x1': x,
                                        'y1': y,
                                        'x2': x + width,
                                        'y2': y + height
                                    }
                                })
                        except Exception as e:
                            logger.debug(f"Не удалось получить координаты: {str(e)}")
            
            full_text = '\n'.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'text': full_text.strip(),
                'confidence': avg_confidence,
                'detailed_data': {
                    'text_regions': text_regions,
                    'total_regions': len(text_regions)
                },
                'word_count': len([w for w in full_text.split() if w.strip()])
            }
            
        except Exception as e:
            logger.error(f"Ошибка Vision Framework: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {
                'text': '',
                'confidence': 0.0,
                'detailed_data': {},
                'word_count': 0
            }
    
    def extract_text(self, image: np.ndarray, config: Optional[str] = None) -> Dict[str, any]:
        """
        Извлечение текста из изображения
        
        Args:
            image: Изображение для обработки
            config: Игнорируется (для совместимости)
            
        Returns:
            Словарь с результатами распознавания
        """
        import tempfile
        
        # Препроцессинг
        processed_image = self.preprocess_image(image)
        
        # Сохраняем во временный файл для Vision Framework
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, processed_image)
        
        try:
            # Распознавание через Vision Framework
            result = self.extract_text_vision(tmp_path)
            return result
        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_path)
            except:
                pass
    
    def extract_text_by_area(self, image: np.ndarray, area: Dict[str, int]) -> Dict[str, any]:
        """
        Извлечение текста из определенной области изображения
        
        Args:
            image: Изображение
            area: Словарь с координатами {'x1', 'y1', 'x2', 'y2'}
            
        Returns:
            Результаты распознавания для области
        """
        x1, y1 = area['x1'], area['y1']
        x2, y2 = area['x2'], area['y2']
        
        # Обрезка области
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'text': '', 'confidence': 0.0, 'detailed_data': {}}
        
        return self.extract_text(roi)
    
    def process_file(self, file_path: str) -> Dict[str, any]:
        """
        Полная обработка файла
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Результаты распознавания
        """
        try:
            image, file_type = self.load_image(file_path)
            result = self.extract_text(image)
            result['file_type'] = file_type
            result['file_path'] = file_path
            return result
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
            raise
