"""Валидаторы критических полей документов"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валидации поля"""
    field_name: str
    value: str
    valid: bool
    confidence: float
    message: Optional[str] = None
    suggested_correction: Optional[str] = None


class FieldValidator:
    """Валидатор полей документов"""
    
    # Регулярные выражения для критических полей
    PATTERNS = {
        'ogrn': {
            'pattern': r'\b\d{13,15}\b',  # ОГРН: 13-15 цифр
            'validation': lambda x: len(re.sub(r'\D', '', x)) in [13, 15],
            'description': 'ОГРН (13 или 15 цифр)'
        },
        'inn': {
            'pattern': r'\b\d{10,12}\b',  # ИНН: 10 или 12 цифр
            'validation': lambda x: len(re.sub(r'\D', '', x)) in [10, 12],
            'description': 'ИНН (10 или 12 цифр)'
        },
        'kpp': {
            'pattern': r'\b\d{9}\b',  # КПП: 9 цифр
            'validation': lambda x: len(re.sub(r'\D', '', x)) == 9,
            'description': 'КПП (9 цифр)'
        },
        'date': {
            'pattern': r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',  # Дата
            'validation': lambda x: bool(re.match(r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}', x)),
            'description': 'Дата (ДД.ММ.ГГГГ)'
        },
        'snils': {
            'pattern': r'\b\d{3}-\d{3}-\d{3}\s\d{2}\b|\b\d{11}\b',  # СНИЛС
            'validation': lambda x: len(re.sub(r'\D', '', x)) == 11,
            'description': 'СНИЛС (11 цифр)'
        },
        'certificate_number': {
            'pattern': r'[№N]\s*[ЕАЭС\s]*[RU\s]*[ДС]\s*-?\s*RU[.\s]*[А-Я]{2}\d{2}[.\s]*[ВВ]\s*\.?\s*\d{5,6}\s*_\s*\d{2}',
            'validation': lambda x: bool(re.search(r'[ЕАЭС]', x, re.IGNORECASE)),
            'description': 'Номер сертификата ЕАЭС'
        },
        'phone': {
            'pattern': r'[+7]?\s*\(?\d{3}\)?\s*\d{3}[-.\s]?\d{2}[-.\s]?\d{2}',
            'validation': lambda x: len(re.sub(r'\D', '', x)) >= 10,
            'description': 'Телефон'
        },
        'email': {
            'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'validation': lambda x: '@' in x and '.' in x.split('@')[1],
            'description': 'Email'
        },
        'number': {
            'pattern': r'[№N]\s*[:\s]*[А-Яа-яA-Za-z0-9\-\.\/\s]+',
            'validation': lambda x: bool(re.search(r'[№N]', x, re.IGNORECASE)),
            'description': 'Номер документа'
        },
        'surname': {
            'pattern': r'\b[А-ЯЁ][а-яё]+\b(?=\s+[А-ЯЁ][А-ЯЁ\.])',  # Фамилия перед инициалами
            'validation': lambda x: len(x) >= 2 and x[0].isupper(),
            'description': 'Фамилия'
        }
    }
    
    def __init__(self):
        """Инициализация валидатора"""
        self.compiled_patterns = {
            name: re.compile(pattern['pattern'], re.IGNORECASE | re.UNICODE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def find_field(self, field_name: str, text: str) -> List[Tuple[str, float]]:
        """
        Поиск значения поля в тексте
        
        Args:
            field_name: Название поля
            text: Текст для поиска
            
        Returns:
            Список найденных значений с уверенностью
        """
        if field_name not in self.compiled_patterns:
            logger.warning(f"Паттерн для поля '{field_name}' не найден")
            return []
        
        pattern = self.compiled_patterns[field_name]
        matches = pattern.findall(text)
        
        results = []
        for match in matches:
            # Очистка найденного значения
            cleaned = re.sub(r'\s+', ' ', str(match).strip())
            
            # Валидация
            validator = self.PATTERNS[field_name]['validation']
            is_valid = validator(cleaned)
            
            # Уверенность зависит от валидности
            confidence = 0.9 if is_valid else 0.5
            
            results.append((cleaned, confidence))
        
        return results
    
    def validate_field(self, field_name: str, value: str, text: str = "") -> ValidationResult:
        """
        Валидация конкретного поля
        
        Args:
            field_name: Название поля
            value: Значение для валидации
            text: Контекстный текст (опционально)
            
        Returns:
            Результат валидации
        """
        if field_name not in self.PATTERNS:
            return ValidationResult(
                field_name=field_name,
                value=value,
                valid=False,
                confidence=0.0,
                message=f"Неизвестный тип поля: {field_name}"
            )
        
        validator = self.PATTERNS[field_name]['validation']
        is_valid = validator(value)
        
        # Проверка по паттерну
        pattern = self.compiled_patterns[field_name]
        matches_pattern = bool(pattern.search(value))
        
        valid = is_valid and matches_pattern
        
        # Поиск альтернативных значений в тексте
        suggested = None
        if not valid and text:
            alternatives = self.find_field(field_name, text)
            if alternatives:
                # Выбираем наиболее подходящий вариант
                suggested = max(alternatives, key=lambda x: x[1])[0]
        
        confidence = 0.9 if valid else 0.5
        
        return ValidationResult(
            field_name=field_name,
            value=value,
            valid=valid,
            confidence=confidence,
            message=f"Поле {self.PATTERNS[field_name]['description']} {'валидно' if valid else 'невалидно'}",
            suggested_correction=suggested
        )
    
    def validate_critical_fields(self, text: str, required_fields: Optional[List[str]] = None) -> Dict[str, ValidationResult]:
        """
        Валидация всех критических полей в тексте
        
        Args:
            text: Текст для анализа
            required_fields: Список обязательных полей (если None - все)
            
        Returns:
            Словарь результатов валидации
        """
        results = {}
        
        fields_to_check = required_fields if required_fields else list(self.PATTERNS.keys())
        
        for field_name in fields_to_check:
            found_values = self.find_field(field_name, text)
            
            if found_values:
                # Берем значение с наибольшей уверенностью
                best_value, best_confidence = max(found_values, key=lambda x: x[1])
                result = self.validate_field(field_name, best_value, text)
                result.confidence = best_confidence
                results[field_name] = result
            else:
                results[field_name] = ValidationResult(
                    field_name=field_name,
                    value="",
                    valid=False,
                    confidence=0.0,
                    message=f"Поле '{field_name}' не найдено в документе"
                )
        
        return results
    
    def extract_important_data(self, text: str) -> Dict[str, List[str]]:
        """
        Извлечение важных данных из текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с найденными данными
        """
        important_data = {
            'inn': [],
            'snils': [],
            'numbers': [],
            'surnames': []
        }
        
        # ИНН
        inn_matches = self.find_field('inn', text)
        important_data['inn'] = [val[0] for val in inn_matches]
        
        # СНИЛС
        snils_matches = self.find_field('snils', text)
        important_data['snils'] = [val[0] for val in snils_matches]
        
        # Номера (№)
        number_matches = self.find_field('number', text)
        important_data['numbers'] = [val[0] for val in number_matches]
        
        # Фамилии (улучшенный поиск)
        # Ищем фамилии в разных форматах
        surname_patterns = [
            r'\b[А-ЯЁ][а-яё]{2,}\b(?=\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)',  # Фамилия И.О.
            r'\b[А-ЯЁ][а-яё]{2,}\b(?=\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)',  # Фамилия Имя Отчество
            r'[А-ЯЁ][а-яё]{3,}(?=\s+[А-ЯЁ]\.)',  # Фамилия И.
        ]
        
        found_surnames = set()
        for pattern in surname_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3 and match[0].isupper():
                    # Исключаем общие слова
                    if match.lower() not in ['россия', 'российская', 'федерация', 'республика', 'область', 'край']:
                        found_surnames.add(match)
        
        important_data['surnames'] = list(found_surnames)[:10]  # Ограничиваем до 10
        
        return important_data
