"""Система автокоррекции текста"""
import json
import os
import re
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import logging

import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from config.settings import settings

logger = logging.getLogger(__name__)

# Латиница → кириллица (типичные OCR-ошибки: фамилии, инициалы). Не трогать в @, доменах.
_LATIN_TO_CYRILLIC = str.maketrans(
    "aecopxyABCEHKMOPTXYIisGNPlL",
    "аесорхуАВСЕНКМОРТХУИисГНРлЛ",
)


class AutoCorrectionSystem:
    """Система автоматической коррекции ошибок OCR"""
    
    def __init__(self):
        """Инициализация системы коррекции"""
        self.corrections_db = self.load_corrections()
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.max_distance = settings.MAX_CORRECTION_DISTANCE
    
    def load_corrections(self) -> Dict[str, str]:
        """
        Загрузка базы автозамен из файла
        
        Returns:
            Словарь замен (ошибка -> исправление)
        """
        corrections_path = settings.CORRECTIONS_DB
        
        # Создание файла, если не существует
        if not os.path.exists(corrections_path):
            os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
            default_corrections = {
                "Маркуталь": "Мариуполь",
                "I": "1",
                "E-ma1l": "E-mail",
            }
            self.save_corrections(default_corrections)
            return default_corrections
        
        try:
            with open(corrections_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка при загрузке базы исправлений: {str(e)}")
            return {}
    
    def save_corrections(self, corrections: Optional[Dict[str, str]] = None):
        """
        Сохранение базы автозамен
        
        Args:
            corrections: Словарь замен (если None - сохраняет текущий)
        """
        if corrections is None:
            corrections = self.corrections_db
        
        corrections_path = settings.CORRECTIONS_DB
        os.makedirs(os.path.dirname(corrections_path), exist_ok=True)
        
        try:
            with open(corrections_path, 'w', encoding='utf-8') as f:
                json.dump(corrections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении базы исправлений: {str(e)}")
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Вычисление схожести двух строк
        
        Args:
            str1: Первая строка
            str2: Вторая строка
            
        Returns:
            Коэффициент схожести (0.0 - 1.0)
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Вычисление расстояния Левенштейна
        
        Args:
            s1: Первая строка
            s2: Вторая строка
            
        Returns:
            Расстояние редактирования
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_similar_correction(self, text: str) -> Optional[Tuple[str, str, float]]:
        """
        Поиск похожего исправления в базе
        
        Args:
            text: Текст для проверки
            
        Returns:
            Кортеж (оригинал, исправление, уверенность) или None
        """
        best_match = None
        best_similarity = 0.0
        
        for original, correction in self.corrections_db.items():
            similarity = self.calculate_similarity(text, original)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = (original, correction, similarity)
        
        return best_match
    
    def suggest_correction(self, text: str, context: Optional[str] = None) -> Optional[Dict[str, any]]:
        """
        Предложение исправления для текста
        
        Args:
            text: Текст для исправления
            context: Контекстный текст (опционально)
            
        Returns:
            Словарь с предложением исправления или None
        """
        # Поиск точного совпадения
        if text in self.corrections_db:
            return {
                'original': text,
                'corrected': self.corrections_db[text],
                'confidence': 1.0,
                'method': 'exact_match'
            }
        
        # Поиск похожего совпадения
        similar = self.find_similar_correction(text)
        if similar:
            original, corrected, confidence = similar
            return {
                'original': text,
                'corrected': corrected,
                'confidence': confidence,
                'method': 'similarity_match',
                'matched_original': original
            }
        
        return None
    
    def is_russian_char(self, char: str) -> bool:
        """
        Проверка, является ли символ русской буквой
        
        Args:
            char: Символ для проверки
            
        Returns:
            True если русская буква
        """
        return '\u0400' <= char <= '\u04FF'
    
    def is_russian_word(self, word: str) -> bool:
        """
        Проверка, является ли слово русским (содержит русские буквы)
        
        Args:
            word: Слово для проверки
            
        Returns:
            True если слово содержит русские буквы
        """
        return any(self.is_russian_char(c) for c in word)

    def _fix_latin_in_russian_words(self, text: str) -> Tuple[str, List[Dict]]:
        """Замена латиницы на кириллицу в словах с примесью кириллицы (фамилии, инициалы). Не трогает @ и домены."""
        fixes = []
        _skip_domain = (".ru", ".com", ".org", ".net", ".рф")

        def repl(m):
            w = m.group(0)
            if "@" in w or any(s in w for s in _skip_domain) or not any(self.is_russian_char(c) for c in w):
                return w
            latin_src = "aecopxyABCEHKMOPTXYIisGNPlL"
            if not any(c in w for c in latin_src):
                return w
            new = w.translate(_LATIN_TO_CYRILLIC)
            if new != w:
                fixes.append({"from": w, "to": new, "confidence": 0.9, "method": "latin_to_cyrillic"})
                return new
            return w

        out = re.sub(r"\w+", repl, text)
        return out, fixes

    def correct_text(
        self,
        text: str,
        *,
        digit_replacement_extended: bool = False,
        extra_corrections: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[Dict[str, any]]]:
        """
        Автокоррекция: 0) extra_corrections (Фаза 3), 1) точные из базы, 2) латиница→кириллица,
        3) 0/8 (и при digit_replacement_extended — 3, 6) в русских словах, 4) похожие из базы.
        """
        corrections_applied = []
        corrected_text = text

        # 0) Доп. замены (напр. из активного обучения, Фаза 3)
        _extras = extra_corrections or {}
        for k in sorted(_extras, key=len, reverse=True):
            if k and k in corrected_text:
                corrected_text = corrected_text.replace(k, _extras[k])
                corrections_applied.append({"from": k, "to": _extras[k], "confidence": 0.85, "method": "extra_corrections"})

        # Исключаем однобуквенные О,В,З,Б — они портят русские слова (цифра→буква делается в п.3)
        _skip = {"О", "В", "З", "Б", "з"}

        # 1) Точные замены из базы (длинные первыми)
        for k in sorted(self.corrections_db, key=len, reverse=True):
            if k in _skip and len(k) == 1:
                continue
            if k in corrected_text:
                corrected_text = corrected_text.replace(k, self.corrections_db[k])
                corrections_applied.append({"from": k, "to": self.corrections_db[k], "confidence": 1.0, "method": "exact_match"})

        # 2) Латиница → кириллица в смешанных словах
        corrected_text, lat_fixes = self._fix_latin_in_russian_words(corrected_text)
        corrections_applied.extend(lat_fixes)

        # 3) 0→о/О, 8→в/В; при digit_replacement_extended ещё 3→з/З, 6→б/Б в русских словах
        digit_extended = digit_replacement_extended
        import re
        word_pattern = re.compile(r'\b[А-Яа-яЁё0-9]+\b')

        def replace_digits_in_russian_word(match):
            word = match.group(0)
            original_word = word
            has_digit = '0' in word or '8' in word or (digit_extended and ('3' in word or '6' in word))
            if not (self.is_russian_word(word) and not word.isdigit() and has_digit):
                return word

            has_lowercase = any(c.islower() for c in word if c.isalpha())
            use_uppercase = not has_lowercase and any(c.isupper() for c in word if c.isalpha())

            if use_uppercase:
                new_word = word.replace('0', 'О').replace('8', 'В')
                if digit_extended:
                    new_word = new_word.replace('3', 'З').replace('6', 'Б')
            else:
                new_word = word.replace('0', 'о').replace('8', 'в')
                if digit_extended:
                    new_word = new_word.replace('3', 'з').replace('6', 'б')

            if new_word != original_word:
                    # Подсчитываем замены для отчета
                    zero_count = word.count('0')
                    eight_count = word.count('8')
                    
                    # Добавляем запись на слово
                    corrections_applied.append({
                        'from': original_word,
                        'to': new_word,
                        'confidence': 0.95,
                        'method': 'contextual_russian_word',
                        'context': word,
                        'zero_replacements': zero_count,
                        'eight_replacements': eight_count,
                        'case': 'uppercase' if use_uppercase else 'lowercase'
                    })
                    
                    return new_word
            
            return word
        
        # Применяем замену цифр на буквы в русских словах
        corrected_text = word_pattern.sub(replace_digits_in_russian_word, corrected_text)
        
        # Затем применяем обычные правила из базы исправлений
        words = re.findall(r'\b\w+\b|\W+', corrected_text)
        
        for word in words:
            if not word.strip() or not word.isalnum():
                continue
            
            suggestion = self.suggest_correction(word)
            if suggestion:
                original = suggestion['original']
                corrected = suggestion['corrected']
                confidence = suggestion['confidence']
                
                # Применение исправления (только если еще не было заменено)
                if original in corrected_text:
                    corrected_text = corrected_text.replace(original, corrected, 1)
                    
                    corrections_applied.append({
                        'from': original,
                        'to': corrected,
                        'confidence': confidence,
                        'method': suggestion.get('method', 'unknown')
                    })
        
        return corrected_text, corrections_applied
    
    def learn_from_mistake(self, original: str, corrected: str, context: Optional[str] = None):
        """
        Обучение на исправлении (добавление в базу)
        
        Args:
            original: Оригинальная (ошибочная) версия
            corrected: Исправленная версия
            context: Контекст (опционально)
        """
        if original != corrected:
            self.corrections_db[original] = corrected
            self.save_corrections()
            logger.info(f"Добавлено исправление: '{original}' -> '{corrected}'")
    
    def add_correction(self, original: str, corrected: str, confirm: bool = True):
        """
        Добавление исправления в базу
        
        Args:
            original: Оригинальная версия
            corrected: Исправленная версия
            confirm: Требуется ли подтверждение
        """
        if confirm:
            # Здесь можно добавить логику подтверждения
            pass
        
        self.learn_from_mistake(original, corrected)
