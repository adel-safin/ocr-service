"""Модель исправления опечаток на основе Transformer"""
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration, T5Tokenizer
)
from typing import List, Dict, Optional, Tuple
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class SpellCorrector:
    """Модель исправления опечаток на основе T5/Transformer"""
    
    def __init__(self, model_name: str = "cointegrated/rut5-base", device: str = 'cpu', local_path: Optional[str] = None):
        """
        Инициализация модели исправления
        
        Args:
            model_name: Название модели из HuggingFace
            device: Устройство для вычислений
            local_path: Путь к локальной модели (если есть)
        """
        self.device = device
        self.model_name = model_name
        
        # Проверка локальной модели
        import os
        from pathlib import Path
        
        if local_path is None:
            # Попытка найти локальную модель
            app_root = Path(__file__).parent.parent
            local_rut5 = app_root / "rut5-base"
            if local_rut5.exists() and local_rut5.is_dir():
                local_path = str(local_rut5)
                logger.info(f"Найдена локальная модель: {local_path}")
        
        model_path = local_path if local_path else model_name
        
        try:
            # Загрузка токенизатора и модели
            # Используем use_fast=False для избежания проблем с конвертацией
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=False,  # Используем медленный токенизатор для совместимости
                    local_files_only=local_path is not None
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить токенизатор с use_fast=False: {str(e)}")
                # Попытка загрузить без параметра use_fast
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=local_path is not None
                )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=local_path is not None
            )
            self.model.to(device)
            self.model.eval()
            logger.info(f"Модель исправления загружена: {model_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель {model_path}: {str(e)}")
            logger.info("Используется базовый режим исправления")
            self.model = None
            self.tokenizer = None
    
    def correct_text(self, text: str, max_length: int = 512) -> str:
        """
        Исправление текста
        
        Args:
            text: Текст для исправления
            max_length: Максимальная длина
            
        Returns:
            Исправленный текст
        """
        if self.model is None or self.tokenizer is None:
            return text
        
        # Для T5 моделей лучше не использовать исправление всего текста сразу
        # так как это может привести к генерации служебных токенов
        # Вместо этого используем простой исправитель для коротких текстов
        # или отключаем ML исправление для длинных текстов
        
        # Если текст слишком длинный или содержит много ошибок OCR, 
        # лучше использовать базовые правила
        if len(text) > 200 or text.count(' ') < 5:
            logger.debug("Текст слишком длинный или короткий для ML исправления, используем базовые правила")
            return text
        
        try:
            # Для T5 используем правильный формат промпта
            # T5 ожидает префикс задачи
            prompt = f"исправить: {text}"
            
            # Токенизация
            inputs = self.tokenizer(
                prompt,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Генерация исправления с правильными параметрами для T5
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=min(max_length, inputs['input_ids'].shape[1] + 50),
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    repetition_penalty=1.2
                )
            
            # Декодирование
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Проверка на служебные токены T5
            if '<extra_id' in corrected or corrected.strip() == '':
                logger.warning("ML модель вернула служебные токены, используем оригинальный текст")
                return text
            
            # Удаление префикса промпта, если есть
            if ":" in corrected and len(corrected.split(":")) > 1:
                corrected = corrected.split(":", 1)[-1].strip()
            
            # Если результат слишком отличается от оригинала, возвращаем оригинал
            if len(corrected) < len(text) * 0.5 or len(corrected) > len(text) * 2:
                logger.warning(f"ML исправление слишком сильно изменило текст, используем оригинал")
                return text
            
            return corrected
        
        except Exception as e:
            logger.error(f"Ошибка при исправлении текста: {str(e)}")
            return text
    
    def correct_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        Пакетное исправление текстов
        
        Args:
            texts: Список текстов для исправления
            batch_size: Размер батча
            
        Returns:
            Список исправленных текстов
        """
        if self.model is None:
            return texts
        
        corrected_texts = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            corrected_batch = [self.correct_text(text) for text in batch]
            corrected_texts.extend(corrected_batch)
        
        return corrected_texts
    
    def correct_with_context(self, text: str, context: Optional[str] = None) -> str:
        """
        Исправление текста с учетом контекста
        
        Args:
            text: Текст для исправления
            context: Контекстный текст
            
        Returns:
            Исправленный текст
        """
        if context:
            # Объединение контекста и текста
            full_text = f"{context} {text}"
            corrected = self.correct_text(full_text)
            # Возврат только исправленного текста (последняя часть)
            return corrected.split()[-len(text.split()):] if len(corrected.split()) >= len(text.split()) else corrected
        else:
            return self.correct_text(text)


class SimpleSpellCorrector:
    """Простая модель исправления на основе правил и статистики"""
    
    def __init__(self):
        """Инициализация простого исправителя"""
        # Частые ошибки OCR
        self.common_errors = {
            'О': '0', 'I': '1', 'З': '3', 'Б': '6', 'В': '8',
            'S': '5', 'G': '6', 'Z': '2', 'l': '1', 'o': '0'
        }
    
    def correct_text(self, text: str) -> str:
        """
        Простое исправление текста
        
        Args:
            text: Текст для исправления
            
        Returns:
            Исправленный текст
        """
        corrected = text
        
        # Замена частых ошибок в цифровых последовательностях
        import re
        
        # Поиск последовательностей, которые могут быть числами
        number_pattern = r'\b[ОIЗБВSGl0-9]+\b'
        
        def replace_in_numbers(match):
            num_str = match.group()
            # Если это похоже на число, заменяем буквы на цифры
            for letter, digit in self.common_errors.items():
                num_str = num_str.replace(letter, digit)
            return num_str
        
        corrected = re.sub(number_pattern, replace_in_numbers, corrected)
        
        return corrected
