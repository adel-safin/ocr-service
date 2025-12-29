#!/usr/bin/env python3
"""
Запуск всех тестов: Фаза 1 -> Фаза 2 -> Фаза 3
"""
import sys
import subprocess
from pathlib import Path

scripts_dir = Path(__file__).parent

print("=" * 80)
print("ПОЛНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ OCR")
print("=" * 80)

# Фаза 1
print("\n" + "=" * 80)
print("ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА")
print("=" * 80)
subprocess.run([sys.executable, str(scripts_dir / "test_phase1.py")])

# Фаза 2
print("\n" + "=" * 80)
print("ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ")
print("=" * 80)
subprocess.run([sys.executable, str(scripts_dir / "test_phase2.py")])

# Фаза 3
print("\n" + "=" * 80)
print("ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ")
print("=" * 80)
subprocess.run([sys.executable, str(scripts_dir / "test_phase3.py")])

print("\n" + "=" * 80)
print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
print("=" * 80)
