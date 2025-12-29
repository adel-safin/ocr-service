#!/usr/bin/env python3
"""
Полное обучение всех ML моделей Фазы 2
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Обучение всех моделей"""
    print("=" * 80)
    print("ОБУЧЕНИЕ ML МОДЕЛЕЙ ФАЗЫ 2")
    print("=" * 80)
    
    scripts_dir = Path(__file__).parent
    
    # Шаг 1: Подготовка данных
    print("\n" + "=" * 80)
    print("ШАГ 1: ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)
    subprocess.run([sys.executable, str(scripts_dir / "prepare_training_data.py")])
    
    # Шаг 2: Обучение классификатора документов
    print("\n" + "=" * 80)
    print("ШАГ 2: ОБУЧЕНИЕ КЛАССИФИКАТОРА ДОКУМЕНТОВ")
    print("=" * 80)
    print("⚠️  Это может занять некоторое время...")
    subprocess.run([sys.executable, str(scripts_dir / "train_classifier.py")])
    
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n💡 Примечания:")
    print("   - Классификатор документов обучен и готов к использованию")
    print("   - ML исправление опечаток требует более сложной дообучки T5")
    print("   - ML оценка качества требует размеченных данных")


if __name__ == "__main__":
    main()
