#!/usr/bin/env python3
"""
Обработка одного файла через все три фазы
"""
import sys
from pathlib import Path
import json
from datetime import datetime

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_single_file(file_path: str):
    """Обработка одного файла через все фазы"""
    print("=" * 100)
    print("ОБРАБОТКА ФАЙЛА - ВСЕ ФАЗЫ")
    print("=" * 100)
    
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        print(f"❌ Файл не найден: {file_path}")
        return
    
    print(f"\n📁 Файл: {file_path_obj.name}")
    print(f"📋 Обработка через все фазы...\n")
    
    # Создание пайплайнов для каждой фазы
    print("🔧 Инициализация пайплайнов...")
    pipeline_phase1 = DocumentPipeline(use_ml=False, use_active_learning=False)
    pipeline_phase2 = DocumentPipeline(use_ml=True, use_active_learning=False)
    pipeline_phase3 = DocumentPipeline(use_ml=True, use_active_learning=True)
    
    # Создание папок для результатов
    output_dir = Path("data/single_file_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_dir = output_dir / "phase1"
    phase2_dir = output_dir / "phase2"
    phase3_dir = output_dir / "phase3"
    
    phase1_dir.mkdir(exist_ok=True)
    phase2_dir.mkdir(exist_ok=True)
    phase3_dir.mkdir(exist_ok=True)
    
    try:
        # ФАЗА 1: Базовый OCR + правила
        print(f"   📌 Фаза 1...", end=" ")
        result1 = pipeline_phase1.process(str(file_path_obj), template="certificate")
        
        # Сохранение текста Фазы 1
        pages_data = result1['extracted_data'].get('pages', [])
        total_pages = result1['extracted_data'].get('total_pages', 1)
        
        phase1_file = phase1_dir / f"{file_path_obj.stem}_phase1.txt"
        with open(phase1_file, 'w', encoding='utf-8') as f:
            f.write(f"ФАЙЛ: {file_path_obj.name}\n")
            f.write(f"ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА\n")
            f.write("=" * 100 + "\n")
            f.write(f"Document ID: {result1['document_id']}\n")
            f.write(f"Качество: {result1['quality_report']['overall_quality']:.2%}\n")
            f.write(f"Уверенность OCR: {result1['quality_report']['ocr_confidence']:.2%}\n")
            f.write(f"Исправлений: {len(result1.get('corrections_applied', []))}\n")
            f.write(f"Всего страниц: {total_pages}\n")
            
            # Детали качества
            quality_report = result1['quality_report']
            f.write(f"\nДЕТАЛИ КАЧЕСТВА:\n")
            f.write(f"  - Уверенность OCR: {quality_report.get('ocr_confidence', 0):.2%}\n")
            img_quality = quality_report.get('image_quality', {})
            if isinstance(img_quality, dict):
                f.write(f"  - Качество изображения: {img_quality.get('overall_quality', 0):.2%}\n")
            else:
                f.write(f"  - Качество изображения: {img_quality:.2%}\n")
            f.write(f"  - Общее качество: {quality_report.get('overall_quality', 0):.2%}\n")
            
            if quality_report.get('warnings'):
                f.write(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ:\n")
                for warning in quality_report['warnings']:
                    f.write(f"  - {warning}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            
            # Сохранение текста по страницам
            if pages_data and len(pages_data) > 1:
                f.write("РАСПОЗНАННЫЙ ТЕКСТ ПО СТРАНИЦАМ:\n")
                f.write("=" * 100 + "\n")
                for page_info in pages_data:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_conf = page_info.get('confidence', 0.0)
                    f.write(f"\n--- СТРАНИЦА {page_num} (уверенность: {page_conf:.2%}) ---\n")
                    f.write(page_text)
                    f.write("\n")
            else:
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result1['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            # Исправления
            if result1.get('corrections_applied'):
                f.write("\nИСПРАВЛЕНИЯ:\n")
                f.write("-" * 100 + "\n")
                for j, correction in enumerate(result1['corrections_applied'], 1):
                    f.write(f"{j}. {correction.get('from', '')} -> {correction.get('to', '')} (метод: {correction.get('method', 'unknown')})\n")
        
        # Сохранение отдельных файлов для каждой страницы
        if pages_data and len(pages_data) > 1:
            pages_dir = phase1_dir / f"{file_path_obj.stem}_pages"
            pages_dir.mkdir(exist_ok=True)
            
            for page_info in pages_data:
                page_num = page_info.get('page_number', 1)
                page_text = page_info.get('text', '')
                page_file = pages_dir / f"page_{page_num:03d}.txt"
                
                with open(page_file, 'w', encoding='utf-8') as pf:
                    pf.write(f"ФАЙЛ: {file_path_obj.name}\n")
                    pf.write(f"ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА\n")
                    pf.write(f"СТРАНИЦА: {page_num} из {total_pages}\n")
                    pf.write(f"Уверенность OCR: {page_info.get('confidence', 0.0):.2%}\n")
                    pf.write("\n" + "=" * 100 + "\n")
                    pf.write("ТЕКСТ СТРАНИЦЫ:\n")
                    pf.write("=" * 100 + "\n")
                    pf.write(page_text)
                    pf.write("\n" + "=" * 100 + "\n")
        
        print(f"✅ Качество: {result1['quality_report']['overall_quality']:.2%}")
        
        # ФАЗА 2: С ML компонентами
        print(f"   🤖 Фаза 2...", end=" ")
        result2 = pipeline_phase2.process(str(file_path_obj), template="certificate")
        
        # Сохранение текста Фазы 2
        pages_data2 = result2['extracted_data'].get('pages', [])
        total_pages2 = result2['extracted_data'].get('total_pages', 1)
        
        phase2_file = phase2_dir / f"{file_path_obj.stem}_phase2.txt"
        with open(phase2_file, 'w', encoding='utf-8') as f:
            f.write(f"ФАЙЛ: {file_path_obj.name}\n")
            f.write(f"ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ\n")
            f.write("=" * 100 + "\n")
            f.write(f"Document ID: {result2['document_id']}\n")
            f.write(f"Качество: {result2['quality_report']['overall_quality']:.2%}\n")
            f.write(f"Уверенность OCR: {result2['quality_report']['ocr_confidence']:.2%}\n")
            f.write(f"Исправлений: {len(result2.get('corrections_applied', []))}\n")
            f.write(f"Всего страниц: {total_pages2}\n")
            
            # Детали качества
            quality_report = result2['quality_report']
            f.write(f"\nДЕТАЛИ КАЧЕСТВА:\n")
            f.write(f"  - Уверенность OCR: {quality_report.get('ocr_confidence', 0):.2%}\n")
            img_quality = quality_report.get('image_quality', {})
            if isinstance(img_quality, dict):
                f.write(f"  - Качество изображения: {img_quality.get('overall_quality', 0):.2%}\n")
            else:
                f.write(f"  - Качество изображения: {img_quality:.2%}\n")
            f.write(f"  - Общее качество: {quality_report.get('overall_quality', 0):.2%}\n")
            
            if quality_report.get('warnings'):
                f.write(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ:\n")
                for warning in quality_report['warnings']:
                    f.write(f"  - {warning}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            
            # Сохранение текста по страницам
            if pages_data2 and len(pages_data2) > 1:
                f.write("РАСПОЗНАННЫЙ ТЕКСТ ПО СТРАНИЦАМ:\n")
                f.write("=" * 100 + "\n")
                for page_info in pages_data2:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_conf = page_info.get('confidence', 0.0)
                    f.write(f"\n--- СТРАНИЦА {page_num} (уверенность: {page_conf:.2%}) ---\n")
                    f.write(page_text)
                    f.write("\n")
            else:
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result2['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            # Исправления
            if result2.get('corrections_applied'):
                f.write("\nИСПРАВЛЕНИЯ:\n")
                f.write("-" * 100 + "\n")
                for j, correction in enumerate(result2['corrections_applied'], 1):
                    f.write(f"{j}. {correction.get('from', '')} -> {correction.get('to', '')} (метод: {correction.get('method', 'unknown')})\n")
        
        # Сохранение отдельных файлов для каждой страницы
        if pages_data2 and len(pages_data2) > 1:
            pages_dir = phase2_dir / f"{file_path_obj.stem}_pages"
            pages_dir.mkdir(exist_ok=True)
            
            for page_info in pages_data2:
                page_num = page_info.get('page_number', 1)
                page_text = page_info.get('text', '')
                page_file = pages_dir / f"page_{page_num:03d}.txt"
                
                with open(page_file, 'w', encoding='utf-8') as pf:
                    pf.write(f"ФАЙЛ: {file_path_obj.name}\n")
                    pf.write(f"ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ\n")
                    pf.write(f"СТРАНИЦА: {page_num} из {total_pages2}\n")
                    pf.write(f"Уверенность OCR: {page_info.get('confidence', 0.0):.2%}\n")
                    pf.write("\n" + "=" * 100 + "\n")
                    pf.write("ТЕКСТ СТРАНИЦЫ:\n")
                    pf.write("=" * 100 + "\n")
                    pf.write(page_text)
                    pf.write("\n" + "=" * 100 + "\n")
        
        print(f"✅ Качество: {result2['quality_report']['overall_quality']:.2%}")
        
        # ФАЗА 3: С активным обучением
        print(f"   🔄 Фаза 3...", end=" ")
        result3 = pipeline_phase3.process(str(file_path_obj), template="certificate")
        
        # Сохранение текста Фазы 3
        pages_data3 = result3['extracted_data'].get('pages', [])
        total_pages3 = result3['extracted_data'].get('total_pages', 1)
        
        phase3_file = phase3_dir / f"{file_path_obj.stem}_phase3.txt"
        with open(phase3_file, 'w', encoding='utf-8') as f:
            f.write(f"ФАЙЛ: {file_path_obj.name}\n")
            f.write(f"ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ\n")
            f.write("=" * 100 + "\n")
            f.write(f"Document ID: {result3['document_id']}\n")
            f.write(f"Качество: {result3['quality_report']['overall_quality']:.2%}\n")
            f.write(f"Уверенность OCR: {result3['quality_report']['ocr_confidence']:.2%}\n")
            f.write(f"Исправлений: {len(result3.get('corrections_applied', []))}\n")
            f.write(f"Всего страниц: {total_pages3}\n")
            
            # Детали качества
            quality_report = result3['quality_report']
            f.write(f"\nДЕТАЛИ КАЧЕСТВА:\n")
            f.write(f"  - Уверенность OCR: {quality_report.get('ocr_confidence', 0):.2%}\n")
            img_quality = quality_report.get('image_quality', {})
            if isinstance(img_quality, dict):
                f.write(f"  - Качество изображения: {img_quality.get('overall_quality', 0):.2%}\n")
            else:
                f.write(f"  - Качество изображения: {img_quality:.2%}\n")
            f.write(f"  - Общее качество: {quality_report.get('overall_quality', 0):.2%}\n")
            
            if quality_report.get('warnings'):
                f.write(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ:\n")
                for warning in quality_report['warnings']:
                    f.write(f"  - {warning}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            
            # Сохранение текста по страницам
            if pages_data3 and len(pages_data3) > 1:
                f.write("РАСПОЗНАННЫЙ ТЕКСТ ПО СТРАНИЦАМ:\n")
                f.write("=" * 100 + "\n")
                for page_info in pages_data3:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_conf = page_info.get('confidence', 0.0)
                    f.write(f"\n--- СТРАНИЦА {page_num} (уверенность: {page_conf:.2%}) ---\n")
                    f.write(page_text)
                    f.write("\n")
            else:
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result3['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            # Исправления
            if result3.get('corrections_applied'):
                f.write("\nИСПРАВЛЕНИЯ:\n")
                f.write("-" * 100 + "\n")
                for j, correction in enumerate(result3['corrections_applied'], 1):
                    f.write(f"{j}. {correction.get('from', '')} -> {correction.get('to', '')} (метод: {correction.get('method', 'unknown')})\n")
        
        # Сохранение отдельных файлов для каждой страницы
        if pages_data3 and len(pages_data3) > 1:
            pages_dir = phase3_dir / f"{file_path_obj.stem}_pages"
            pages_dir.mkdir(exist_ok=True)
            
            for page_info in pages_data3:
                page_num = page_info.get('page_number', 1)
                page_text = page_info.get('text', '')
                page_file = pages_dir / f"page_{page_num:03d}.txt"
                
                with open(page_file, 'w', encoding='utf-8') as pf:
                    pf.write(f"ФАЙЛ: {file_path_obj.name}\n")
                    pf.write(f"ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ\n")
                    pf.write(f"СТРАНИЦА: {page_num} из {total_pages3}\n")
                    pf.write(f"Уверенность OCR: {page_info.get('confidence', 0.0):.2%}\n")
                    pf.write("\n" + "=" * 100 + "\n")
                    pf.write("ТЕКСТ СТРАНИЦЫ:\n")
                    pf.write("=" * 100 + "\n")
                    pf.write(page_text)
                    pf.write("\n" + "=" * 100 + "\n")
        
        print(f"✅ Качество: {result3['quality_report']['overall_quality']:.2%}")
        
        print(f"\n   ✅ Завершено\n")
        
        print(f"\n💾 Результаты сохранены:")
        print(f"   - Фаза 1: {phase1_file}")
        print(f"   - Фаза 2: {phase2_file}")
        print(f"   - Фаза 3: {phase3_file}")
        
        if pages_data and len(pages_data) > 1:
            print(f"   - Страницы Фазы 1: {phase1_dir / f'{file_path_obj.stem}_pages'}/")
        if pages_data2 and len(pages_data2) > 1:
            print(f"   - Страницы Фазы 2: {phase2_dir / f'{file_path_obj.stem}_pages'}/")
        if pages_data3 and len(pages_data3) > 1:
            print(f"   - Страницы Фазы 3: {phase3_dir / f'{file_path_obj.stem}_pages'}/")
        
        print("\n" + "=" * 100)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("=" * 100)
        
    except Exception as e:
        print(f"   ❌ Ошибка: {str(e)}\n")
        import traceback
        logger.error(f"Ошибка при обработке {file_path_obj}: {traceback.format_exc()}")


def main():
    """Основная функция"""
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python process_single_file.py <путь_к_файлу>")
        print("Пример: python process_single_file.py '../сертификаты/29-52 ПАСПОРТА .pdf'")
        return
    
    file_path = sys.argv[1]
    process_single_file(file_path)


if __name__ == "__main__":
    main()

