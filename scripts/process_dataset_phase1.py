#!/usr/bin/env python3
"""
Обработка всех пар документов из папки "Датасет" по Фазе 1
Сохранение результатов в папку data
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
from utils.dataset_loader import DatasetLoader
import logging

logging.basicConfig(
    level=logging.WARNING,  # Уменьшаем логирование
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_dataset_phase1():
    """Обработка всех пар документов по Фазе 1"""
    print("=" * 100)
    print("ОБРАБОТКА ДАТАСЕТА - ФАЗА 1")
    print("=" * 100)
    
    # Путь к датасету
    dataset_root = Path("../Датасет")
    
    if not dataset_root.exists():
        print(f"❌ Папка не найдена: {dataset_root}")
        return
    
    # Загрузка датасета
    loader = DatasetLoader(str(dataset_root))
    
    # Получение ВСЕХ изображений, не только пар
    documents_dir = dataset_root / "Наборы однотипных документов со сканами"
    
    if not documents_dir.exists():
        print(f"❌ Директория не найдена: {documents_dir}")
        return
    
    all_images = []
    doc_types = loader.get_all_document_types()
    
    print(f"\n📋 Найдено типов документов: {len(doc_types)}")
    
    # Собираем все изображения из всех папок
    for doc_type_dir in documents_dir.iterdir():
        if not doc_type_dir.is_dir():
            continue
        
        doc_type_name = doc_type_dir.name
        images_in_type = []
        
        for file in doc_type_dir.iterdir():
            if file.is_file() and file.suffix.lower() in {'.pdf', '.jpg', '.jpeg', '.png', '.bmp'}:
                # Проверяем, есть ли эталон
                reference_path = None
                base_name = file.stem
                
                # Ищем эталон с похожим именем
                for ref_file in doc_type_dir.iterdir():
                    if ref_file.is_file() and ref_file.suffix.lower() in {'.doc', '.docx', '.txt', '.xlsx'}:
                        # Проверяем совпадение базового имени
                        ref_base = loader._get_base_name(ref_file.stem)
                        img_base = loader._get_base_name(base_name)
                        
                        if ref_base == img_base or ref_file.stem == base_name:
                            reference_path = str(ref_file)
                            break
                
                all_images.append({
                    'document_type': doc_type_name,
                    'image_path': str(file),
                    'reference_path': reference_path,
                    'has_reference': reference_path is not None
                })
                images_in_type.append(file.name)
        
        print(f"   {doc_type_name}: {len(images_in_type)} изображений")
    
    print(f"\n📁 Всего изображений для обработки: {len(all_images)}")
    print(f"   С эталонами: {sum(1 for img in all_images if img['has_reference'])}")
    print(f"   Без эталонов: {sum(1 for img in all_images if not img['has_reference'])}")
    
    if not all_images:
        print("❌ Изображений не найдено")
        return
    
    all_pairs = all_images
    
    # Создание пайплайна Фазы 1
    print("\n🔧 Инициализация пайплайна Фазы 1...")
    pipeline = DocumentPipeline(use_ml=False, use_active_learning=False)
    
    # Создание папки для результатов
    output_dir = Path("data/dataset_results_phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    processed = 0
    errors = 0
    
    print(f"\n⏳ Начало обработки...\n")
    
    for i, pair in enumerate(all_pairs, 1):
        image_path = Path(pair['image_path'])
        doc_type = pair['document_type']
        
        print(f"[{i}/{len(all_pairs)}] {doc_type}: {image_path.name}")
        
        try:
            # Обработка документа
            image_path_str = pair['image_path']
            result = pipeline.process(image_path_str, template=doc_type.lower().replace(' ', '_'))
            
            # Сохранение полного текста в отдельный файл
            image_path_obj = Path(pair['image_path'])
            
            # Получаем информацию о страницах
            pages_data = result['extracted_data'].get('pages', [])
            total_pages = result['extracted_data'].get('total_pages', 1)
            
            # Сохраняем основной файл со всеми страницами
            text_file = output_dir / f"{image_path_obj.stem}_phase1.txt"
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"ТИП ДОКУМЕНТА: {doc_type}\n")
                f.write(f"ФАЙЛ: {image_path_obj.name}\n")
                f.write(f"Document ID: {result['document_id']}\n")
                f.write(f"Качество: {result['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Уверенность OCR: {result['quality_report']['ocr_confidence']:.2%}\n")
                f.write(f"Исправлений: {len(result.get('corrections_applied', []))}\n")
                f.write(f"Всего страниц: {total_pages}\n")
                
                # Добавление эталонного текста, если есть
                if pair.get('has_reference') and pair.get('reference_path'):
                    try:
                        reference_text = loader.load_reference_text(pair['reference_path'])
                        if reference_text.strip():
                            f.write("\n" + "=" * 100 + "\n")
                            f.write("ЭТАЛОННЫЙ ТЕКСТ:\n")
                            f.write("=" * 100 + "\n")
                            f.write(reference_text)
                            f.write("\n" + "=" * 100 + "\n")
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить эталон: {str(e)}")
                
                # Сохранение текста по страницам
                if pages_data and len(pages_data) > 1:
                    f.write("\n" + "=" * 100 + "\n")
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
                    f.write("\n" + "=" * 100 + "\n")
                    f.write("РАСПОЗНАННЫЙ ТЕКСТ:\n")
                    f.write("=" * 100 + "\n")
                    f.write(result['extracted_data']['full_text'])
                    f.write("\n" + "=" * 100 + "\n")
                
                # Добавление информации об исправлениях
                if result.get('corrections_applied'):
                    f.write("\nИСПРАВЛЕНИЯ:\n")
                    f.write("-" * 100 + "\n")
                    for j, correction in enumerate(result['corrections_applied'], 1):
                        f.write(f"{j}. {correction.get('from', '')} -> {correction.get('to', '')}\n")
            
            # Сохранение отдельных файлов для каждой страницы (если больше 1 страницы)
            if pages_data and len(pages_data) > 1:
                pages_dir = output_dir / f"{image_path_obj.stem}_pages"
                pages_dir.mkdir(exist_ok=True)
                
                for page_info in pages_data:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_file = pages_dir / f"page_{page_num:03d}.txt"
                    
                    with open(page_file, 'w', encoding='utf-8') as pf:
                        pf.write(f"ТИП ДОКУМЕНТА: {doc_type}\n")
                        pf.write(f"ФАЙЛ: {image_path_obj.name}\n")
                        pf.write(f"СТРАНИЦА: {page_num} из {total_pages}\n")
                        pf.write(f"Уверенность OCR: {page_info.get('confidence', 0.0):.2%}\n")
                        pf.write("\n" + "=" * 100 + "\n")
                        pf.write("ТЕКСТ СТРАНИЦЫ:\n")
                        pf.write("=" * 100 + "\n")
                        pf.write(page_text)
                        pf.write("\n" + "=" * 100 + "\n")
            
            # Сохранение структурированных данных
            result_data = {
                'document_type': doc_type,
                'filename': image_path.name,
                'document_id': result['document_id'],
                'quality': {
                    'overall': result['quality_report']['overall_quality'],
                    'ocr_confidence': result['quality_report']['ocr_confidence'],
                    'text_quality': result['quality_report'].get('text_quality', 0),
                    'image_quality': result['quality_report'].get('image_quality', 0)
                },
                'corrections_count': len(result.get('corrections_applied', [])),
                'text_length': len(result['extracted_data']['full_text']),
                'text_file': str(text_file.relative_to(Path('data'))),
                'extracted_fields': result['extracted_data'].get('fields', {}),
                'validation_results': result.get('validation_results', {}),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            results.append(result_data)
            processed += 1
            
            print(f"   ✅ Обработано: качество {result['quality_report']['overall_quality']:.2%}, "
                  f"исправлений: {len(result.get('corrections_applied', []))}, "
                  f"текст: {len(result['extracted_data']['full_text'])} символов")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}")
            errors += 1
            import traceback
            logger.error(f"Ошибка при обработке {image_path}: {traceback.format_exc()}")
            continue
    
    # Сохранение сводного отчета
    summary = {
        'total_pairs': len(all_pairs),
        'processed': processed,
        'errors': errors,
        'document_types': doc_types,
        'pairs_by_type': {doc_type: len(loader.find_document_pairs(doc_type)) 
                          for doc_type in doc_types},
        'results': results,
        'processing_timestamp': datetime.now().isoformat()
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Статистика
    if results:
        avg_quality = sum(r['quality']['overall'] for r in results) / len(results)
        avg_ocr_conf = sum(r['quality']['ocr_confidence'] for r in results) / len(results)
        total_corrections = sum(r['corrections_count'] for r in results)
        total_text_length = sum(r['text_length'] for r in results)
        
        print("\n" + "=" * 100)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 100)
        print(f"\n📊 Обработано: {processed}/{len(all_pairs)}")
        print(f"❌ Ошибок: {errors}")
        print(f"\n📈 Средние показатели:")
        print(f"   Качество: {avg_quality:.2%}")
        print(f"   Уверенность OCR: {avg_ocr_conf:.2%}")
        print(f"   Исправлений: {total_corrections} (в среднем {total_corrections/processed:.1f} на документ)")
        print(f"   Длина текста: {total_text_length} символов (в среднем {total_text_length/processed:.0f} на документ)")
        
        print(f"\n💾 Результаты сохранены:")
        print(f"   - Тексты: {output_dir}/")
        print(f"   - Сводка: {summary_file}")
    
    print("\n" + "=" * 100)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 100)
    
    return results


def main():
    """Основная функция"""
    results = process_dataset_phase1()
    
    if results:
        print(f"\n✅ Успешно обработано {len(results)} документов")
        print(f"📁 Результаты в папке: data/dataset_results_phase1/")


if __name__ == "__main__":
    main()
