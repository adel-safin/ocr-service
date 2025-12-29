#!/usr/bin/env python3
"""
Обработка сертификатов из папки "сертификаты"
Вывод полного текста по каждой фазе
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.WARNING,  # Уменьшаем логирование для чистоты вывода
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_certificates():
    """Обработка всех сертификатов"""
    print("=" * 100)
    print("ОБРАБОТКА СЕРТИФИКАТОВ")
    print("=" * 100)
    
    # Путь к папке с сертификатами
    certificates_dir = Path("../сертификаты")
    
    if not certificates_dir.exists():
        print(f"❌ Папка не найдена: {certificates_dir}")
        return
    
    # Поиск всех PDF файлов
    pdf_files = sorted(list(certificates_dir.glob("*.pdf")))
    
    if not pdf_files:
        print(f"❌ PDF файлы не найдены в {certificates_dir}")
        return
    
    print(f"\n📁 Найдено сертификатов: {len(pdf_files)}")
    print(f"📋 Обработка через все фазы...\n")
    
    # Создание пайплайнов для каждой фазы
    pipeline_phase1 = DocumentPipeline(use_ml=False, use_active_learning=False)
    pipeline_phase2 = DocumentPipeline(use_ml=True, use_active_learning=False)
    pipeline_phase3 = DocumentPipeline(use_ml=True, use_active_learning=True)
    
    results = []
    
    for i, cert_file in enumerate(pdf_files, 1):
        print("\n" + "=" * 100)
        print(f"СЕРТИФИКАТ {i}/{len(pdf_files)}: {cert_file.name}")
        print("=" * 100)
        
        cert_result = {
            'filename': cert_file.name,
            'phase1': None,
            'phase2': None,
            'phase3': None
        }
        
        try:
            # ФАЗА 1: Базовый OCR + правила
            print("\n" + "-" * 100)
            print("ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА")
            print("-" * 100)
            result1 = pipeline_phase1.process(str(cert_file), template="certificate")
            cert_result['phase1'] = result1
            
            print(f"✅ Document ID: {result1['document_id']}")
            print(f"📊 Качество: {result1['quality_report']['overall_quality']:.2%}")
            print(f"✏️  Исправлений: {len(result1.get('corrections_applied', []))}")
            # Сохранение текста Фазы 1 в файл
            output_dir = Path("data/outputs/certificates")
            output_dir.mkdir(parents=True, exist_ok=True)
            phase1_file = output_dir / f"{cert_file.stem}_phase1.txt"
            with open(phase1_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result1['document_id']}\n")
                f.write(f"Качество: {result1['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Исправлений: {len(result1.get('corrections_applied', []))}\n")
                f.write("\n" + "=" * 100 + "\n")
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result1['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            print(f"💾 Текст Фазы 1 сохранен: {phase1_file}")
            print(f"📄 Длина текста: {len(result1['extracted_data']['full_text'])} символов")
            
            # ФАЗА 2: С ML компонентами
            print("\n" + "-" * 100)
            print("ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ")
            print("-" * 100)
            result2 = pipeline_phase2.process(str(cert_file), template="certificate")
            cert_result['phase2'] = result2
            
            print(f"✅ Document ID: {result2['document_id']}")
            print(f"📊 Качество: {result2['quality_report']['overall_quality']:.2%}")
            print(f"✏️  Исправлений: {len(result2.get('corrections_applied', []))}")
            
            # Сохранение текста Фазы 2
            phase2_file = output_dir / f"{cert_file.stem}_phase2.txt"
            with open(phase2_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result2['document_id']}\n")
                f.write(f"Качество: {result2['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Исправлений: {len(result2.get('corrections_applied', []))}\n")
                f.write("\n" + "=" * 100 + "\n")
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result2['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            print(f"💾 Текст Фазы 2 сохранен: {phase2_file}")
            print(f"📄 Длина текста: {len(result2['extracted_data']['full_text'])} символов")
            
            # ФАЗА 3: С активным обучением
            print("\n" + "-" * 100)
            print("ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ")
            print("-" * 100)
            result3 = pipeline_phase3.process(str(cert_file), template="certificate")
            cert_result['phase3'] = result3
            
            print(f"✅ Document ID: {result3['document_id']}")
            print(f"📊 Качество: {result3['quality_report']['overall_quality']:.2%}")
            print(f"✏️  Исправлений: {len(result3.get('corrections_applied', []))}")
            
            # Сохранение текста Фазы 3
            phase3_file = output_dir / f"{cert_file.stem}_phase3.txt"
            with open(phase3_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result3['document_id']}\n")
                f.write(f"Качество: {result3['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Исправлений: {len(result3.get('corrections_applied', []))}\n")
                f.write("\n" + "=" * 100 + "\n")
                f.write("ПОЛНЫЙ ТЕКСТ:\n")
                f.write("=" * 100 + "\n")
                f.write(result3['extracted_data']['full_text'])
                f.write("\n" + "=" * 100 + "\n")
            
            print(f"💾 Текст Фазы 3 сохранен: {phase3_file}")
            print(f"📄 Длина текста: {len(result3['extracted_data']['full_text'])} символов")
            
            # Сравнение
            print("\n" + "-" * 100)
            print("СРАВНЕНИЕ ФАЗ")
            print("-" * 100)
            print(f"Фаза 1 - Качество: {result1['quality_report']['overall_quality']:.2%}, "
                  f"Исправлений: {len(result1.get('corrections_applied', []))}")
            print(f"Фаза 2 - Качество: {result2['quality_report']['overall_quality']:.2%}, "
                  f"Исправлений: {len(result2.get('corrections_applied', []))}")
            print(f"Фаза 3 - Качество: {result3['quality_report']['overall_quality']:.2%}, "
                  f"Исправлений: {len(result3.get('corrections_applied', []))}")
            
            results.append(cert_result)
            
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {cert_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Итоговая статистика
    print("\n" + "=" * 100)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 100)
    print(f"\n📊 Обработано сертификатов: {len(results)}")
    
    if results:
        avg_quality_phase1 = sum(r['phase1']['quality_report']['overall_quality'] 
                                for r in results if r['phase1']) / len(results)
        avg_quality_phase2 = sum(r['phase2']['quality_report']['overall_quality'] 
                                for r in results if r['phase2']) / len(results)
        avg_quality_phase3 = sum(r['phase3']['quality_report']['overall_quality'] 
                                for r in results if r['phase3']) / len(results)
        
        print(f"\n📈 Среднее качество:")
        print(f"   Фаза 1: {avg_quality_phase1:.2%}")
        print(f"   Фаза 2: {avg_quality_phase2:.2%}")
        print(f"   Фаза 3: {avg_quality_phase3:.2%}")
        
        total_corrections_phase1 = sum(len(r['phase1'].get('corrections_applied', [])) 
                                       for r in results if r['phase1'])
        total_corrections_phase2 = sum(len(r['phase2'].get('corrections_applied', [])) 
                                       for r in results if r['phase2'])
        total_corrections_phase3 = sum(len(r['phase3'].get('corrections_applied', [])) 
                                       for r in results if r['phase3'])
        
        print(f"\n✏️  Всего исправлений:")
        print(f"   Фаза 1: {total_corrections_phase1}")
        print(f"   Фаза 2: {total_corrections_phase2}")
        print(f"   Фаза 3: {total_corrections_phase3}")
    
    print("\n" + "=" * 100)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 100)
    
    return results


def main():
    """Основная функция"""
    results = process_certificates()
    
    # Сохранение результатов в файл
    if results:
        import json
        output_file = Path("data/outputs/certificates_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Упрощенная версия для сохранения (без полных текстов в JSON, они уже выведены)
        simplified_results = []
        for r in results:
            simplified_results.append({
                'filename': r['filename'],
                'phase1': {
                    'document_id': r['phase1']['document_id'],
                    'quality': r['phase1']['quality_report']['overall_quality'],
                    'corrections_count': len(r['phase1'].get('corrections_applied', [])),
                    'text_length': len(r['phase1']['extracted_data']['full_text'])
                },
                'phase2': {
                    'document_id': r['phase2']['document_id'],
                    'quality': r['phase2']['quality_report']['overall_quality'],
                    'corrections_count': len(r['phase2'].get('corrections_applied', [])),
                    'text_length': len(r['phase2']['extracted_data']['full_text'])
                },
                'phase3': {
                    'document_id': r['phase3']['document_id'],
                    'quality': r['phase3']['quality_report']['overall_quality'],
                    'corrections_count': len(r['phase3'].get('corrections_applied', [])),
                    'text_length': len(r['phase3']['extracted_data']['full_text'])
                }
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены: {output_file}")


if __name__ == "__main__":
    main()
