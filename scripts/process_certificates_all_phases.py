#!/usr/bin/env python3
"""
Обработка всех сертификатов через все три фазы
Сохранение результатов для каждой фазы
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
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_certificates_all_phases():
    """Обработка всех сертификатов через все фазы"""
    print("=" * 100)
    print("ОБРАБОТКА СЕРТИФИКАТОВ - ВСЕ ФАЗЫ")
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
    print("🔧 Инициализация пайплайнов...")
    pipeline_phase1 = DocumentPipeline(use_ml=False, use_active_learning=False)
    pipeline_phase2 = DocumentPipeline(use_ml=True, use_active_learning=False)
    pipeline_phase3 = DocumentPipeline(use_ml=True, use_active_learning=True)
    
    # Создание папок для результатов
    output_dir = Path("data/certificates_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phase1_dir = output_dir / "phase1"
    phase2_dir = output_dir / "phase2"
    phase3_dir = output_dir / "phase3"
    
    phase1_dir.mkdir(exist_ok=True)
    phase2_dir.mkdir(exist_ok=True)
    phase3_dir.mkdir(exist_ok=True)
    
    results = []
    processed = 0
    errors = 0
    
    print(f"\n⏳ Начало обработки...\n")
    
    for i, cert_file in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {cert_file.name}")
        
        cert_result = {
            'filename': cert_file.name,
            'phase1': None,
            'phase2': None,
            'phase3': None
        }
        
        try:
            # ФАЗА 1: Базовый OCR + правила
            print(f"   📌 Фаза 1...", end=" ")
            result1 = pipeline_phase1.process(str(cert_file), template="certificate")
            cert_result['phase1'] = result1
            
            # Сохранение текста Фазы 1
            pages_data = result1['extracted_data'].get('pages', [])
            total_pages = result1['extracted_data'].get('total_pages', 1)
            
            phase1_file = phase1_dir / f"{cert_file.stem}_phase1.txt"
            with open(phase1_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 1: БАЗОВЫЙ OCR + ПРАВИЛА\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result1['document_id']}\n")
                f.write(f"Качество: {result1['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Уверенность OCR: {result1['quality_report']['ocr_confidence']:.2%}\n")
                f.write(f"Исправлений: {len(result1.get('corrections_applied', []))}\n")
                f.write(f"Всего страниц: {total_pages}\n")
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
            
            # Сохранение отдельных файлов для каждой страницы (если больше 1 страницы)
            if pages_data and len(pages_data) > 1:
                pages_dir = phase1_dir / f"{cert_file.stem}_pages"
                pages_dir.mkdir(exist_ok=True)
                
                for page_info in pages_data:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_file = pages_dir / f"page_{page_num:03d}.txt"
                    
                    with open(page_file, 'w', encoding='utf-8') as pf:
                        pf.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
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
            result2 = pipeline_phase2.process(str(cert_file), template="certificate")
            cert_result['phase2'] = result2
            
            # Сохранение текста Фазы 2
            pages_data2 = result2['extracted_data'].get('pages', [])
            total_pages2 = result2['extracted_data'].get('total_pages', 1)
            
            phase2_file = phase2_dir / f"{cert_file.stem}_phase2.txt"
            with open(phase2_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 2: МАШИННОЕ ОБУЧЕНИЕ\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result2['document_id']}\n")
                f.write(f"Качество: {result2['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Уверенность OCR: {result2['quality_report']['ocr_confidence']:.2%}\n")
                f.write(f"Исправлений: {len(result2.get('corrections_applied', []))}\n")
                f.write(f"Всего страниц: {total_pages2}\n")
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
            
            # Сохранение отдельных файлов для каждой страницы (если больше 1 страницы)
            if pages_data2 and len(pages_data2) > 1:
                pages_dir = phase2_dir / f"{cert_file.stem}_pages"
                pages_dir.mkdir(exist_ok=True)
                
                for page_info in pages_data2:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_file = pages_dir / f"page_{page_num:03d}.txt"
                    
                    with open(page_file, 'w', encoding='utf-8') as pf:
                        pf.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
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
            result3 = pipeline_phase3.process(str(cert_file), template="certificate")
            cert_result['phase3'] = result3
            
            # Сохранение текста Фазы 3
            pages_data3 = result3['extracted_data'].get('pages', [])
            total_pages3 = result3['extracted_data'].get('total_pages', 1)
            
            phase3_file = phase3_dir / f"{cert_file.stem}_phase3.txt"
            with open(phase3_file, 'w', encoding='utf-8') as f:
                f.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                f.write(f"ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ\n")
                f.write("=" * 100 + "\n")
                f.write(f"Document ID: {result3['document_id']}\n")
                f.write(f"Качество: {result3['quality_report']['overall_quality']:.2%}\n")
                f.write(f"Уверенность OCR: {result3['quality_report']['ocr_confidence']:.2%}\n")
                f.write(f"Исправлений: {len(result3.get('corrections_applied', []))}\n")
                f.write(f"Всего страниц: {total_pages3}\n")
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
            
            # Сохранение отдельных файлов для каждой страницы (если больше 1 страницы)
            if pages_data3 and len(pages_data3) > 1:
                pages_dir = phase3_dir / f"{cert_file.stem}_pages"
                pages_dir.mkdir(exist_ok=True)
                
                for page_info in pages_data3:
                    page_num = page_info.get('page_number', 1)
                    page_text = page_info.get('text', '')
                    page_file = pages_dir / f"page_{page_num:03d}.txt"
                    
                    with open(page_file, 'w', encoding='utf-8') as pf:
                        pf.write(f"СЕРТИФИКАТ: {cert_file.name}\n")
                        pf.write(f"ФАЗА 3: АКТИВНОЕ ОБУЧЕНИЕ\n")
                        pf.write(f"СТРАНИЦА: {page_num} из {total_pages3}\n")
                        pf.write(f"Уверенность OCR: {page_info.get('confidence', 0.0):.2%}\n")
                        pf.write("\n" + "=" * 100 + "\n")
                        pf.write("ТЕКСТ СТРАНИЦЫ:\n")
                        pf.write("=" * 100 + "\n")
                        pf.write(page_text)
                        pf.write("\n" + "=" * 100 + "\n")
            
            print(f"✅ Качество: {result3['quality_report']['overall_quality']:.2%}")
            
            # Сохранение структурированных данных
            result_data = {
                'filename': cert_file.name,
                'phase1': {
                    'document_id': result1['document_id'],
                    'quality': result1['quality_report']['overall_quality'],
                    'ocr_confidence': result1['quality_report']['ocr_confidence'],
                    'corrections_count': len(result1.get('corrections_applied', [])),
                    'text_length': len(result1['extracted_data']['full_text']),
                    'text_file': str(phase1_file.relative_to(Path('data')))
                },
                'phase2': {
                    'document_id': result2['document_id'],
                    'quality': result2['quality_report']['overall_quality'],
                    'ocr_confidence': result2['quality_report']['ocr_confidence'],
                    'corrections_count': len(result2.get('corrections_applied', [])),
                    'text_length': len(result2['extracted_data']['full_text']),
                    'text_file': str(phase2_file.relative_to(Path('data')))
                },
                'phase3': {
                    'document_id': result3['document_id'],
                    'quality': result3['quality_report']['overall_quality'],
                    'ocr_confidence': result3['quality_report']['ocr_confidence'],
                    'corrections_count': len(result3.get('corrections_applied', [])),
                    'text_length': len(result3['extracted_data']['full_text']),
                    'text_file': str(phase3_file.relative_to(Path('data')))
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            results.append(result_data)
            processed += 1
            
            print(f"   ✅ Завершено\n")
            
        except Exception as e:
            print(f"   ❌ Ошибка: {str(e)}\n")
            errors += 1
            import traceback
            logger.error(f"Ошибка при обработке {cert_file}: {traceback.format_exc()}")
            continue
    
    # Сохранение сводного отчета
    summary = {
        'total_certificates': len(pdf_files),
        'processed': processed,
        'errors': errors,
        'results': results,
        'statistics': {},
        'processing_timestamp': datetime.now().isoformat()
    }
    
    # Статистика
    if results:
        # Фаза 1
        phase1_qualities = [r['phase1']['quality'] for r in results]
        phase1_ocr_conf = [r['phase1']['ocr_confidence'] for r in results]
        phase1_corrections = [r['phase1']['corrections_count'] for r in results]
        
        # Фаза 2
        phase2_qualities = [r['phase2']['quality'] for r in results]
        phase2_ocr_conf = [r['phase2']['ocr_confidence'] for r in results]
        phase2_corrections = [r['phase2']['corrections_count'] for r in results]
        
        # Фаза 3
        phase3_qualities = [r['phase3']['quality'] for r in results]
        phase3_ocr_conf = [r['phase3']['ocr_confidence'] for r in results]
        phase3_corrections = [r['phase3']['corrections_count'] for r in results]
        
        summary['statistics'] = {
            'phase1': {
                'avg_quality': sum(phase1_qualities) / len(phase1_qualities),
                'avg_ocr_confidence': sum(phase1_ocr_conf) / len(phase1_ocr_conf),
                'total_corrections': sum(phase1_corrections),
                'avg_corrections': sum(phase1_corrections) / len(phase1_corrections)
            },
            'phase2': {
                'avg_quality': sum(phase2_qualities) / len(phase2_qualities),
                'avg_ocr_confidence': sum(phase2_ocr_conf) / len(phase2_ocr_conf),
                'total_corrections': sum(phase2_corrections),
                'avg_corrections': sum(phase2_corrections) / len(phase2_corrections)
            },
            'phase3': {
                'avg_quality': sum(phase3_qualities) / len(phase3_qualities),
                'avg_ocr_confidence': sum(phase3_ocr_conf) / len(phase3_ocr_conf),
                'total_corrections': sum(phase3_corrections),
                'avg_corrections': sum(phase3_corrections) / len(phase3_corrections)
            }
        }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Вывод статистики
    print("\n" + "=" * 100)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 100)
    print(f"\n📊 Обработано: {processed}/{len(pdf_files)}")
    print(f"❌ Ошибок: {errors}")
    
    if summary['statistics']:
        stats = summary['statistics']
        print(f"\n📈 ФАЗА 1 (Базовый OCR + правила):")
        print(f"   Среднее качество: {stats['phase1']['avg_quality']:.2%}")
        print(f"   Средняя уверенность OCR: {stats['phase1']['avg_ocr_confidence']:.2%}")
        print(f"   Всего исправлений: {stats['phase1']['total_corrections']} (в среднем {stats['phase1']['avg_corrections']:.1f})")
        
        print(f"\n🤖 ФАЗА 2 (Машинное обучение):")
        print(f"   Среднее качество: {stats['phase2']['avg_quality']:.2%}")
        print(f"   Средняя уверенность OCR: {stats['phase2']['avg_ocr_confidence']:.2%}")
        print(f"   Всего исправлений: {stats['phase2']['total_corrections']} (в среднем {stats['phase2']['avg_corrections']:.1f})")
        
        print(f"\n🔄 ФАЗА 3 (Активное обучение):")
        print(f"   Среднее качество: {stats['phase3']['avg_quality']:.2%}")
        print(f"   Средняя уверенность OCR: {stats['phase3']['avg_ocr_confidence']:.2%}")
        print(f"   Всего исправлений: {stats['phase3']['total_corrections']} (в среднем {stats['phase3']['avg_corrections']:.1f})")
        
        # Сравнение
        print(f"\n📊 СРАВНЕНИЕ ФАЗ:")
        print(f"   Качество: Фаза 1: {stats['phase1']['avg_quality']:.2%} | Фаза 2: {stats['phase2']['avg_quality']:.2%} | Фаза 3: {stats['phase3']['avg_quality']:.2%}")
        print(f"   Исправлений: Фаза 1: {stats['phase1']['total_corrections']} | Фаза 2: {stats['phase2']['total_corrections']} | Фаза 3: {stats['phase3']['total_corrections']}")
    
    print(f"\n💾 Результаты сохранены:")
    print(f"   - Фаза 1: {phase1_dir}/")
    print(f"   - Фаза 2: {phase2_dir}/")
    print(f"   - Фаза 3: {phase3_dir}/")
    print(f"   - Сводка: {summary_file}")
    
    print("\n" + "=" * 100)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 100)
    
    return results


def main():
    """Основная функция"""
    results = process_certificates_all_phases()
    
    if results:
        print(f"\n✅ Успешно обработано {len(results)} сертификатов")
        print(f"📁 Результаты в папке: data/certificates_results/")


if __name__ == "__main__":
    main()

