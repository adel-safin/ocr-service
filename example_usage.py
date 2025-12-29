#!/usr/bin/env python3
"""
Пример использования OCR сервиса напрямую (без API)
"""
import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from core.processor import DocumentPipeline


def main():
    """Пример обработки документа"""
    
    # Создание пайплайна
    pipeline = DocumentPipeline()
    
    # Путь к документу (измените на свой)
    file_path = "../Датасет/Наборы однотипных документов со сканами/Акт АОСР/1 АОСР.pdf"
    
    if not Path(file_path).exists():
        print(f"❌ Файл не найден: {file_path}")
        print("💡 Укажите путь к существующему PDF файлу")
        return
    
    print(f"📄 Обработка документа: {file_path}\n")
    
    try:
        # Обработка документа
        result = pipeline.process(
            file_path=file_path,
            template="act_aosr",
            required_fields=["ogrn", "inn", "date"]
        )
        
        # Вывод результатов
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ ОБРАБОТКИ")
        print("=" * 60)
        print(f"Document ID: {result['document_id']}")
        print(f"Дата обработки: {result['processing_date']}")
        print(f"Качество: {result['quality_report']['overall_quality']:.2%}")
        print(f"Требует проверки: {'Да' if result['needs_review'] else 'Нет'}")
        
        print("\n" + "-" * 60)
        print("КРИТИЧЕСКИЕ ПОЛЯ")
        print("-" * 60)
        critical_fields = result['extracted_data']['critical_fields']
        for field_name, field_data in critical_fields.items():
            status = "✅" if field_data['valid'] else "❌"
            print(f"{status} {field_name.upper()}:")
            print(f"   Значение: {field_data['value']}")
            print(f"   Уверенность: {field_data['confidence']:.2%}")
            print(f"   Валидно: {field_data['valid']}")
            if field_data.get('suggested_correction'):
                print(f"   Предложенное исправление: {field_data['suggested_correction']}")
            print()
        
        print("-" * 60)
        print("ПРИМЕНЕННЫЕ ИСПРАВЛЕНИЯ")
        print("-" * 60)
        corrections = result.get('corrections_applied', [])
        if corrections:
            for i, correction in enumerate(corrections, 1):
                print(f"{i}. '{correction['from']}' -> '{correction['to']}' "
                      f"(уверенность: {correction['confidence']:.2%})")
        else:
            print("Исправления не применялись")
        
        print("\n" + "-" * 60)
        print("ПРОБЛЕМЫ КАЧЕСТВА")
        print("-" * 60)
        issues = result['quality_report'].get('issues', [])
        if issues:
            for issue in issues:
                print(f"⚠️  {issue.get('type', 'unknown')}: {issue.get('message', '')}")
        else:
            print("Проблем не обнаружено")
        
        print("\n" + "=" * 60)
        print("ПОЛНЫЙ ТЕКСТ (первые 500 символов)")
        print("=" * 60)
        full_text = result['extracted_data']['full_text']
        print(full_text[:500] + ("..." if len(full_text) > 500 else ""))
        
    except Exception as e:
        print(f"❌ Ошибка при обработке: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
