# OCR Document Processing Service

Сервис для обработки документов с использованием OCR (Tesseract) и автоматической коррекции ошибок распознавания.

## Архитектура

Система реализована в три фазы:

### Фаза 1: Базовый OCR + правила ✅

- Фиксированные шаблоны документов
- Регулярные выражения для критических полей
- Простые автозамены
- FastAPI сервис

### Фаза 2: Машинное обучение ✅

- Классификатор типов документов (CNN)
- Модель исправления опечаток (Transformer)
- ML детекция качества изображения
- Утилита для работы с датасетом

### Фаза 3: Активное обучение ✅

- Feedback loop от пользователей
- Автоматическое обновление базы исправлений
- Адаптация под новые типы документов
- Анализ паттернов и рекомендации

## Установка

### Локальная установка

1. Установите системные зависимости:

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-rus tesseract-ocr-script-Cyrl poppler-utils
```

2. Установите Python зависимости:

```bash
cd app
pip install -r requirements.txt
```

3. Настройте конфигурацию (опционально):

```bash
cp .env.example .env
# Отредактируйте .env при необходимости
```

### Docker установка

```bash
docker-compose up -d
```

## Использование

### Запуск сервиса

```bash
# Локально
cd app
uvicorn api.main:app --reload

# Docker
docker-compose up
```

Сервис будет доступен по адресу: http://localhost:8000

### API Endpoints

#### 1. Обработка одного документа

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -F "file=@document.pdf" \
  -F "template=act_aosr" \
  -F "required_fields=ogrn,inn,date"
```

#### 2. Пакетная обработка

```bash
curl -X POST "http://localhost:8000/api/v1/batch_process" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "template=act_aosr"
```

#### 3. Подтверждение исправления

```bash
curl -X POST "http://localhost:8000/api/v1/confirm_correction" \
  -H "Content-Type: application/json" \
  -d '{
    "original": "Маркуталь",
    "corrected": "Мариуполь",
    "add_to_db": true
  }'
```

#### 4. Получение базы исправлений

```bash
curl "http://localhost:8000/api/v1/corrections_db"
```

#### 5. Проверка здоровья

```bash
curl "http://localhost:8000/api/v1/health"
```

## Структура проекта

```
app/
├── api/              # FastAPI endpoints
├── core/             # Основная логика (OCR, валидация, коррекция)
├── services/         # Вспомогательные сервисы
├── config/           # Конфигурация
├── data/             # Данные (шаблоны, исправления, результаты)
└── models/           # ML модели (для Фазы 2)
```

## Поддерживаемые форматы

- PDF
- JPG/JPEG
- PNG
- BMP

## Поддерживаемые поля

- ОГРН (13 или 15 цифр)
- ИНН (10 или 12 цифр)
- КПП (9 цифр)
- Дата
- СНИЛС
- Номер сертификата ЕАЭС
- Телефон
- Email

## Разработка

### Добавление нового типа поля

Отредактируйте `core/validators.py` и добавьте новый паттерн в `FieldValidator.PATTERNS`.

### Добавление автозамены

Используйте API endpoint `/confirm_correction` или отредактируйте `data/corrections.json`.

### Активное обучение

Система автоматически собирает данные при обработке документов. Для анализа используйте:

```bash
python scripts/analyze_feedback.py
```

## Дополнительная документация

- `PHASE2_README.md` - Документация по машинному обучению
- `PHASE3_README.md` - Документация по активному обучению
- `../COMPLETE_SYSTEM.md` - Полное описание системы

## Лицензия

MIT
