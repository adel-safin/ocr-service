"""FastAPI приложение"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

import sys
from pathlib import Path

# Добавление корня приложения в путь
app_root = Path(__file__).parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from api.routes import router
from config.settings import settings

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Сервис обработки документов с OCR и автокоррекцией"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутеров
app.include_router(router, prefix=settings.API_PREFIX, tags=["OCR"])

# Подключение роутеров для обратной связи
from api.feedback_routes import router as feedback_router
app.include_router(feedback_router, prefix=settings.API_PREFIX, tags=["Feedback"])

# Подключение статических файлов для шаблонов
templates_dir = app_root / "templates"
if templates_dir.exists():
    app.mount("/static", StaticFiles(directory=str(templates_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Веб-интерфейс для загрузки файлов"""
    templates_dir = app_root / "templates"
    index_file = templates_dir / "index.html"
    
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """
        <html>
            <head><title>OCR Service</title></head>
            <body>
                <h1>OCR Document Processing Service</h1>
                <p>API доступен по адресу: <a href="/docs">/docs</a></p>
                <p>Веб-интерфейс не найден. Пожалуйста, создайте файл templates/index.html</p>
            </body>
        </html>
        """


@app.get("/api/info")
async def api_info():
    """Информация об API"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "running",
        "endpoints": {
            "process": f"{settings.API_PREFIX}/process",
            "batch_process": f"{settings.API_PREFIX}/batch_process",
            "confirm_correction": f"{settings.API_PREFIX}/confirm_correction",
            "corrections_db": f"{settings.API_PREFIX}/corrections_db",
            "health": f"{settings.API_PREFIX}/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
