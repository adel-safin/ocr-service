"""OCR на основе PaddleOCR (CPU, lang=ru) через HTTP API в Docker.

Сервер: paddleocr-server (Docker). use_gpu=False, use_angle_cls=True, lang='ru'.
Клиент отправляет изображения на POST /v1/ocr, без локальных моделей.

Алгоритм: PDF → изображения (pdf2image) → каждая картинка на /v1/ocr → текст.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import requests
from pdf2image import convert_from_path, pdfinfo_from_path

import sys
app_root = Path(__file__).resolve().parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from config.settings import settings

logger = logging.getLogger(__name__)


class OCRServerUnavailable(Exception):
    """Сервер PaddleOCR (Docker) не отвечает на /health."""

# Таймаут на один снимок (CPU может быть медленным)
_HTTP_TIMEOUT = int(os.environ.get("PADDLEOCR_HTTP_TIMEOUT", "120"))
# Макс. сторона после ресайза. 1500 — баланс под 12–15 GB. При OOM: OCRM_MAX_SIDE=1200, OCRM_DPI=200 или OCRM_LOWMEM=1
_OCR_MAX_SIDE = int(os.environ.get("OCRM_MAX_SIDE", "1500"))


def _base_url() -> str:
    return getattr(settings, "PADDLEOCR_SERVER_URL", None) or os.environ.get(
        "PADDLEOCR_SERVER_URL", "http://127.0.0.1:8080"
    )


def _ocr_server_url() -> str:
    return f"{_base_url().rstrip('/')}/v1/ocr"


def check_ocr_server_available() -> bool:
    """Проверка доступности PaddleOCR: GET /health."""
    try:
        r = requests.get(f"{_base_url().rstrip('/')}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.warning("PaddleOCR /health недоступен: %s", e)
        return False


def _call_paddle_http(image_path: str) -> Dict[str, Any]:
    """POST изображения на PaddleOCR CPU /v1/ocr. Повторы при обрыве (Connection reset, RemoteDisconnected)."""
    import time as _time
    url = _ocr_server_url()
    empty = {"text": "", "confidence": 0.0, "detailed_data": {"text_regions": [], "total_regions": 0}, "word_count": 0}
    last_err = None
    for attempt in range(3):
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    url,
                    files={"file": (os.path.basename(image_path), f, "image/png")},
                    timeout=_HTTP_TIMEOUT,
                )
            r.raise_for_status()
            out = r.json()
            if not (out.get("text") or "").strip():
                logger.warning("PaddleOCR вернул пустой text. _debug=%s", out.get("_debug"))
            return out
        except requests.exceptions.RequestException as e:
            last_err = e
            body = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = (e.response.text or "")[:500]
                except Exception:
                    pass
            logger.warning("PaddleOCR HTTP (попытка %d/3) %s: %s. body=%s", attempt + 1, url, e, body)
            if attempt < 2:
                _time.sleep(5)
        except Exception as e:
            logger.exception("Ошибка при вызове PaddleOCR: %s", e)
            return empty
    if last_err is not None:
        logger.error("Ошибка PaddleOCR HTTP после 3 попыток: %s", last_err)
    return empty


class OCREngine:
    """OCR через PaddleOCR CPU (Docker): lang=ru, use_angle_cls. Один HTTP-запрос на снимок."""

    def __init__(self) -> None:
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Препроцессинг: масштаб, CLAHE для яркости/контраста, затем BGR для PaddleOCR."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        h, w = gray.shape[:2]
        if h < 800 or w < 800:
            scale = max(800 / h, 800 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape[:2]
        if max(h, w) > _OCR_MAX_SIDE:
            scale = _OCR_MAX_SIDE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # CLAHE — помогает при неоптимальной яркости; отключить: OCRM_SKIP_CLAHE=1
        if not os.environ.get("OCRM_SKIP_CLAHE"):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        if len(gray.shape) == 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray

    def _default_dpi(self) -> int:
        return int(os.environ.get("OCRM_DPI", "300"))

    def load_image(
        self, file_path: str, page: Optional[int] = None, dpi: Optional[int] = None
    ) -> Tuple[np.ndarray, str]:
        """Загрузка из PDF или файла (jpg, png, bmp). DPI для PDF: OCRM_DPI или 300."""
        if dpi is None:
            dpi = self._default_dpi()
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            if page is not None:
                images = convert_from_path(file_path, dpi=dpi, first_page=page, last_page=page)
            else:
                images = convert_from_path(file_path, dpi=dpi, first_page=1, last_page=1)
            if not images:
                raise ValueError(f"Не удалось конвертировать PDF: {file_path}")
            arr = np.array(images[0])
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), "pdf"
        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {file_path}")
            return img, "image"
        raise ValueError(f"Неподдерживаемый формат: {ext}")

    def get_pdf_page_count(self, file_path: str) -> int:
        try:
            info = pdfinfo_from_path(file_path)
            return int(info.get("Pages", 1))
        except Exception:
            return 1

    def extract_text(self, image: np.ndarray, config: Optional[str] = None) -> Dict[str, Any]:
        """Изображение → препроцессинг → временный файл → POST /v1/ocr."""
        processed = self.preprocess_image(image)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        try:
            os.close(fd)
            cv2.imwrite(tmp, processed)
            out = _call_paddle_http(tmp)
            return out
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def extract_text_by_area(
        self, image: np.ndarray, area: Dict[str, int], dpi: int = 900
    ) -> Dict[str, Any]:
        """Вырезанная область → временный файл → POST /v1/ocr. dpi оставлен для совместимости."""
        x1 = max(0, int(area.get("x1", 0)))
        y1 = max(0, int(area.get("y1", 0)))
        x2 = max(0, int(area.get("x2", 0)))
        y2 = max(0, int(area.get("y2", 0)))
        h, w = image.shape[:2]
        x1, x2 = min(x1, w), min(x2, w)
        y1, y2 = min(y1, h), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return {"text": "", "confidence": 0.0, "detailed_data": {"text_regions": [], "total_regions": 0}, "word_count": 0}
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return {"text": "", "confidence": 0.0, "detailed_data": {"text_regions": [], "total_regions": 0}, "word_count": 0}
        # Препроцессинг (ресайз при >OCRM_MAX_SIDE), чтобы не OOM на больших вырезках при 900 DPI
        crop = self.preprocess_image(crop)
        fd, tmp = tempfile.mkstemp(suffix=".png")
        try:
            os.close(fd)
            cv2.imwrite(tmp, crop)
            return _call_paddle_http(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def process_file_all_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """PDF → страницы в изображения → каждая на /v1/ocr → список результатов."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext != ".pdf":
            img, ft = self.load_image(file_path)
            res = self.extract_text(img)
            res["file_type"] = ft
            res["file_path"] = file_path
            res["page_number"] = 1
            res["total_pages"] = 1
            return [res]

        n = self.get_pdf_page_count(file_path)
        results: List[Dict[str, Any]] = []
        for p in range(1, n + 1):
            try:
                img, _ = self.load_image(file_path, page=p)
                res = self.extract_text(img)
                res["file_type"] = "pdf"
                res["file_path"] = file_path
                res["page_number"] = p
                res["total_pages"] = n
                results.append(res)
            except Exception as e:
                logger.error("Ошибка страницы %s файла %s: %s", p, file_path, e)
        return results

    def process_file(self, file_path: str) -> Dict[str, Any]:
        img, ft = self.load_image(file_path)
        res = self.extract_text(img)
        res["file_type"] = ft
        res["file_path"] = file_path
        return res
