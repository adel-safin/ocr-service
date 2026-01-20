#!/usr/bin/env python3
"""
Извлечение пар (ошибка OCR → эталон) из датасета OCR Arena (PDF + .txt).
Нужен запущенный PaddleOCR (./scripts/start_paddleocr_server.sh) и app.

Использование:
  cd app && python scripts/build_corrections_from_dataset.py [--max-docs N] [--apply]
  --max-docs N   обработать не более N документов (по умолчанию все)
  --apply        добавить предложенные пары в data/corrections.json (интерактивно или все)
  --merge-all    при --apply: добавить все предложенные без запроса
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

app_root = Path(__file__).resolve().parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from config.settings import settings

# OCR Arena: проект/Датасет/Распознавание документов OCR Arena
ROOT = Path(__file__).resolve().parent.parent.parent
OCR_ARENA = ROOT / "Датасет" / "Распознавание документов OCR Arena"
CORRECTIONS_PATH = Path(settings.CORRECTIONS_DB)


def find_pdf_txt_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    if not OCR_ARENA.exists():
        return pairs
    for cat in OCR_ARENA.iterdir():
        if not cat.is_dir():
            continue
        for p in cat.glob("*.pdf"):
            txt = p.with_suffix(".txt")
            if txt.exists():
                pairs.append((p, txt))
    return sorted(pairs)


def load_reference(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8")


def _collect_replace_pairs(ocr_line: str, gt_line: str, min_sim: float = 0.4) -> list[tuple[str, str]]:
    out = []
    sm = SequenceMatcher(None, ocr_line, gt_line)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        a, b = ocr_line[i1:i2].strip(), gt_line[j1:j2].strip()
        if not a or not b or a == b:
            continue
        if len(a) > 80 or len(b) > 80:
            continue
        if a.isdigit() and b.isdigit():
            continue
        if SequenceMatcher(None, a, b).ratio() < min_sim:
            continue
        out.append((a, b))
    return out


def align_and_extract(ocr_text: str, gt_text: str) -> list[tuple[str, str]]:
    ocr_lines = [s.strip() for s in ocr_text.splitlines() if s.strip()]
    gt_lines = [s.strip() for s in gt_text.splitlines() if s.strip()]
    pairs = []
    for ol in ocr_lines:
        best, best_r = "", 0.0
        for gl in gt_lines:
            r = SequenceMatcher(None, ol, gl).ratio()
            if r > best_r:
                best_r, best = r, gl
        if best_r >= 0.5 and best:
            pairs.extend(_collect_replace_pairs(ol, best))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--max-docs", type=int, default=0, help="Макс. документов (0=все)")
    ap.add_argument("--apply", action="store_true", help="Добавить в corrections.json")
    ap.add_argument("--merge-all", action="store_true", help="При --apply: добавить все без запроса")
    args = ap.parse_args()

    pairs = find_pdf_txt_pairs()
    if not pairs:
        print("Пар PDF+.txt не найдено в", OCR_ARENA)
        sys.exit(1)

    if args.max_docs and args.max_docs > 0:
        pairs = pairs[: args.max_docs]
    print("Найдено пар:", len(pairs))

    os.environ.setdefault("OCRM_MAX_SIDE", "1500")
    os.environ.setdefault("OCRM_DPI", "300")

    try:
        from core.ocr_engine import OCREngine
        from core.ocr_engine import check_ocr_server_available

        if not check_ocr_server_available():
            print("Сервер PaddleOCR недоступен. Запустите: ./scripts/start_paddleocr_server.sh")
            sys.exit(1)
        ocr_engine = OCREngine()
    except Exception as e:
        print("Ошибка инициализации OCR:", e)
        sys.exit(1)

    collected: Counter[tuple[str, str]] = Counter()
    for i, (pdf_path, txt_path) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {pdf_path.name}...", end=" ", flush=True)
        try:
            ref = load_reference(txt_path)
            res = ocr_engine.process_file(str(pdf_path))
            ocr = res.get("text") or ""
            for a, b in align_and_extract(ocr, ref):
                collected[(a, b)] += 1
            print("ok")
        except Exception as e:
            print("err:", e)

    if not collected:
        print("Пар для добавления не найдено.")
        return

    # Свёртка: (ocr, gt) -> gt; при конфликтах (один ocr → разный gt) берём самый частый
    by_ocr: dict[str, list[tuple[str, int]]] = {}
    for (a, b), c in collected.items():
        by_ocr.setdefault(a, []).append((b, c))
    suggestions = []
    for ocr, candidates in by_ocr.items():
        candidates.sort(key=lambda x: -x[1])
        gt = candidates[0][0]
        if gt == ocr:
            continue
        suggestions.append({"original": ocr, "corrected": gt, "count": candidates[0][1]})

    suggestions.sort(key=lambda x: -x["count"])
    print("\nПредложенные замены (топ-50):")
    for s in suggestions[:50]:
        print(f"  \"{s['original']}\" → \"{s['corrected']}\" (встреч: {s['count']})")

    out_json = Path(settings.DATA_DIR) / "suggested_corrections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)
    print("\nСохранено:", out_json)

    if args.apply and suggestions:
        db_path = CORRECTIONS_PATH
        db = {}
        if db_path.exists():
            with open(db_path, "r", encoding="utf-8") as f:
                db = json.load(f)
        to_add = suggestions
        if not args.merge_all:
            print("\nДобавить все (a) или выбрать (c)? [a/c/N]: ", end="")
            choice = input().strip().lower()
            if choice == "n" or not choice:
                return
            if choice == "c":
                to_add = [s for s in suggestions if input(f"  {s['original']} → {s['corrected']}? [y/N]: ").strip().lower() == "y"]
        for s in to_add:
            o, c = s["original"], s["corrected"]
            if o and o not in db:
                db[o] = c
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        print("Обновлено:", db_path)


if __name__ == "__main__":
    main()
