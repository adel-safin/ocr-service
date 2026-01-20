#!/usr/bin/env python3
"""
Прогон по датасету: Фаза 1 → 2 → 3, сравнение с эталоном.

Фазы (без обучения от эталона прирост мал):
  Фаза 1 — OCR + правила (0/8 в словах, corrections.json, фильтр шума).
  Фаза 2 — ещё 3→з, 6→б в словах + T5 по абзацам (выкл: USE_T5=0).
  Фаза 3 — то же + правки из feedback (UI или --learn-from-ideal в этом/прошлых прогонах).

Обучение: с --learn-from-ideal после Фазы 1 из (raw ocr, эталон) извлекаются пары
(ошибка→правка) в feedback; reload до Фазы 3 — она применит их в этом же прогоне.

Использование:
  run_dataset_phases_by_pair.py [--max N] [--start N] [--only-arena] [--only-phase1] [--learn-from-ideal]
  --max N         макс. пар (0=все)
  --start N       с пары № N (1-based)
  --only-arena    только OCR Arena
  --only-phase1   только Фаза 1 (один OCR на документ, без 2/3)
  --learn-from-ideal  после Фазы 1: (ocr, эталон) → пары в feedback для Фазы 3 в след. прогонах
  --restart-ocr-between  перезапуск PaddleOCR между парами

OCRM_MAX_SIDE=1500, OCRM_DPI=300. При OOM: OCRM_LOWMEM=1. USE_T5=0 — выключить T5.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

app_root = Path(__file__).resolve().parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

ROOT = app_root.parent
DATASET = ROOT / "Датасет"
ARENA = DATASET / "Распознавание документов OCR Arena"
NABORY = DATASET / "Наборы однотипных документов со сканами"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _load_ref(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8")
    if ext in (".docx", ".doc"):
        try:
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text or "" for p in d.paragraphs)
        except Exception as e:
            return f"[Ошибка чтения {path}: {e}]"
    if ext == ".xlsx":
        try:
            import pandas as pd
            df = pd.read_excel(str(path), sheet_name=0, header=None)
            return df.to_string()
        except Exception as e:
            return f"[Ошибка чтения xlsx {path}: {e}]"
    return ""


def _find_pairs(only_arena: bool) -> list[tuple[Path, Path, str]]:
    out: list[tuple[Path, Path, str]] = []

    # OCR Arena: PDF + .txt с тем же stem
    if ARENA.exists():
        for cat in sorted(ARENA.iterdir()):
            if not cat.is_dir() or cat.name.startswith("."):
                continue
            for p in sorted(cat.glob("*.pdf")):
                txt = p.with_suffix(".txt")
                if txt.exists():
                    out.append((p, txt, f"arena_{cat.name}"))

    if only_arena:
        return out

    # Наборы: PDF + docx/doc/txt (по stem, с нормализацией)
    if NABORY.exists():
        for cat in sorted(NABORY.iterdir()):
            if not cat.is_dir() or cat.name.startswith("."):
                continue
            refs = {_norm(r.stem): r for r in cat.glob("*.docx")}
            for r in list(cat.glob("*.doc")) + list(cat.glob("*.txt")) + list(cat.glob("*.xlsx")):
                if _norm(r.stem) not in refs:
                    refs[_norm(r.stem)] = r
            for p in sorted(cat.glob("*.pdf")):
                r = refs.get(_norm(p.stem))
                if not r:
                    r = p.with_suffix(".docx")
                    if not r.exists():
                        r = p.with_suffix(".doc")
                    if not r.exists():
                        r = p.with_suffix(".txt")
                    if not r.exists():
                        continue
                out.append((p, Path(r), f"nabory_{cat.name}"))

    return out


def _is_ocr_connection_error(e: BaseException) -> bool:
    from core.ocr_engine import OCRServerUnavailable
    if isinstance(e, OCRServerUnavailable):
        return True
    s = str(e)
    return (
        "PaddleOCR" in s or "недоступен" in s or "Connection" in s
        or "RemoteDisconnected" in s or "ConnectionResetError" in s or "Connection aborted" in s
    )


def _restart_paddleocr_and_wait() -> bool:
    from core.ocr_engine import check_ocr_server_available
    subprocess.run(["docker", "restart", "paddleocr-local"], check=False, capture_output=True)
    for _ in range(45):
        time.sleep(2)
        if check_ocr_server_available():
            return True
    return False


def _learn_pairs_from_ideal(text: str, ideal: str, doc_id: str, fc, max_pairs: int = 40) -> int:
    """Выравнивает text и ideal, извлекает пары (ошибка, правка), добавляет в feedback. Возвращает количество добавленных."""
    if not text or not ideal:
        return 0
    sm = SequenceMatcher(None, text, ideal)
    seen: set[tuple[str, str]] = set()
    added = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        a, b = text[i1:i2].strip(), ideal[j1:j2].strip()
        if not a or not b or a == b:
            continue
        if (a, b) in seen:
            continue
        if not (2 <= len(a) <= 60 and 2 <= len(b) <= 60):
            continue
        if a.isspace() or b.isspace():
            continue
        seen.add((a, b))
        try:
            fc.add_correction_feedback(
                original=a, corrected=b, document_id=doc_id,
                context="learn_from_ideal", confidence=0.5
            )
            added += 1
        except Exception:
            pass
        if added >= max_pairs:
            break
    return added


def main() -> None:
    os.environ.setdefault("OCRM_MAX_SIDE", "1500")
    os.environ.setdefault("OCRM_DPI", "300")
    if os.environ.get("OCRM_LOWMEM"):
        os.environ["OCRM_MAX_SIDE"] = "1200"
        os.environ["OCRM_DPI"] = "200"

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--max", type=int, default=0, help="Макс. пар (0=все)")
    ap.add_argument("--start", type=int, default=1, help="С какой пары начать (1-based)")
    ap.add_argument("--only-arena", action="store_true", help="Только OCR Arena")
    ap.add_argument("--only-phase1", action="store_true", help="Только Фаза 1 (1 OCR на документ)")
    ap.add_argument("--learn-from-ideal", action="store_true",
                    help="После Фазы 1: (raw ocr, эталон) → feedback; Фаза 3 в этом прогоне применит")
    ap.add_argument("--restart-ocr-between", action="store_true", help="Перезапуск PaddleOCR между парами")
    args = ap.parse_args()

    pairs = _find_pairs(args.only_arena)
    if not pairs:
        print("Пар не найдено в", DATASET)
        sys.exit(1)

    start = max(1, args.start)
    end = start + args.max - 1 if args.max > 0 else len(pairs)
    pairs = pairs[start - 1 : end]
    total = len(_find_pairs(args.only_arena))

    print("Пар всего:", total, "| обрабатываем:", len(pairs), f"(с #{start})")
    out_dir = (ROOT / "data" / "phase_run").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Результаты:", out_dir)
    print()

    from core.processor import DocumentPipeline
    from core.ocr_engine import check_ocr_server_available

    if not check_ocr_server_available():
        print("Сервер PaddleOCR недоступен. Запустите: ./scripts/start_paddleocr_server.sh")
        sys.exit(1)

    feedback_collector = None
    if args.learn_from_ideal:
        from services.feedback_collector import FeedbackCollector
        feedback_collector = FeedbackCollector()
        print("Режим --learn-from-ideal: (raw ocr, эталон) → feedback; Фаза 3 применит их в этом же прогоне.")
        print()

    pipeline1 = DocumentPipeline(use_ml=False, use_active_learning=False)
    phases = [("Фаза 1", pipeline1, "1")]
    if not args.only_phase1:
        phases += [
            ("Фаза 2", DocumentPipeline(use_ml=True, use_active_learning=False), "2"),
            ("Фаза 3", DocumentPipeline(use_ml=True, use_active_learning=True), "3"),
        ]

    for i, (pdf_path, ref_path, tag) in enumerate(pairs, start=start):
        stem = re.sub(r'[^\w\s\-]', '', f"{tag}_{pdf_path.stem}").replace(" ", "_").strip("_") or "doc"
        print("=" * 70)
        print(f"  ПАРА #{i} / {total}: {tag} — {pdf_path.name}")
        print("  Эталон:", ref_path.name)
        print("=" * 70)

        ideal = _load_ref(ref_path).strip()
        (out_dir / f"{stem}_ideal.txt").write_text(ideal, encoding="utf-8")

        r = None
        text1 = ""
        for name, pl, ph in phases:
            try:
                r = pl.process(str(pdf_path))
                text = (r.get("extracted_data") or {}).get("full_text") or ""
            except Exception as e:
                if _is_ocr_connection_error(e):
                    print(f"  [!] {name}: обрыв связи, перезапуск PaddleOCR и повтор...")
                    if _restart_paddleocr_and_wait():
                        try:
                            r = pl.process(str(pdf_path))
                            text = (r.get("extracted_data") or {}).get("full_text") or ""
                        except Exception as e2:
                            r = None
                            text = f"[Ошибка: {e2}]"
                    else:
                        r = None
                        text = f"[Ошибка: {e}] (PaddleOCR не поднялся после перезапуска)"
                else:
                    r = None
                    text = f"[Ошибка: {e}]"
            (out_dir / f"{stem}_phase{ph}.txt").write_text(text, encoding="utf-8")
            match = SequenceMatcher(None, _norm(text), _norm(ideal)).ratio() if (text and ideal) else 0.0
            print(f"  {name}: {len(text)} симв.  соотв. эталону: {match:.0%}")

            if ph == "1" and args.learn_from_ideal and feedback_collector and r:
                raw = (r.get("extracted_data") or {}).get("raw_text") or ""
                n = _learn_pairs_from_ideal(raw, ideal, stem, feedback_collector)
                if n:
                    print(f"  [learn-from-ideal] добавлено {n} пар в feedback (Фаза 3 применит в этом прогоне).")
                if len(phases) >= 3:
                    phases[2][1].active_learning.feedback_collector.reload_from_disk()

        print()
        print("  --- ИДЕАЛ (первые 350 симв.) ---")
        print(ideal[:350] + ("…" if len(ideal) > 350 else ""))
        print()
        print("  --- ФАЗА 1 (первые 350 симв.) ---")
        t1 = (out_dir / f"{stem}_phase1.txt").read_text(encoding="utf-8")
        print(t1[:350] + ("…" if len(t1) > 350 else ""))
        print()
        fase_files = ", ".join(f"{stem}_phase{p}.txt" for (_, _, p) in phases) + f", {stem}_ideal.txt"
        print("  Файлы:", fase_files)

        if i < len(pairs):
            try:
                key = input("\n  [Enter] — следующий документ, [q] — выход: ").strip().lower()
            except EOFError:
                key = ""
            if key == "q":
                print("Выход.")
                break
            if args.restart_ocr_between:
                print("  Перезапуск PaddleOCR...")
                if _restart_paddleocr_and_wait():
                    print("  PaddleOCR снова доступен.")
                else:
                    print("  [!] PaddleOCR не поднялся за 90 с, следующие пары могут падать.")

    n_files = len(list(out_dir.glob("*.txt")))
    print("\nГотово. Результаты в", out_dir, f"({n_files} файлов)")
    print("  Папка в корне проекта: проект/data/phase_run (не внутри app/)")


if __name__ == "__main__":
    main()
