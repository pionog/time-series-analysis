"""Mapowanie modeli rekonstrukcji na makrokategorie RQ4 (wg gemini.txt + korekty z porównania)."""
from __future__ import annotations

import csv
import os
import re
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_PATH = os.path.join(SCRIPT_DIR, "edytowalne", "gemini.txt")
POROWNANIE_PATH = os.path.join(SCRIPT_DIR, "edytowalne", "porownanie_kategoryzacji.csv")
MANUAL_CATEGORIES_PATH = os.path.join(SCRIPT_DIR, "edytowalne", "rq4_kategorie_modele.csv")

RQ4_MACRO_CATEGORY_NAMES: dict[int, str] = {
    1: "Modele Statystyczne i Klasyczne",
    2: "Klasyczne Uczenie Maszynowe",
    3: "Sieci Rekurencyjne i ich warianty",
    4: "Transformery i Mechanizmy Uwagi",
    5: "Sieci Splotowe i Czasowe",
    6: "Grafowe Sieci Neuronowe",
    7: "Faktoryzacja Macierzowa i Dekompozycja",
    8: "Modele Generatywne i Probabilistyczne",
    9: "Architektury Hybrydowe",
    10: "Logika Rozmyta i Systemy Hybrydowe",
}

RQ4_CATEGORY_TO_QUESTION: dict[int, str] = {
    1: "RQ4-A2",
    2: "RQ4-A3",
    3: "RQ4-A4",
    4: "RQ4-A5",
    5: "RQ4-A6",
    6: "RQ4-A7",
    7: "RQ4-A8",
    8: "RQ4-A9",
    9: "RQ4-A10",
    10: "RQ4-A11",
}

# Modele wyłączone z kategorii (metryki, meta-tagi, niejednoznaczne).
EXCLUDED_MODELS: frozenset[str] = frozenset(
    {
        "IA",
        "Correlation",
        "K-step prediction",
        "Neural Networks",
        "MTSSP",
        "EFB",
        "FDW",
        "PI^m",
        "PI (pisane wielkim symbolem)",
        "PI - mi",
        "PI - M",
        "pi (pisane ma?ym symbolem)",
        "pi - M",
        "pi - mi",
        "PI^m-mi/M",
    }
)

# Korekty względem surowego gemini.txt (zgodnie z porownanie_kategoryzacji.csv).
_CATEGORY_OVERRIDES: dict[str, int] = {
    "Gaussian Distribution": 8,
    "Time Series Decomposition": 7,
    "NARX": 3,
    "SPIN": 6,
    "TRF": 8,
    "RMD-FSE": 7,
    "DLMCC": 7,
    "PSR": 7,
    "ARL Simpute": 1,
    "TimesNet": 5,
}


def _parse_gemini(path: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or " - " not in line:
                continue
            name, raw_cat = line.rsplit(" - ", 1)
            name = name.strip()
            raw_cat = raw_cat.strip().upper().rstrip("?")
            if raw_cat == "X" or not raw_cat.isdigit():
                continue
            mapping[name] = int(raw_cat)
    return mapping


def _parse_manual_categories(path: str) -> dict[str, int | None]:
    """Ręczne przypisania z edytowalne/rq4_kategorie_modele.csv (najwyższy priorytet)."""
    mapping: dict[str, int | None] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    for row in csv.DictReader(lines, delimiter=";"):
        model = (row.get("model") or "").strip()
        if not model:
            continue
        raw = (row.get("kategoria") or row.get("reczna") or "").strip().rstrip("?")
        if raw.upper() == "X":
            mapping[model] = None
        elif raw.isdigit() and 1 <= int(raw) <= 10:
            mapping[model] = int(raw)
    return mapping


def _parse_porownanie(path: str) -> dict[str, int | None]:
    """Zwraca kategorię z kolumny auto (gdy jest) lub None dla X."""
    mapping: dict[str, int | None] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            model = row.get("model", "").strip()
            if not model or model.startswith("PI ("):
                continue
            auto = row.get("auto", "").strip().rstrip("?")
            reczna = row.get("reczna", "").strip().rstrip("?")
            chosen = reczna if reczna else auto
            if chosen.upper() == "X":
                mapping[model] = None
            elif chosen.isdigit():
                mapping[model] = int(chosen)
    return mapping


def build_model_category_map() -> dict[str, int]:
    """
    Łączy źródła przypisań (od najwyższego priorytetu):
      1. edytowalne/rq4_kategorie_modele.csv — Twoja ręczna weryfikacja
      2. _CATEGORY_OVERRIDES w kodzie (korekty wbudowane)
      3. edytowalne/porownanie_kategoryzacji.csv — kolumna reczna lub auto
      4. edytowalne/gemini.txt — surowa kategoryzacja Gemini
    Modele z EXCLUDED_MODELS lub kategoria X są pomijane w RQ4-A2–A11.
    """
    result: dict[str, int] = {}
    manual = _parse_manual_categories(MANUAL_CATEGORIES_PATH)
    gemini = _parse_gemini(GEMINI_PATH)
    porownanie = _parse_porownanie(POROWNANIE_PATH)

    all_names = (
        set(manual)
        | set(gemini)
        | set(porownanie)
        | set(_CATEGORY_OVERRIDES)
        | EXCLUDED_MODELS
    )

    for name in all_names:
        if name in EXCLUDED_MODELS:
            continue
        if name in manual:
            if manual[name] is None:
                continue
            result[name] = manual[name]
            continue
        if name in _CATEGORY_OVERRIDES:
            result[name] = _CATEGORY_OVERRIDES[name]
            continue
        if name in porownanie and porownanie[name] is not None:
            result[name] = porownanie[name]
            continue
        if name in porownanie and porownanie[name] is None:
            continue
        if name in gemini:
            result[name] = gemini[name]
    return result


MODEL_TO_CATEGORY: dict[str, int] = build_model_category_map()


def counts_for_question(
    recon: dict[str, set[str]],
    question_id: str,
) -> Counter:
    counts: Counter = Counter()
    target_cats = {cat for cat, qid in RQ4_CATEGORY_TO_QUESTION.items() if qid == question_id}
    if not target_cats:
        return counts
    for model, cat in MODEL_TO_CATEGORY.items():
        if cat not in target_cats:
            continue
        article_hits = sum(1 for tags in recon.values() if model in tags)
        if article_hits:
            counts[model] = article_hits
    return counts


def macro_category_totals(recon: dict[str, set[str]]) -> Counter:
    totals: Counter = Counter()
    for model, cat in MODEL_TO_CATEGORY.items():
        article_hits = sum(1 for tags in recon.values() if model in tags)
        if article_hits:
            totals[RQ4_MACRO_CATEGORY_NAMES[cat]] += article_hits
    return totals


def unassigned_models(recon: dict[str, set[str]]) -> Counter:
    counts: Counter = Counter()
    for tags in recon.values():
        for model in tags:
            if model not in MODEL_TO_CATEGORY and model not in EXCLUDED_MODELS:
                counts[model] += 1
    return counts
