"""Eksport list modeli RQ4 do ręcznej weryfikacji i korekty przypisań."""
from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import date

from build_research_html import parse_questions_csv
from domain_groups import DOMAIN_QUESTION_IDS, recon_models_for_group, resolve_domain_groups
from research_answers import INPUT_DIR, load_feature_csv, parse_tagging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "edytowalne")
TXT_PATH = os.path.join(OUTPUT_DIR, "rq4_modele_do_edycji.txt")
CSV_PATH = os.path.join(OUTPUT_DIR, "rq4_modele_do_edycji.csv")

# Odzwierciedla obecne przypisania w research_answers.py (mogą wymagać korekty).
RQ4_MODEL_BUCKETS: dict[str, tuple[str, ...]] = {
    "RQ4-A2": ("ARIMA", "SARIMA", "Holt-Winters", "ETS", "Kalman filter"),
    "RQ4-A3": (
        "KNN Impute",
        "Mean Imputation",
        "Zero Imputation",
        "MICE",
        "Multiple Imputation",
        "LOCF",
        "Backward Fill",
        "Forward Fill",
        "DropNaN",
        "IMPWE",
    ),
    "RQ4-A4": (
        "LSTM",
        "Bi-LSTM",
        "GRU-D",
        "GRUI",
        "M-RNN",
        "Bi-FCNN",
        "Uni-FCNN",
        "LSTNet",
        "M-LSTNet",
        "BRITS",
        "SAITS",
    ),
    "RQ4-A5": ("GRIN", "GATGPT", "MTGNN", "GRUI"),
    "RQ4-A6": ("GATGPT", "GPT4TS", "Standard attention mechanism", "TimesNet"),
    "RQ4-A7": ("GAN", "US-GAN", "GMM", "MF", "MICE", "CSDI", "Gaussian Distribution"),
}

RQ4_A1_MACRO_CATEGORIES = (
    "Modele Statystyczne i Klasyczne",
    "Klasyczne Uczenie Maszynowe",
    "Sieci Rekurencyjne i ich warianty",
    "Transformery i Mechanizmy Uwagi",
    "Sieci Splotowe i Czasowe",
    "Grafowe Sieci Neuronowe",
    "Faktoryzacja Macierzowa i Dekompozycja",
    "Modele Generatywne i Probabilistyczne",
    "Architektury Hybrydowe",
    "Logika Rozmyta i Systemy Hybrydowe",
)


def _question_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for block in parse_questions_csv():
        if block["rq_id"] != "RQ4":
            continue
        for aq in block["analytical"]:
            if aq.get("id"):
                texts[aq["id"]] = aq["text"]
    return texts


def _counts_from_tagging(recon: dict[str, set[str]], models: tuple[str, ...]) -> list[tuple[str, int]]:
    counter = Counter(
        {m: sum(1 for tags in recon.values() if m in tags) for m in models}
    )
    return sorted(
        ((name, count) for name, count in counter.items() if count),
        key=lambda x: (-x[1], x[0]),
    )


def _all_recon_models() -> list[tuple[str, int]]:
    rows = load_feature_csv("SLR(TAGGING-models data reconstruct)_feature_results.csv")
    items = [(row["Cecha"], int(row["Liczba"])) for row in rows]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def _bench_models() -> list[tuple[str, int]]:
    rows = load_feature_csv("SLR(TAGGING-bench model data recon)_feature_results.csv")
    items = [(row["Cecha"], int(row["Liczba"])) for row in rows]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def _assigned_models() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for qid, models in RQ4_MODEL_BUCKETS.items():
        for model in models:
            if model not in mapping:
                mapping[model] = qid
            else:
                mapping[model] = f"{mapping[model]},{qid}"
    return mapping


def _counts_rq4_a6(recon: dict[str, set[str]], bench: dict[str, set[str]]) -> list[tuple[str, int]]:
    models = RQ4_MODEL_BUCKETS["RQ4-A6"]
    counter = Counter({m: sum(1 for tags in recon.values() if m in tags) for m in models})
    counter["Standard attention mechanism"] = counter.get("Standard attention mechanism", 0) + sum(
        1 for tags in bench.values() if "Standard attention mechanism" in tags
    )
    return sorted(
        ((name, count) for name, count in counter.items() if count),
        key=lambda x: (-x[1], x[0]),
    )


def _write_txt(
    *,
    question_texts: dict[str, str],
    recon: dict[str, set[str]],
    bench: dict[str, set[str]],
    apps: dict[str, set[str]],
    assigned: dict[str, str],
    all_models: list[tuple[str, int]],
    bench_models: list[tuple[str, int]],
    csv_rows: list[dict[str, str]],
) -> None:
    lines: list[str] = [
        "=" * 80,
        "RQ4 — MODELE REKONSTRUKCJI — PLIK DO RĘCZNEJ WERYFIKACJI",
        "=" * 80,
        f"Wygenerowano: {date.today().isoformat()}",
        "Źródło danych: input_csv/SLR(TAGGING-models data reconstruct).csv",
        f"Liczba artykułów w SLR: {csv_rows[0]['liczba_artykulow'] if csv_rows else '—'}",
        "",
        "UWAGA: Sekcje RQ4-A2–A7 odzwierciedlają OBECNE przypisania ze skryptu research_answers.py.",
        "       Mogą być błędne (np. RQ4-A3 pyta o (A)NN, a lista zawiera metody imputacji).",
        "",
        "INSTRUKCJA EDYCJI:",
        "  1. Przenieś modele między sekcjami według właściwej kategorii.",
        "  2. Dopisz brakujące modele z sekcji „Nieprzypisane”.",
        "  3. Uzupełnij makrokategorie w RQ4-A1.",
        "  4. Po korekcie można zaktualizować research_answers.py lub zaimportować CSV.",
        "",
    ]

    lines.extend(["-" * 80, "RQ4-A1: " + question_texts.get("RQ4-A1", ""), "-" * 80, ""])
    for idx, category in enumerate(RQ4_A1_MACRO_CATEGORIES, start=1):
        lines.append(f"  [{idx}] {category}")
        lines.append("      (wpisz modele, po jednym w linii, np. „ARIMA — 2”)")
        lines.append("")
    lines.append("")

    for qid in (
        "RQ4-A2",
        "RQ4-A3",
        "RQ4-A4",
        "RQ4-A5",
        "RQ4-A6",
        "RQ4-A7",
    ):
        lines.extend(["-" * 80, f"{qid}: {question_texts.get(qid, '')}", "-" * 80, ""])
        items = (
            _counts_rq4_a6(recon, bench)
            if qid == "RQ4-A6"
            else _counts_from_tagging(recon, RQ4_MODEL_BUCKETS[qid])
        )
        if not items:
            lines.append("  (brak modeli w obecnym przypisaniu)")
        else:
            for idx, (name, count) in enumerate(items, start=1):
                lines.append(f"  {idx:>2}. {name} ({count})")
        lines.append("")

    lines.extend(["-" * 80, f"RQ4-A8: {question_texts.get('RQ4-A8', '')}", "-" * 80, ""])
    if bench_models:
        for idx, (name, count) in enumerate(bench_models, start=1):
            lines.append(f"  {idx:>2}. {name} ({count})")
    else:
        lines.append("  (brak danych)")
    lines.append("")

    for qid, label in DOMAIN_QUESTION_IDS.items():
        if not qid.startswith("RQ4-"):
            continue
        lines.extend(["-" * 80, f"{qid}: {question_texts.get(qid, '')}", "-" * 80, ""])
        groups = resolve_domain_groups(label)
        if not groups:
            lines.append("  (brak mapowania domeny)")
            lines.append("")
            continue
        for group in groups:
            counts = recon_models_for_group(apps, recon, group)
            lines.append(f"  Domena: {group}")
            if not counts:
                lines.append("    (brak modeli)")
            else:
                for idx, (name, count) in enumerate(counts.most_common(), start=1):
                    lines.append(f"    {idx:>2}. {name} ({count})")
            lines.append("")

    unassigned = [
        (name, count)
        for name, count in all_models
        if name not in assigned
    ]
    lines.extend(
        [
            "-" * 80,
            "MODELE NIEPRZYPISANE DO RQ4-A2–A7 (pełna lista ze skoroszytu)",
            "-" * 80,
            "",
        ]
    )
    if not unassigned:
        lines.append("  (wszystkie modele są w bucketach A2–A7)")
    else:
        for idx, (name, count) in enumerate(unassigned, start=1):
            lines.append(f"  {idx:>3}. {name} ({count})")
    lines.append("")

    lines.extend(
        [
            "-" * 80,
            "SPIS WSZYSTKICH MODELI — OBECNE PRZYPISANIE (do szybkiej korekty)",
            "Kolumny: model | liczba | obecne_pytanie | docelowe_pytanie | uwagi",
            "-" * 80,
            "",
        ]
    )
    for name, count in all_models:
        current = assigned.get(name, "")
        lines.append(
            f"  {name} | {count} | {current or '—'} | |"
        )
    lines.append("")

    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_csv(
    *,
    question_texts: dict[str, str],
    recon: dict[str, set[str]],
    bench: dict[str, set[str]],
    apps: dict[str, set[str]],
    assigned: dict[str, str],
    all_models: list[tuple[str, int]],
    bench_models: list[tuple[str, int]],
) -> None:
    rows: list[dict[str, str]] = []

    for idx, category in enumerate(RQ4_A1_MACRO_CATEGORIES, start=1):
        rows.append(
            {
                "pytanie_id": "RQ4-A1",
                "pytanie": question_texts.get("RQ4-A1", ""),
                "podsekcja": category,
                "kolejnosc": str(idx),
                "model": "",
                "liczba_artykulow": "",
                "zrodlo": "makrokategoria",
                "obecne_pytanie": "",
                "docelowe_pytanie": "",
                "uwagi": "do uzupełnienia",
            }
        )

    for qid in (
        "RQ4-A2",
        "RQ4-A3",
        "RQ4-A4",
        "RQ4-A5",
        "RQ4-A6",
        "RQ4-A7",
    ):
        items = (
            _counts_rq4_a6(recon, bench)
            if qid == "RQ4-A6"
            else _counts_from_tagging(recon, RQ4_MODEL_BUCKETS[qid])
        )
        for idx, (name, count) in enumerate(items, start=1):
            rows.append(
                {
                    "pytanie_id": qid,
                    "pytanie": question_texts.get(qid, ""),
                    "podsekcja": "",
                    "kolejnosc": str(idx),
                    "model": name,
                    "liczba_artykulow": str(count),
                    "zrodlo": "automatyczne",
                    "obecne_pytanie": qid,
                    "docelowe_pytanie": "",
                    "uwagi": "",
                }
            )

    for idx, (name, count) in enumerate(bench_models, start=1):
        rows.append(
            {
                "pytanie_id": "RQ4-A8",
                "pytanie": question_texts.get("RQ4-A8", ""),
                "podsekcja": "",
                "kolejnosc": str(idx),
                "model": name,
                "liczba_artykulow": str(count),
                "zrodlo": "bench",
                "obecne_pytanie": "RQ4-A8",
                "docelowe_pytanie": "",
                "uwagi": "",
            }
        )

    for qid, label in DOMAIN_QUESTION_IDS.items():
        if not qid.startswith("RQ4-"):
            continue
        groups = resolve_domain_groups(label)
        order = 0
        for group in groups:
            counts = recon_models_for_group(apps, recon, group)
            for name, count in counts.most_common():
                order += 1
                rows.append(
                    {
                        "pytanie_id": qid,
                        "pytanie": question_texts.get(qid, ""),
                        "podsekcja": group,
                        "kolejnosc": str(order),
                        "model": name,
                        "liczba_artykulow": str(count),
                        "zrodlo": "domena",
                        "obecne_pytanie": qid,
                        "docelowe_pytanie": "",
                        "uwagi": "",
                    }
                )

    for idx, (name, count) in enumerate(
        [(n, c) for n, c in all_models if n not in assigned],
        start=1,
    ):
        rows.append(
            {
                "pytanie_id": "",
                "pytanie": "Nieprzypisane (A2–A7)",
                "podsekcja": "",
                "kolejnosc": str(idx),
                "model": name,
                "liczba_artykulow": str(count),
                "zrodlo": "nieprzypisane",
                "obecne_pytanie": "",
                "docelowe_pytanie": "",
                "uwagi": "",
            }
        )

    fieldnames = [
        "pytanie_id",
        "pytanie",
        "podsekcja",
        "kolejnosc",
        "model",
        "liczba_artykulow",
        "zrodlo",
        "obecne_pytanie",
        "docelowe_pytanie",
        "uwagi",
    ]
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    question_texts = _question_texts()
    recon = parse_tagging(
        os.path.join(INPUT_DIR, "SLR(TAGGING-models data reconstruct).csv")
    )
    bench = parse_tagging(
        os.path.join(INPUT_DIR, "SLR(TAGGING-bench model data recon).csv")
    )
    apps = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-applications).csv"))
    assigned = _assigned_models()
    all_models = _all_recon_models()
    bench_models = _bench_models()

    csv_rows_preview: list[dict[str, str]] = []
    if all_models:
        rows = load_feature_csv("SLR(TAGGING-models data reconstruct)_feature_results.csv")
        total_articles = rows[0].get("Lacznie_artykulow", "") if rows else ""
        csv_rows_preview = [{"liczba_artykulow": total_articles}]

    _write_txt(
        question_texts=question_texts,
        recon=recon,
        bench=bench,
        apps=apps,
        assigned=assigned,
        all_models=all_models,
        bench_models=bench_models,
        csv_rows=csv_rows_preview,
    )
    _write_csv(
        question_texts=question_texts,
        recon=recon,
        bench=bench,
        apps=apps,
        assigned=assigned,
        all_models=all_models,
        bench_models=bench_models,
    )
    print(f"Zapisano: {TXT_PATH}")
    print(f"Zapisano: {CSV_PATH}")


if __name__ == "__main__":
    main()
