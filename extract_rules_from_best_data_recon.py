"""
Ekstrakcja reguł z pliku SLR(TAGGING-best data recon models).csv
dla RQ6-A3 i RQ6-A4 na podstawie literatury (SLR).

Wyjście (folder rules_results/):
  - evidence_long.csv                      — konfiguracja × metryka × zwycięzca
  - configurations.csv                     — jedna obserwacja na konfigurację
  - rules_rq6_a4.csv                       — reguły wg pasma missing rate
  - rules_rq6_a4_by_metric.csv             — reguły A4: missing rate + metryka
  - rules_rq6_a4_by_exact_rate.csv         — reguły A4: dokładny missing rate
  - rules_rq6_a4_by_article_vote.csv       — reguły A4: głos artykułu
  - rules_rq6_a4_by_context.csv            — reguły A4 per kolumna kontekstu (domena, order, …)
  - rules_rq6_a4_by_context_metric.csv     — reguły A4: kontekst + metryka
  - rules_rq6_a4_by_context_and_band.csv   — reguły A4: kontekst + przedział missing rate
  - rules_rq6_a3_recon_vs_task.csv         — RQ6-A3: rekonstrukcja vs zadanie
  - rules_rq6_a3_metric_consistency.csv    — RQ6-A3: spójność metryk rekonstrukcji
  - rules_rq6_a3_summary.txt               — podsumowanie tekstowe
"""

import csv
import os
from collections import Counter, defaultdict

from missing_rate_utils import (
    BAND_LABELS,
    UNKNOWN_BAND,
    classify_missing_rate,
    parse_missing_rate,
)

# --- Ustawienia ---
INPUT_DIR = "input_csv"
OUTPUT_DIR = "rules_results"
FILE_NAME = "SLR(TAGGING-best data recon models).csv"
REJECT_KEYWORD = "REJECTED"

META_COLUMNS = {
    "ID",
    "Title",
    "Tagger",
    "Dataset",
    "Dataset domain",
    "Dataset features",
    "Task Models",
    "Order",
    "Timing",
    "Missing rate",
    "Prediction Horizon",
    "Batch size",
    "Comment",
}

# Kolumny kontekstu eksperymentu (RQ6-A4) — mapowanie pole_wewnętrzne -> nagłówek CSV
CONTEXT_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("dataset", "Dataset"),
    ("dataset_domain", "Dataset domain"),
    ("dataset_features", "Dataset features"),
    ("task_model", "Task Models"),
    ("order", "Order"),
    ("timing", "Timing"),
    ("prediction_horizon", "Prediction Horizon"),
    ("batch_size", "Batch size"),
)

CONTEXT_FIELD_KEYS = [key for key, _ in CONTEXT_COLUMN_MAP] + ["missing_rate_band"]

UNKNOWN_CONTEXT_VALUES = {"", "not specified", "nieznany", "n/a", "na"}

# Metryki traktowane jako jakość rekonstrukcji / imputacji
RECONSTRUCTION_METRICS = {
    "RMSE",
    "MAPE",
    "SMAPE",
    "MAE",
    "NRMSE",
    "MSE",
    "MAD",
    "Log-likelihood",
    "Log-likelihood ratio",
    "CORR",
    "RSE",
    "RSquare",
    "MASE",
    "Dstat",
    "BIC",
    "PCC (COR)",
    "STD",
    "R2",
    "LMI",
    "TIC",
    "IA",
    "ND",
    "WMAPE",
    "MSD",
    "Maximum Absolute Error",
}

# Metryki zadania końcowego (klasyfikacja, ranking, operacyjne)
TASK_METRICS = {
    "AUC",
    "F1-Score",
    "Accuracy",
    "Precision",
    "Specifity",
    "Sensivity",
    "Recall",
    "AUPRC",
    "Model size",
    "Runtime",
    "Other",
}

MIN_SUPPORT_A4 = 2  # minimalna liczba obserwacji w paśmie, by wygenerować regułę A4
MIN_SUPPORT_A3 = 1  # minimalna liczba metryk obu typów w konfiguracji dla A3
MAX_ALGORITHM_NAME_LEN = 80  # odfiltrowuje komentarze wpisane w komórki metryk


def is_valid_algorithm_name(name: str) -> bool:
    if not name or len(name) > MAX_ALGORITHM_NAME_LEN:
        return False
    upper = name.upper()
    invalid_fragments = (
        REJECT_KEYWORD,
        "SHOULD BE",
        "NOT INCLUDED",
        "NOT SPECIFIED",
        "NO INSTITUTIONAL",
        "TABLE WITH",
        "ARTICLE IS",
        "EXTENDED ABSTRACT",
    )
    return not any(fragment in upper for fragment in invalid_fragments)


def band_to_rule_id(band: str) -> str:
    """Np. '10-20%' -> '10-20pct', 'nieznany' -> 'nieznany'."""
    return band.replace("%", "pct")


def slugify(value: str, max_len: int = 40) -> str:
    """Bezpieczny fragment rule_id z wartości kontekstu."""
    slug = "".join(c if c.isalnum() else "-" for c in value)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:max_len] or "wartosc"


def is_known_context_value(value: str) -> bool:
    return bool(value and value.strip().lower() not in UNKNOWN_CONTEXT_VALUES)


def empty_context_state() -> dict:
    state = {key: "" for key, _ in CONTEXT_COLUMN_MAP}
    state.update(
        {
            "article_id": "",
            "title": "",
            "missing_rate_raw": "",
            "missing_rate": None,
            "missing_rate_band": UNKNOWN_BAND,
            "missing_rate_unknown_reason": "brak_wartosci",
        }
    )
    return state


def reset_context_on_article_change(current: dict) -> None:
    for key, _ in CONTEXT_COLUMN_MAP:
        current[key] = ""
    current["missing_rate_raw"] = ""
    current["missing_rate"] = None
    current["missing_rate_band"] = UNKNOWN_BAND
    current["missing_rate_unknown_reason"] = "brak_wartosci"


def apply_row_context(current: dict, row: list, col_idx: dict) -> None:
    if row[col_idx["Title"]]:
        current["title"] = row[col_idx["Title"]]
    for field_key, col_name in CONTEXT_COLUMN_MAP:
        if col_name in col_idx and row[col_idx[col_name]]:
            current[field_key] = row[col_idx[col_name]]
    if row[col_idx["Missing rate"]]:
        current["missing_rate_raw"] = row[col_idx["Missing rate"]]
        rate, band, unknown_reason = classify_missing_rate(current["missing_rate_raw"])
        current["missing_rate"] = rate
        current["missing_rate_band"] = band
        current["missing_rate_unknown_reason"] = unknown_reason


def build_config_key(current: dict) -> str:
    parts = [current["article_id"]]
    for key, _ in CONTEXT_COLUMN_MAP:
        parts.append(current[key])
    parts.append(current["missing_rate_raw"])
    return "|".join(parts)


def context_fields_from_state(current: dict) -> dict:
    fields = {key: current[key] for key, _ in CONTEXT_COLUMN_MAP}
    fields.update(
        {
            "missing_rate_raw": current["missing_rate_raw"],
            "missing_rate": current["missing_rate"]
            if current["missing_rate"] is not None
            else "",
            "missing_rate_band": current["missing_rate_band"],
            "missing_rate_unknown_reason": current["missing_rate_unknown_reason"],
        }
    )
    return fields


def find_headers(reader):
    for row in reader:
        if row and row[0].strip() == "ID":
            return [h.strip() for h in row]
    return None


def parse_rows(csv_path: str) -> list[dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Plik {csv_path} nie istnieje.")

    records = []
    rejected_ids: set[str] = set()

    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = find_headers(reader)
        if headers is None:
            raise ValueError("Nie znaleziono nagłówka zaczynającego się od 'ID'.")

        col_idx = {name: i for i, name in enumerate(headers) if name}
        for col in ("ID", "Title", "Dataset", "Task Models", "Missing rate"):
            if col not in col_idx:
                raise ValueError(f"Brak kolumny: {col}")

        metric_columns = [h for h in headers if h and h not in META_COLUMNS]
        current = empty_context_state()

        for raw_row in reader:
            if not raw_row:
                continue

            row = [cell.strip() for cell in raw_row]
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))

            if any(REJECT_KEYWORD in cell.upper() for cell in row if cell):
                aid = row[col_idx["ID"]] or current["article_id"]
                if aid:
                    rejected_ids.add(aid)
                continue

            new_id = row[col_idx["ID"]]
            if new_id:
                if new_id != current["article_id"]:
                    reset_context_on_article_change(current)
                current["article_id"] = new_id

            if current["article_id"] in rejected_ids:
                continue

            apply_row_context(current, row, col_idx)

            if not current["article_id"]:
                continue

            config_key = build_config_key(current)
            context = context_fields_from_state(current)

            for metric in metric_columns:
                idx = col_idx.get(metric)
                if idx is None:
                    continue
                winner = row[idx].strip()
                if not winner or not is_valid_algorithm_name(winner):
                    continue

                metric_kind = "inne"
                if metric in RECONSTRUCTION_METRICS:
                    metric_kind = "rekonstrukcja"
                elif metric in TASK_METRICS:
                    metric_kind = "zadanie"

                records.append(
                    {
                        "article_id": current["article_id"],
                        "title": current["title"],
                        **context,
                        "metric": metric,
                        "metric_kind": metric_kind,
                        "winner_algorithm": winner,
                        "config_key": config_key,
                    }
                )

    return records


def write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def config_winners(records: list[dict]) -> list[dict]:
    """
    Jedna obserwacja na konfigurację: zwycięzca = najczęstszy algorytm
    wśród metryk w tej samej konfiguracji (unika podwójnego liczenia RMSE+MAE).
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["config_key"]].append(rec)

    configs = []
    for config_key, rows in grouped.items():
        counter = Counter(r["winner_algorithm"] for r in rows)
        top_alg, _ = counter.most_common(1)[0]
        base = rows[0]
        entry = {
            "config_key": config_key,
            "article_id": base["article_id"],
            "title": base["title"],
            "winner_algorithm": top_alg,
            "metrics_count": len(rows),
            "metrics": ";".join(sorted({r["metric"] for r in rows})),
            "all_winners": ";".join(f"{a}({c})" for a, c in counter.most_common()),
        }
        for key in CONTEXT_FIELD_KEYS:
            entry[key] = base.get(key, "")
        entry["missing_rate_raw"] = base.get("missing_rate_raw", "")
        entry["missing_rate"] = base.get("missing_rate", "")
        entry["missing_rate_unknown_reason"] = base.get(
            "missing_rate_unknown_reason", ""
        )
        configs.append(entry)
    return configs


def _make_context_rule(
    *,
    rule_id: str,
    dimension_label: str,
    dimension_field: str,
    dimension_value: str,
    recommended_algorithm: str,
    support: int,
    total: int,
    article_ids: list[str],
    extra_condition: str = "",
    metric: str = "",
) -> dict:
    condition = f"{dimension_field} = {dimension_value}"
    if extra_condition:
        condition = f"{condition} AND {extra_condition}"
    rule_text = (
        f"JEŚLI {dimension_label} = {dimension_value}"
        + (f" i {extra_condition}" if extra_condition else "")
        + (f" i metryka = {metric}" if metric else "")
        + f" TO rozważ model rekonstrukcji: {recommended_algorithm} "
        f"(support={support}/{total})"
    )
    row = {
        "rule_id": rule_id,
        "research_question": "RQ6-A4",
        "context_dimension": dimension_label,
        "context_field": dimension_field,
        "context_value": dimension_value,
        "condition": condition,
        "recommended_algorithm": recommended_algorithm,
        "support": support,
        "total_configurations": total,
        "confidence": round(support / total, 4) if total else 0,
        "unique_articles": len(article_ids),
        "article_ids": ";".join(article_ids),
        "rule_text": rule_text,
    }
    if metric:
        row["metric"] = metric
    return row


def build_rules_a4_by_context(
    configs: list[dict],
    records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Reguły RQ6-A4 warunkowane każdą kolumną kontekstu z CSV
    (domena, dataset, order, timing, horyzont predykcji, batch size, …).
    """
    dimension_specs = [
        ("dataset", "Dataset"),
        ("dataset_domain", "Dataset domain"),
        ("dataset_features", "Dataset features"),
        ("task_model", "Task Models"),
        ("order", "Order"),
        ("timing", "Timing"),
        ("prediction_horizon", "Prediction Horizon"),
        ("batch_size", "Batch size"),
        ("missing_rate_band", "Missing rate band"),
    ]

    rules_by_context: list[dict] = []
    rules_by_context_metric: list[dict] = []
    rules_by_context_band: list[dict] = []

    for field_key, field_label in dimension_specs:
        grouped: dict[str, Counter] = defaultdict(Counter)
        articles: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))

        for cfg in configs:
            value = str(cfg.get(field_key, "")).strip()
            if not is_known_context_value(value):
                continue
            if field_key == "missing_rate_band" and value == UNKNOWN_BAND:
                continue
            grouped[value][cfg["winner_algorithm"]] += 1
            articles[value][cfg["winner_algorithm"]].add(cfg["article_id"])

        for value, counter in grouped.items():
            total = sum(counter.values())
            top_alg, top_count = counter.most_common(1)[0]
            if top_count < MIN_SUPPORT_A4:
                continue
            article_ids = sorted(
                articles[value][top_alg],
                key=lambda x: int(x) if x.isdigit() else x,
            )
            rules_by_context.append(
                _make_context_rule(
                    rule_id=f"A4-ctx-{field_key}-{slugify(value)}",
                    dimension_label=field_label,
                    dimension_field=field_key,
                    dimension_value=value,
                    recommended_algorithm=top_alg,
                    support=top_count,
                    total=total,
                    article_ids=article_ids,
                )
            )

        # kontekst + metryka (na poziomie obserwacji, nie konfiguracji)
        by_value_metric: dict[str, dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        articles_vm: dict[str, dict[str, dict[str, set]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        for rec in records:
            value = str(rec.get(field_key, "")).strip()
            if not is_known_context_value(value):
                continue
            if field_key == "missing_rate_band" and value == UNKNOWN_BAND:
                continue
            metric = rec["metric"]
            alg = rec["winner_algorithm"]
            by_value_metric[value][metric][alg] += 1
            articles_vm[value][metric][alg].add(rec["article_id"])

        for value, metrics in by_value_metric.items():
            for metric, counter in metrics.items():
                total = sum(counter.values())
                top_alg, top_count = counter.most_common(1)[0]
                if top_count < MIN_SUPPORT_A4:
                    continue
                article_ids = sorted(
                    articles_vm[value][metric][top_alg],
                    key=lambda x: int(x) if x.isdigit() else x,
                )
                rules_by_context_metric.append(
                    _make_context_rule(
                        rule_id=f"A4-ctx-{field_key}-{slugify(value)}-{metric}",
                        dimension_label=field_label,
                        dimension_field=field_key,
                        dimension_value=value,
                        recommended_algorithm=top_alg,
                        support=top_count,
                        total=total,
                        article_ids=article_ids,
                        metric=metric,
                    )
                )

        # kontekst + przedział missing rate (np. domena Traffic przy 20-30%)
        if field_key == "missing_rate_band":
            continue

        combo: dict[tuple[str, str], Counter] = defaultdict(Counter)
        combo_articles: dict[tuple[str, str], dict[str, set]] = defaultdict(
            lambda: defaultdict(set)
        )
        for cfg in configs:
            value = str(cfg.get(field_key, "")).strip()
            band = str(cfg.get("missing_rate_band", "")).strip()
            if not is_known_context_value(value):
                continue
            if not band or band == UNKNOWN_BAND:
                continue
            key = (value, band)
            combo[key][cfg["winner_algorithm"]] += 1
            combo_articles[key][cfg["winner_algorithm"]].add(cfg["article_id"])

        for (value, band), counter in combo.items():
            total = sum(counter.values())
            top_alg, top_count = counter.most_common(1)[0]
            if top_count < MIN_SUPPORT_A4:
                continue
            article_ids = sorted(
                combo_articles[(value, band)][top_alg],
                key=lambda x: int(x) if x.isdigit() else x,
            )
            band_id = band_to_rule_id(band)
            rules_by_context_band.append(
                _make_context_rule(
                    rule_id=f"A4-ctx-{field_key}-{slugify(value)}-band-{band_id}",
                    dimension_label=field_label,
                    dimension_field=field_key,
                    dimension_value=value,
                    recommended_algorithm=top_alg,
                    support=top_count,
                    total=total,
                    article_ids=article_ids,
                    extra_condition=f"missing_rate_band = {band}",
                )
            )

    rules_by_context.sort(key=lambda r: (-r["support"], r["rule_id"]))
    rules_by_context_metric.sort(key=lambda r: (-r["support"], r["rule_id"]))
    rules_by_context_band.sort(key=lambda r: (-r["support"], r["rule_id"]))
    return rules_by_context, rules_by_context_metric, rules_by_context_band


def build_rules_a4(records: list[dict], configs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """RQ6-A4: przy danym paśmie missing rate wybierz najczęstszego zwycięzcę."""
    overall = defaultdict(Counter)
    by_metric = defaultdict(lambda: defaultdict(Counter))
    by_exact_rate = defaultdict(Counter)
    articles_overall = defaultdict(lambda: defaultdict(set))
    articles_by_metric = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    articles_by_rate = defaultdict(lambda: defaultdict(set))

    for cfg in configs:
        band = cfg["missing_rate_band"]
        alg = cfg["winner_algorithm"]
        aid = cfg["article_id"]
        rate_key = cfg["missing_rate_raw"] or UNKNOWN_BAND

        overall[band][alg] += 1
        articles_overall[band][alg].add(aid)
        by_exact_rate[rate_key][alg] += 1
        articles_by_rate[rate_key][alg].add(aid)

    for rec in records:
        band = rec["missing_rate_band"]
        alg = rec["winner_algorithm"]
        metric = rec["metric"]
        aid = rec["article_id"]

        by_metric[band][metric][alg] += 1
        articles_by_metric[band][metric][alg].add(aid)

    rules = []
    rules_by_metric = []

    for band in BAND_LABELS:
        if band not in overall:
            continue
        total = sum(overall[band].values())
        if total == 0:
            continue
        top_alg, top_count = overall[band].most_common(1)[0]
        if top_count < MIN_SUPPORT_A4:
            continue

        band_id = band_to_rule_id(band)
        article_ids = sorted(articles_overall[band][top_alg], key=lambda x: int(x) if x.isdigit() else x)
        rules.append(
            {
                "rule_id": f"A4-{band_id}",
                "research_question": "RQ6-A4",
                "condition": f"missing_rate_band = {band}",
                "recommended_algorithm": top_alg,
                "support": top_count,
                "total_configurations_in_band": total,
                "confidence": round(top_count / total, 4),
                "unique_articles": len(article_ids),
                "article_ids": ";".join(article_ids),
                "rule_text": (
                    f"JEŚLI missing_rate ∈ [{band}] "
                    f"TO rozważ model rekonstrukcji: {top_alg} "
                    f"(support={top_count}/{total}, confidence={top_count / total:.2%})"
                ),
            }
        )

        for metric, counter in sorted(by_metric[band].items()):
            m_total = sum(counter.values())
            m_top, m_count = counter.most_common(1)[0]
            if m_count < MIN_SUPPORT_A4:
                continue
            m_articles = sorted(
                articles_by_metric[band][metric][m_top],
                key=lambda x: int(x) if x.isdigit() else x,
            )
            rules_by_metric.append(
                {
                    "rule_id": f"A4-{band_id}-{metric}",
                    "research_question": "RQ6-A4",
                    "condition": f"missing_rate_band = {band} AND metric = {metric}",
                    "metric": metric,
                    "recommended_algorithm": m_top,
                    "support": m_count,
                    "total_observations": m_total,
                    "confidence": round(m_count / m_total, 4),
                    "unique_articles": len(m_articles),
                    "article_ids": ";".join(m_articles),
                    "rule_text": (
                        f"JEŚLI missing_rate ∈ [{band}] i metryka = {metric} "
                        f"TO rozważ: {m_top} (support={m_count}/{m_total})"
                    ),
                }
            )

    rules_by_rate = []
    for rate_key, counter in sorted(
        by_exact_rate.items(),
        key=lambda x: (parse_missing_rate(x[0]) is None, parse_missing_rate(x[0]) or 0),
    ):
        total = sum(counter.values())
        if total < MIN_SUPPORT_A4:
            continue
        top_alg, top_count = counter.most_common(1)[0]
        article_ids = sorted(
            articles_by_rate[rate_key][top_alg],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        rules_by_rate.append(
            {
                "rule_id": f"A4-rate-{rate_key}",
                "research_question": "RQ6-A4",
                "condition": f"missing_rate = {rate_key}%",
                "recommended_algorithm": top_alg,
                "support": top_count,
                "total_configurations": total,
                "confidence": round(top_count / total, 4),
                "unique_articles": len(article_ids),
                "article_ids": ";".join(article_ids),
                "rule_text": (
                    f"JEŚLI missing_rate = {rate_key}% "
                    f"TO rozważ: {top_alg} (support={top_count}/{total})"
                ),
            }
        )

    # Głos artykułu: jeden artykuł = jeden głos (zwycięzca większościowy w paśmie)
    article_vote = defaultdict(Counter)
    article_ids_by_band_alg = defaultdict(lambda: defaultdict(set))
    by_band_article: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for cfg in configs:
        band = cfg["missing_rate_band"]
        by_band_article[band][cfg["article_id"]][cfg["winner_algorithm"]] += 1

    for band, articles in by_band_article.items():
        for aid, counter in articles.items():
            top_alg, _ = counter.most_common(1)[0]
            article_vote[band][top_alg] += 1
            article_ids_by_band_alg[band][top_alg].add(aid)

    rules_by_article = []
    for band in BAND_LABELS:
        if band not in article_vote:
            continue
        total = sum(article_vote[band].values())
        if total == 0:
            continue
        top_alg, top_count = article_vote[band].most_common(1)[0]
        if top_count < 1:
            continue
        band_id = band_to_rule_id(band)
        article_ids = sorted(
            article_ids_by_band_alg[band][top_alg],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        rules_by_article.append(
            {
                "rule_id": f"A4-articles-{band_id}",
                "research_question": "RQ6-A4",
                "condition": f"missing_rate_band = {band}",
                "recommended_algorithm": top_alg,
                "article_votes": top_count,
                "total_articles_in_band": total,
                "confidence": round(top_count / total, 4),
                "article_ids": ";".join(article_ids),
                "rule_text": (
                    f"JEŚLI missing_rate ∈ [{band}] "
                    f"TO rozważ: {top_alg} "
                    f"(głosy artykułów={top_count}/{total})"
                ),
            }
        )

    rules.sort(key=lambda r: (-r["support"], r["rule_id"]))
    rules_by_metric.sort(key=lambda r: (-r["support"], r["rule_id"]))
    rules_by_rate.sort(key=lambda r: (-r["support"], r["rule_id"]))
    rules_by_article.sort(key=lambda r: (-r["article_votes"], r["rule_id"]))
    return rules, rules_by_metric, rules_by_rate, rules_by_article


def build_rules_a3(records: list[dict], configs: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """
    RQ6-A3:
    (1) recon vs task — gdy w konfiguracji są oba typy metryk;
    (2) spójność wielu metryk rekonstrukcji w tej samej konfiguracji.
    """
    by_config = defaultdict(lambda: {"recon": Counter(), "task": Counter(), "meta": {}})

    for rec in records:
        key = rec["config_key"]
        kind = rec["metric_kind"]
        if kind == "rekonstrukcja":
            by_config[key]["recon"][rec["winner_algorithm"]] += 1
        elif kind == "zadanie":
            by_config[key]["task"][rec["winner_algorithm"]] += 1
        if not by_config[key]["meta"]:
            by_config[key]["meta"] = {
                "article_id": rec["article_id"],
                "title": rec["title"],
                "dataset": rec["dataset"],
                "task_model": rec["task_model"],
                "missing_rate_raw": rec["missing_rate_raw"],
                "missing_rate_band": rec["missing_rate_band"],
                "missing_rate_unknown_reason": rec.get(
                    "missing_rate_unknown_reason", ""
                ),
            }

    rules_transfer = []
    agree_count = 0
    partial_count = 0
    disagree_count = 0
    insufficient_count = 0

    for config_key, data in sorted(by_config.items(), key=lambda x: x[0]):
        recon = data["recon"]
        task = data["task"]
        meta = data["meta"]

        if not recon or not task:
            insufficient_count += 1
            continue

        top_recon, _ = recon.most_common(1)[0]
        top_task, _ = task.most_common(1)[0]
        recon_set = set(recon.keys())
        task_set = set(task.keys())
        overlap = recon_set & task_set

        if top_recon == top_task:
            agreement = "pełna_zgodność_top1"
            agree_count += 1
        elif overlap:
            agreement = "częściowa_zgodność"
            partial_count += 1
        else:
            agreement = "brak_zgodności"
            disagree_count += 1

        rules_transfer.append(
            {
                "rule_id": f"A3-transfer-{meta['article_id']}",
                "analysis_type": "recon_vs_task",
                "research_question": "RQ6-A3",
                "config_key": config_key,
                "article_id": meta["article_id"],
                "title": meta["title"],
                "dataset": meta["dataset"],
                "task_model": meta["task_model"],
                "missing_rate": meta["missing_rate_raw"],
                "missing_rate_band": meta["missing_rate_band"],
                "missing_rate_unknown_reason": meta.get(
                    "missing_rate_unknown_reason", ""
                ),
                "top_reconstruction_algorithm": top_recon,
                "top_task_algorithm": top_task,
                "reconstruction_winners": ";".join(f"{a}({c})" for a, c in recon.most_common()),
                "task_winners": ";".join(f"{a}({c})" for a, c in task.most_common()),
                "agreement": agreement,
                "overlap_algorithms": ";".join(sorted(overlap)) if overlap else "",
                "note": "Bezpośrednie porównanie rekonstrukcja vs zadanie końcowe",
            }
        )

    rules_metric_consistency = []
    consistent = 0
    inconsistent = 0
    single_metric = 0

    grouped_metrics: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped_metrics[rec["config_key"]].append(rec)

    for config_key, rows in sorted(grouped_metrics.items()):
        if len(rows) < 2:
            single_metric += 1
            continue
        counter = Counter(r["winner_algorithm"] for r in rows)
        unique_algs = len(counter)
        base = rows[0]
        if unique_algs == 1:
            consistent += 1
            agreement = "wszystkie_metryki_zgodne"
        else:
            inconsistent += 1
            agreement = "metryki_wskazuja_rozne_modele"

        rules_metric_consistency.append(
            {
                "rule_id": f"A3-metrics-{base['article_id']}-{config_key[:30]}",
                "analysis_type": "multi_metric_consistency",
                "research_question": "RQ6-A3",
                "config_key": config_key,
                "article_id": base["article_id"],
                "title": base["title"],
                "dataset": base["dataset"],
                "task_model": base["task_model"],
                "missing_rate": base["missing_rate_raw"],
                "missing_rate_band": base["missing_rate_band"],
                "missing_rate_unknown_reason": base.get(
                    "missing_rate_unknown_reason", ""
                ),
                "metrics": ";".join(sorted({r["metric"] for r in rows})),
                "winners": ";".join(f"{a}({c})" for a, c in counter.most_common()),
                "agreement": agreement,
                "note": (
                    "Ten sam algorytm wygrywa na wszystkich metrykach rekonstrukcji "
                    "w tej konfiguracji (proxy spójności jakości)"
                ),
            }
        )

    total_comparable = agree_count + partial_count + disagree_count
    multi = consistent + inconsistent
    summary_lines = [
        "Podsumowanie RQ6-A3",
        "",
        "Przedziały missing rate: co 10% (0-10%, 10-20%, ..., 90-100%).",
        "Nieznany procent = brak_wartosci | nienumeryczna_wartosc | ujemna_wartosc | poza_zakresem.",
        "",
        "A) Rekonstrukcja vs zadanie końcowe (bezpośrednio):",
        f"   - konfiguracje z oboma typami metryk: {total_comparable}",
        f"   - pełna zgodność top-1: {agree_count}",
        f"   - częściowa zgodność: {partial_count}",
        f"   - brak zgodności: {disagree_count}",
        f"   - tylko jeden typ metryki w konfiguracji: {insufficient_count}",
        "   UWAGA: W tym pliku SLR prawie wyłącznie metryki rekonstrukcji;",
        "   pełna odpowiedź na RQ6-A3 wymaga też danych o metrykach zadania końcowego.",
        "",
        "B) Spójność wielu metryk rekonstrukcji (proxy w tym pliku):",
        f"   - konfiguracje z >=2 metrykami: {multi}",
        f"   - ten sam zwycięzca na wszystkich metrykach: {consistent}",
        f"   - różni zwycięzcy: {inconsistent}",
        f"   - tylko 1 metryka: {single_metric}",
    ]
    if total_comparable:
        summary_lines.append(
            f"   - odsetek zgodności recon/task top-1: {agree_count / total_comparable:.1%}"
        )
    if multi:
        summary_lines.append(
            f"   - odsetek spójności multi-metric: {consistent / multi:.1%}"
        )

    return rules_transfer, rules_metric_consistency, summary_lines


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_DIR, FILE_NAME)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)

    print(f"Wczytywanie: {input_path}")
    records = parse_rows(input_path)
    print(f"  -> {len(records)} obserwacji (wierszy dowodowych)")

    evidence_fields = [
        "article_id",
        "title",
        "dataset",
        "dataset_domain",
        "dataset_features",
        "task_model",
        "order",
        "timing",
        "prediction_horizon",
        "batch_size",
        "missing_rate_raw",
        "missing_rate",
        "missing_rate_band",
        "missing_rate_unknown_reason",
        "metric",
        "metric_kind",
        "winner_algorithm",
        "config_key",
    ]
    evidence_path = os.path.join(output_dir, "evidence_long.csv")
    write_csv(evidence_path, evidence_fields, records)
    print(f"Zapisano: {evidence_path}")

    configs = config_winners(records)
    config_fields = [
        "config_key",
        "article_id",
        "title",
        "dataset",
        "dataset_domain",
        "dataset_features",
        "task_model",
        "order",
        "timing",
        "prediction_horizon",
        "batch_size",
        "missing_rate_raw",
        "missing_rate",
        "missing_rate_band",
        "missing_rate_unknown_reason",
        "winner_algorithm",
        "metrics_count",
        "metrics",
        "all_winners",
    ]
    config_path = os.path.join(output_dir, "configurations.csv")
    write_csv(config_path, config_fields, configs)
    print(f"Zapisano: {config_path} ({len(configs)} konfiguracji)")

    rules_a4, rules_a4_metric, rules_a4_rate, rules_a4_article = build_rules_a4(
        records, configs
    )
    a4_path = os.path.join(output_dir, "rules_rq6_a4.csv")
    write_csv(
        a4_path,
        [
            "rule_id",
            "research_question",
            "condition",
            "recommended_algorithm",
            "support",
            "total_configurations_in_band",
            "confidence",
            "unique_articles",
            "article_ids",
            "rule_text",
        ],
        rules_a4,
    )
    print(f"Zapisano: {a4_path} ({len(rules_a4)} reguł)")

    a4m_path = os.path.join(output_dir, "rules_rq6_a4_by_metric.csv")
    write_csv(
        a4m_path,
        [
            "rule_id",
            "research_question",
            "condition",
            "metric",
            "recommended_algorithm",
            "support",
            "total_observations",
            "confidence",
            "unique_articles",
            "article_ids",
            "rule_text",
        ],
        rules_a4_metric,
    )
    print(f"Zapisano: {a4m_path} ({len(rules_a4_metric)} reguł)")

    a4r_path = os.path.join(output_dir, "rules_rq6_a4_by_exact_rate.csv")
    write_csv(
        a4r_path,
        [
            "rule_id",
            "research_question",
            "condition",
            "recommended_algorithm",
            "support",
            "total_configurations",
            "confidence",
            "unique_articles",
            "article_ids",
            "rule_text",
        ],
        rules_a4_rate,
    )
    print(f"Zapisano: {a4r_path} ({len(rules_a4_rate)} reguł)")

    a4a_path = os.path.join(output_dir, "rules_rq6_a4_by_article_vote.csv")
    write_csv(
        a4a_path,
        [
            "rule_id",
            "research_question",
            "condition",
            "recommended_algorithm",
            "article_votes",
            "total_articles_in_band",
            "confidence",
            "article_ids",
            "rule_text",
        ],
        rules_a4_article,
    )
    print(f"Zapisano: {a4a_path} ({len(rules_a4_article)} reguł)")

    rules_ctx, rules_ctx_metric, rules_ctx_band = build_rules_a4_by_context(
        configs, records
    )
    context_rule_fields = [
        "rule_id",
        "research_question",
        "context_dimension",
        "context_field",
        "context_value",
        "condition",
        "recommended_algorithm",
        "support",
        "total_configurations",
        "confidence",
        "unique_articles",
        "article_ids",
        "rule_text",
    ]
    a4ctx_path = os.path.join(output_dir, "rules_rq6_a4_by_context.csv")
    write_csv(a4ctx_path, context_rule_fields, rules_ctx)
    print(f"Zapisano: {a4ctx_path} ({len(rules_ctx)} reguł)")

    a4ctxm_path = os.path.join(output_dir, "rules_rq6_a4_by_context_metric.csv")
    write_csv(
        a4ctxm_path,
        context_rule_fields + ["metric"],
        rules_ctx_metric,
    )
    print(f"Zapisano: {a4ctxm_path} ({len(rules_ctx_metric)} reguł)")

    a4ctxb_path = os.path.join(output_dir, "rules_rq6_a4_by_context_and_band.csv")
    write_csv(a4ctxb_path, context_rule_fields, rules_ctx_band)
    print(f"Zapisano: {a4ctxb_path} ({len(rules_ctx_band)} reguł)")

    rules_a3_transfer, rules_a3_metrics, summary_lines = build_rules_a3(records, configs)

    a3_transfer_path = os.path.join(output_dir, "rules_rq6_a3_recon_vs_task.csv")
    write_csv(
        a3_transfer_path,
        [
            "rule_id",
            "analysis_type",
            "research_question",
            "config_key",
            "article_id",
            "title",
            "dataset",
            "task_model",
            "missing_rate",
            "missing_rate_band",
            "missing_rate_unknown_reason",
            "top_reconstruction_algorithm",
            "top_task_algorithm",
            "reconstruction_winners",
            "task_winners",
            "agreement",
            "overlap_algorithms",
            "note",
        ],
        rules_a3_transfer,
    )
    print(f"Zapisano: {a3_transfer_path} ({len(rules_a3_transfer)} wierszy)")

    a3_path = os.path.join(output_dir, "rules_rq6_a3_metric_consistency.csv")
    write_csv(
        a3_path,
        [
            "rule_id",
            "analysis_type",
            "research_question",
            "config_key",
            "article_id",
            "title",
            "dataset",
            "task_model",
            "missing_rate",
            "missing_rate_band",
            "missing_rate_unknown_reason",
            "metrics",
            "winners",
            "agreement",
            "note",
        ],
        rules_a3_metrics,
    )
    print(f"Zapisano: {a3_path} ({len(rules_a3_metrics)} wierszy)")

    summary_path = os.path.join(output_dir, "rules_rq6_a3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Zapisano: {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
