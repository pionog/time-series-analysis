"""Budowanie odpowiedzi na pytania badawcze — logika analityczna."""
import csv
import html
import os
import re
from collections import Counter, defaultdict

from degradation_categories import (
    GENERAL_DEGRADATION_CATEGORIES,
    TAG_TO_GENERAL,
    count_general_categories,
    heatmap_general_columns,
)
from transformation_categories import TRANSFORMATION_CATEGORY_ORDER, categorize_transformation
from     domain_groups import (
    APPLICATION_DOMAIN_GROUPS,
    DOMAIN_QUESTION_IDS,
    article_in_group,
    count_articles_per_group,
    degradation_for_group,
    domain_general_degradation_matrix,
    general_degradation_for_group,
    general_degradation_for_groups,
    recon_models_for_group,
    resolve_domain_groups,
    structural_degradation_by_group,
)
from rq4_model_mapping import RQ4_CATEGORY_TO_QUESTION, macro_category_totals, counts_for_question
from html_tables import (
    render_bullet_list,
    render_count_lines,
    render_matrix_table,
    sample_note,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_csv")
CSV_RESULTS = os.path.join(SCRIPT_DIR, "csv_results")
RULES_DIR = os.path.join(SCRIPT_DIR, "rules_results")

STRUCTURAL_DEGRADATION = {
    "Interrupted time series",
    "Non-stationarity",
    "Changepoint",
    "Non-linear",
    "Outliers",
}
DRIFT_DEGRADATION = {"Drift", "Changepoint"}
STRUCTURAL_EXCLUDED_FROM_GENERAL = {"Drift", "Outliers"}
PREDICTION_TYPE_COLS = ("Interval", "Single-value", "Unknown", "Other")
MISSINGNESS_TYPES = {"MNAR", "MAR", "MCAR", "Unknown missingness"}
NOISE_TYPES = {"Additive distortions", "White Noise", "Gaussian Noise", "Unknown noise"}
DEGRADATION_WAY = {"Natural", "Synthetic", "Unknown Degradation"}

DIMENSION_COLS = ("Univariate", "Multivariate")
FREQUENCY_ORDER = (
    "Yearly", "Half a year", "Quaterly", "Monthly", "Weekly", "Daily", "Hourly",
    "Half an hour", "15 minutes", "10 minutes", "5 minutes", "Minute",
    "5 seconds", "Seconds", "100 milliseconds", "Other",
)
FREQUENCY_COLS = FREQUENCY_ORDER

RECON_METRIC_COLS = {
    "RMSE", "MAPE", "SMAPE", "MAE", "NRMSE", "MSE", "MAD", "Log-likelihood",
    "Log-likelihood ratio", "CORR", "RSE", "MASE", "Dstat", "BIC", "PCC (COR)",
    "STD", "R2", "LMI", "TIC", "IA", "ND", "WMAPE", "MSD", "Maximum Absolute Error",
}
TASK_METRIC_COLS = {
    "AUC", "F1-Score", "Accuracy", "Precision", "Specifity", "Sensivity",
    "Recall", "AUPRC",
}


def parse_tagging(path: str) -> dict[str, set[str]]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx: dict[str, int] = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers:
            return {}

        feature_cols = [
            h for h in headers
            if h and h not in {"ID", "Title", "Tagger", "Comment", "Comments"}
        ]
        result: dict[str, set[str]] = {}
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            tags = {
                fc for fc in feature_cols
                if row[col_idx[fc]].strip().lower() == "x"
            }
            result[aid] = tags
        return result


def _read_data_description_header(path: str) -> tuple[list[str], dict[str, int]] | tuple[None, None]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                return headers, {h: i for i, h in enumerate(headers) if h}
    return None, None


def parse_prediction_type_counts(path: str) -> Counter:
    headers, col_idx = _read_data_description_header(path)
    if not headers:
        return Counter()
    other_indices = [i for i, h in enumerate(headers) if h == "Other"]
    type_columns: list[tuple[str, int]] = []
    for col in ("Interval", "Single-value", "Unknown"):
        if col in col_idx:
            type_columns.append((col, col_idx[col]))
    if other_indices:
        type_columns.append(("Other", other_indices[0]))
    counts: Counter = Counter()
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            for col, idx in type_columns:
                if row[idx].strip().lower() == "x":
                    counts[col] += 1
    return counts


def frequency_other_details(path: str) -> list[tuple[str, str, str]]:
    headers, col_idx = _read_data_description_header(path)
    if not headers:
        return []
    other_indices = [i for i, h in enumerate(headers) if h == "Other"]
    if not other_indices:
        return []
    freq_other_idx = other_indices[-1]
    desc_idx = col_idx.get("Transformation description")
    title_idx = col_idx.get("Title")
    id_idx = col_idx["ID"]
    rows: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[id_idx].strip()
            if not aid or not aid.isdigit():
                continue
            if row[freq_other_idx].strip().lower() != "x":
                continue
            title = row[title_idx].strip() if title_idx is not None else ""
            note = ""
            if desc_idx is not None:
                note = row[desc_idx].strip()
            if not note:
                note = (
                    "Autorzy nie podali standardowej jednostki czasu w skoroszycie "
                    "(np. nieregularna próbkowanie, Hz, niestandardowy interwał)."
                )
            rows.append((aid, title, note))
    return rows


def count_rq3_a2_categories(degrad: dict[str, set[str]]) -> list[tuple[str, int]]:
    structural_tags = GENERAL_DEGRADATION_CATEGORIES["Zniekształcenia strukturalne"]
    counts: dict[str, int] = {c: 0 for c in GENERAL_DEGRADATION_CATEGORIES}
    drift_n = 0
    outliers_n = 0
    for tags in degrad.values():
        for cat, cat_tags in GENERAL_DEGRADATION_CATEGORIES.items():
            if cat == "Zniekształcenia strukturalne":
                continue
            if tags & cat_tags:
                counts[cat] += 1
        if tags & structural_tags:
            if "Drift" in tags:
                drift_n += 1
            if "Outliers" in tags:
                outliers_n += 1
            if tags & (structural_tags - STRUCTURAL_EXCLUDED_FROM_GENERAL):
                counts["Zniekształcenia strukturalne"] += 1
    items = [(k, v) for k, v in counts.items() if v]
    if drift_n:
        items.append(("Drift (wyodrębnione ze strukturalnych)", drift_n))
    if outliers_n:
        items.append(("Outliers (wyodrębnione ze strukturalnych)", outliers_n))
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def domain_quality_heatmap(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    groups: list[str],
    total: int,
) -> str:
    matrix: Counter = Counter()
    tag_totals: Counter = Counter()
    for aid in set(apps) & set(degrad):
        if not any(article_in_group(apps[aid], g) for g in groups):
            continue
        for tag in degrad[aid]:
            general = TAG_TO_GENERAL.get(tag)
            if not general:
                continue
            matrix[(tag, general)] += 1
            tag_totals[tag] += 1
    if not matrix:
        return ""
    row_labels = [tag for tag, _ in tag_totals.most_common()]
    col_labels = heatmap_general_columns()
    parts = [
        "<p><strong>Mapa ciepła: szczegółowe problemy × ogólna kategoria</strong></p>",
        render_matrix_table(
            row_labels,
            col_labels,
            matrix,
            total_articles=total,
            caption="Oś Y: szczegółowy problem · Oś X: ogólna kategoria jakości danych",
        ),
    ]
    return "\n".join(parts)


def parse_transformation_descriptions(path: str) -> Counter:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx: dict[str, int] = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers or "Transformation description" not in col_idx:
            return Counter()

        idx = col_idx["Transformation description"]
        counts: Counter = Counter()
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            text = row[idx].strip()
            if text:
                counts[text] += 1
        return counts


def parse_degrad_method_with_comments(path: str) -> tuple[Counter, list[str]]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx: dict[str, int] = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers:
            return Counter(), []

        method_cols = [
            h for h in headers
            if h and h not in {"ID", "Title", "Tagger", "Comments", "Comment"}
        ]
        counts: Counter = Counter()
        comments: list[str] = []
        comment_idx = col_idx.get("Comments") or col_idx.get("Comment")

        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            for mc in method_cols:
                if row[col_idx[mc]].strip().lower() == "x":
                    counts[mc] += 1
            if comment_idx is not None:
                c = row[comment_idx].strip()
                if c:
                    comments.append(f"[{aid}] {c}")
        return counts, comments


def _frequency_column_indices(headers: list[str]) -> dict[str, int]:
    other_indices = [i for i, h in enumerate(headers) if h == "Other"]
    freq_other_idx = other_indices[-1] if other_indices else None
    indices: dict[str, int] = {}
    for f in FREQUENCY_ORDER:
        if f == "Other":
            if freq_other_idx is not None:
                indices[f] = freq_other_idx
        else:
            try:
                indices[f] = headers.index(f)
            except ValueError:
                continue
    return indices


def dimension_frequency_matrix(path: str) -> tuple[dict[tuple[str, str], int], int, int]:
    headers, col_idx = _read_data_description_header(path)
    if not headers:
        return {}, 0, 0
    freq_idx = _frequency_column_indices(headers)
    dim_idx = {d: col_idx[d] for d in DIMENSION_COLS if d in col_idx}

    matrix: dict[tuple[str, str], int] = defaultdict(int)
    with_both = 0
    articles = 0

    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            articles += 1
            dims = [d for d, idx in dim_idx.items() if row[idx].strip().lower() == "x"]
            freqs = [f for f, idx in freq_idx.items() if row[idx].strip().lower() == "x"]
            if dims and freqs:
                with_both += 1
                for d in dims:
                    for f in freqs:
                        matrix[(d, f)] += 1
    return dict(matrix), with_both, articles


def build_rq2_a3_html(path: str) -> str:
    matrix, with_both, articles = dimension_frequency_matrix(path)
    if not matrix:
        return ""
    active_freqs = [f for f in FREQUENCY_ORDER if any(matrix.get((d, f), 0) for d in DIMENSION_COLS)]
    transposed = {(f, d): matrix.get((d, f), 0) for d in DIMENSION_COLS for f in active_freqs}
    parts = [
        f"<p>Artykuły z obiema informacjami: <strong>{with_both}</strong> / {articles}"
        f"{sample_note(with_both, articles)}</p>",
        render_matrix_table(
            active_freqs,
            list(DIMENSION_COLS),
            transposed,
            total_articles=articles,
            caption="Oś X: wymiarowość · Oś Y: częstotliwość (od najniższej do najwyższej)",
        ),
        (
            "<p><strong>Other</strong> (częstotliwość): jednostka czasu niepasująca do standardowej "
            "siatki w skoroszycie — np. nieregularne próbkowanie, Hz, niestandardowy interwał "
            "lub brak jednoznacznej informacji u autorów.</p>"
        ),
    ]
    other_rows = frequency_other_details(path)
    if other_rows:
        lis = []
        for aid, title, note in other_rows:
            short = title[:80] + ("…" if len(title) > 80 else "")
            lis.append(
                f"<li><strong>ID {html.escape(aid)}</strong>"
                f"{': ' + html.escape(short) if short else ''}"
                f" — {html.escape(note[:200])}"
                f"{'…' if len(note) > 200 else ''}</li>"
            )
        parts.append(
            "<p>Artykuły z częstotliwością <em>Other</em>:</p>"
            f'<ul class="bullet-list">{"".join(lis)}</ul>'
        )
    return "\n".join(parts)


def build_rq2_a5_html(desc_counts: Counter, total: int) -> str:
    if not desc_counts:
        return ""
    grouped: dict[str, list[tuple[str, int]]] = {c: [] for c in TRANSFORMATION_CATEGORY_ORDER}
    for text, count in desc_counts.items():
        grouped[categorize_transformation(text)].append((text, count))
    category_counts = [
        (cat, sum(c for _, c in items))
        for cat, items in grouped.items()
        if items
    ]
    category_counts.sort(key=lambda x: (-x[1], x[0]))
    parts = [
        "<p>Opisy z kolumny <em>Transformation description</em> zgrupowane według typu rozwiązania:</p>",
        render_count_lines(category_counts, total),
    ]
    for cat, items in grouped.items():
        if not items:
            continue
        items.sort(key=lambda x: (-x[1], x[0]))
        parts.append(
            f'<details class="sub-answer"><summary>{html.escape(cat)} ({sum(c for _, c in items)})</summary>'
            f"{render_bullet_list(items, total)}</details>"
        )
    return "\n".join(parts)


def domain_quality_answer(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    groups: list[str],
    total: int,
) -> str:
    general = general_degradation_for_groups(apps, degrad, groups)
    if not general:
        return ""
    parts = [
        "<p><strong>Ogólne kategorie problemów z jakością:</strong></p>",
        render_count_lines(general.most_common(), total),
        domain_quality_heatmap(apps, degrad, groups, total),
    ]
    for group in groups:
        specific = degradation_for_group(apps, degrad, group)
        if not specific:
            continue
        parts.append(
            f'<details class="sub-answer"><summary>Szczegóły — {html.escape(group)}</summary>'
            f"{render_count_lines(specific.most_common(), total)}</details>"
        )
    return "\n".join(parts)


def parse_summary_winners(path: str) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx: dict[str, int] = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers:
            return {}

        metric_cols = [h for h in headers if h and h not in {"ID", "Title", "Tagger"}]
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            winners = {
                h: row[col_idx[h]].strip()
                for h in metric_cols
                if row[col_idx[h]].strip()
            }
            if winners:
                result[aid] = winners
        return result


def build_rq6_a2_html(apps: dict[str, set[str]], total: int) -> str:
    summary_path = os.path.join(INPUT_DIR, "SLR(TAGGING-best data recon SUMMARY).csv")
    algo_path = os.path.join(
        CSV_RESULTS, "SLR(TAGGING-best data recon SUMMARY)_algorithm_summary.csv"
    )
    winners_by_article = parse_summary_winners(summary_path)

    parts: list[str] = []

    # --- ogólny ranking ---
    rows = []
    with open(algo_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append((row["Algorithm"], int(row["Total"]), int(row["Articles"])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: (-x[1], -x[2], x[0]))
    if rows:
        parts.append("<p><strong>Ogólny ranking algorytmów (SUMMARY):</strong></p>")
        parts.append(
            render_bullet_list([(f"{a} — łącznie {t}, artykuły {art}", t) for a, t, art in rows[:15]], total)
        )

    # --- zwycięzcy per metryka rekonstrukcji ---
    metric_winners: Counter = Counter()
    with open(algo_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        metric_headers = [h for h in reader.fieldnames or [] if h not in {"Algorithm", "Total", "Articles"}]
        for row in reader:
            alg = row.get("Algorithm", "")
            for m in metric_headers:
                try:
                    if int(row.get(m, 0) or 0) > 0:
                        metric_winners[(m, alg)] += int(row[m])
                except ValueError:
                    continue

    by_metric: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (metric, alg), cnt in metric_winners.items():
        by_metric[metric].append((alg, cnt))
    highlight_metrics = top_used_metrics(5)
    metrics_to_show = [m for m in highlight_metrics if m in by_metric]
    if by_metric and metrics_to_show:
        parts.append(
            "<p><strong>Najczęściej wygrywające algorytmy wg metryki rekonstrukcji</strong> "
            f"(top {len(metrics_to_show)} metryk wg użycia w literaturze: "
            f"{', '.join(html.escape(m) for m in metrics_to_show)}):</p>"
        )
        for metric in metrics_to_show:
            top = sorted(by_metric[metric], key=lambda x: (-x[1], x[0]))[:3]
            line = ", ".join(f"{a} ({c})" for a, c in top)
            note = sample_note(sum(c for _, c in top), total)
            parts.append(f"<p>{html.escape(metric)}: {html.escape(line)}{note}</p>")

    # --- wg typu zadania (applications) ---
    task_map = {
        "Prediction": "predykcja",
        "Classification": "klasyfikacja",
        "Anomaly detection": "wykrywanie anomalii",
    }
    for task_tag, label in task_map.items():
        aids = {aid for aid, tags in apps.items() if task_tag in tags}
        algo_counts: Counter = Counter()
        for aid in aids:
            if aid not in winners_by_article:
                continue
            for metric, alg in winners_by_article[aid].items():
                if metric in RECON_METRIC_COLS or metric in TASK_METRIC_COLS:
                    algo_counts[alg] += 1
        if algo_counts:
            parts.append(f"<p><strong>Zwycięzcy w artykułach z zadaniem: {label}</strong> "
                         f"({len(aids)} artykułów){sample_note(len(aids), total)}:</p>")
            parts.append(render_bullet_list(algo_counts.most_common(10), total))

    # --- kombinacje dataset + task model + artykuł ---
    bdr_path = os.path.join(INPUT_DIR, "SLR(TAGGING-best data recon models).csv")
    combos: set[tuple[str, str, str]] = set()
    with open(bdr_path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx: dict[str, int] = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if headers:
            for row in reader:
                if not row:
                    continue
                row = row + [""] * (len(headers) - len(row))
                aid = row[col_idx["ID"]].strip()
                if not aid or not aid.isdigit():
                    continue
                dataset = row[col_idx.get("Dataset", -1)].strip() if "Dataset" in col_idx else ""
                task = row[col_idx.get("Task Models", -1)].strip() if "Task Models" in col_idx else ""
                if dataset and task:
                    combos.add((aid, dataset, task))

    if combos:
        by_article = len({c[0] for c in combos})
        parts.append(
            f"<p><strong>Kombinacje dataset + task model + artykuł</strong> "
            f"(best data recon models): <strong>{len(combos)}</strong> wierszy konfiguracyjnych, "
            f"<strong>{by_article}</strong> artykułów{sample_note(len(combos), total)}.</p>"
        )
        combo_counts: Counter = Counter()
        for aid, ds, tm in combos:
            combo_counts[(ds, tm)] += 1
        top_combos = [(f"{ds} + {tm}", c) for (ds, tm), c in combo_counts.most_common(10)]
        parts.append("<p>Najczęstsze pary dataset–model zadaniowy:</p>")
        parts.append(render_bullet_list(top_combos, total))

    return "\n".join(parts)


def load_feature_csv(filename: str) -> list[dict]:
    path = os.path.join(CSV_RESULTS, filename)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt_list(items: list[tuple[str, int]], total: int | None = None, min_count: int = 1) -> str:
    filtered = [(n, c) for n, c in items if c >= min_count]
    return render_count_lines(filtered, total)


def fmt_counter(counter: Counter, total: int | None = None, min_count: int = 1) -> str:
    return fmt_list(counter.most_common(), total, min_count)


def _read_rule_texts_from_csv(
    filename: str,
    *,
    context_field: str | None = None,
    limit: int | None = None,
) -> list[str]:
    path = os.path.join(RULES_DIR, filename)
    if not os.path.exists(path):
        return []
    lines: list[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if context_field is not None and row.get("context_field") != context_field:
                continue
            text = row.get("rule_text", "").strip()
            if text:
                lines.append(text)
    if limit is not None and len(lines) > limit:
        extra = len(lines) - limit
        lines = lines[:limit] + [f"... oraz {extra} kolejnych reguł."]
    return lines


def _format_rule_sections(sections: list[tuple[str, list[str]]]) -> str:
    parts: list[str] = []
    for title, lines in sections:
        if not lines:
            continue
        parts.append(f"<strong>{html.escape(title)}</strong>")
        parts.append("<br>\n".join(html.escape(line) for line in lines))
    return "<br><br>\n".join(parts)


def read_rq6_a3_summary() -> str:
    path = os.path.join(RULES_DIR, "rules_rq6_a3_summary.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    if "\nReguły" in text:
        text = text.split("\nReguły", 1)[0].strip()
    return html.escape(text).replace("\n", "<br>\n")


def build_rq6_a4_answer() -> str:
    return _format_rule_sections(
        [
            (
                "Reguły per wartość order:",
                _read_rule_texts_from_csv("rules_rq6_a4_by_context.csv", context_field="order"),
            ),
            (
                "Reguły order × przedział missing rate:",
                _read_rule_texts_from_csv(
                    "rules_rq6_a4_by_context_and_band.csv", context_field="order"
                ),
            ),
        ]
    )


def build_rq6_a5_answer() -> str:
    return _format_rule_sections(
        [
            (
                "Reguły per domena zbioru danych:",
                _read_rule_texts_from_csv(
                    "rules_rq6_a4_by_context.csv", context_field="dataset_domain"
                ),
            ),
            (
                "Reguły domena × przedział missing rate:",
                _read_rule_texts_from_csv(
                    "rules_rq6_a4_by_context_and_band.csv", context_field="dataset_domain"
                ),
            ),
        ]
    )


def build_rq6_a6_answer() -> str:
    ctx_band_all = _read_rule_texts_from_csv("rules_rq6_a4_by_context_and_band.csv")
    main = _format_rule_sections(
        [
            (
                "Głos konfiguracji (missing rate):",
                _read_rule_texts_from_csv("rules_rq6_a4.csv"),
            ),
            (
                "Głos artykułu (mniej stronnicze):",
                _read_rule_texts_from_csv("rules_rq6_a4_by_article_vote.csv"),
            ),
            (
                "Per przedział missing rate:",
                _read_rule_texts_from_csv(
                    "rules_rq6_a4_by_context.csv", context_field="missing_rate_band"
                ),
            ),
            (
                "Kontekst × przedział missing rate (top 10):",
                ctx_band_all[:10],
            ),
        ]
    )
    rest = ctx_band_all[10:]
    if not rest:
        return main
    rest_block = (
        '<details class="sub-answer"><summary>'
        f"Pozostałe reguły kontekst × przedział missing rate ({len(rest)})</summary>"
        + "<br>\n".join(html.escape(rule) for rule in rest)
        + "</details>"
    )
    if main:
        return main + "<br><br>\n" + rest_block
    return rest_block


def models_for_task_category(
    apps: dict[str, set[str]],
    task_models: dict[str, set[str]],
    category: str,
) -> Counter:
    counts: Counter = Counter()
    for aid in set(apps) & set(task_models):
        if category not in apps[aid]:
            continue
        for model in task_models[aid]:
            counts[model] += 1
    return counts


def items_from_feature_rows(rows: list[dict]) -> list[tuple[str, int]]:
    items = [(row["Cecha"], int(row["Liczba"])) for row in rows]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


def build_count_answer_with_details(
    items: list[tuple[str, int]],
    total: int,
    min_count: int = 2,
    details_summary: str = "Pełna lista (również pojedyncze wystąpienia)",
) -> str:
    if not items:
        return ""
    main_items = [(n, c) for n, c in items if c >= min_count]
    parts = [render_count_lines(main_items, total)]
    if len(items) > len(main_items):
        parts.append(
            f'<details class="sub-answer"><summary>{html.escape(details_summary)}</summary>'
            f"{render_count_lines(items, total)}</details>"
        )
    return "\n".join(parts)


def top_used_metrics(limit: int = 5) -> list[str]:
    """Najczęściej tagowane metryki (kolejność z csv_results)."""
    rows = load_feature_csv("SLR(TAGGING-metrics used)_feature_results.csv")
    return [row["Cecha"] for row in rows[:limit]]


def merge_dataset_feature_counts(
    datasets_rows: list[dict],
    benchmark_rows: list[dict],
) -> Counter:
    """RQ1-A3: suma liczników z RQ1-A1 (datasets) i RQ1-A2 (benchmark datasets) per nazwa."""
    merged: Counter = Counter()
    for row in datasets_rows + benchmark_rows:
        merged[row["Cecha"]] += int(row["Liczba"])
    return merged


def build_rq1_a3_html(
    datasets_rows: list[dict],
    benchmark_rows: list[dict],
    total: int,
    min_count: int = 2,
) -> str:
    ds_map = {r["Cecha"]: int(r["Liczba"]) for r in datasets_rows}
    bench_map = {r["Cecha"]: int(r["Liczba"]) for r in benchmark_rows}
    merged = merge_dataset_feature_counts(datasets_rows, benchmark_rows)

    both = sorted(set(ds_map) & set(bench_map))
    intro = (
        "<p>Dla każdej nazwy zbioru: <strong>liczba artykułów = suma</strong> z kolumny "
        "<em>Dataset</em> (RQ1-A1) i <em>Benchmark datasets</em> (RQ1-A2). "
        f"Unikalnych nazw po połączeniu: <strong>{len(merged)}</strong>.</p>"
    )
    if both:
        examples = []
        for name in both[:3]:
            examples.append(
                f"{name}: {ds_map[name]}+{bench_map[name]}={merged[name]}"
            )
        intro += (
            "<p>Nazwy występujące w obu kolumnach (przykłady): "
            + "; ".join(html.escape(e) for e in examples)
            + ".</p>"
        )
    intro += (
        "<p>Poniżej zbiory z ≥2 wystąpieniami; pełna lista w sekcji rozwijanej.</p>"
    )
    items = [(n, c) for n, c in merged.most_common() if c >= min_count]
    parts = [intro, render_count_lines(items, total)]
    all_items = merged.most_common()
    if len(all_items) > len(items):
        parts.append(
            '<details class="sub-answer"><summary>Pełna lista połączona (również pojedyncze wystąpienia)</summary>'
            f"{render_count_lines(all_items, total)}</details>"
        )
    return "\n".join(parts)


def set_from_csv(
    answers: dict,
    key: str,
    filename: str,
    total: int,
    min_count: int = 1,
    features: set[str] | None = None,
):
    rows = load_feature_csv(filename)
    items = []
    for row in rows:
        name = row["Cecha"]
        count = int(row["Liczba"])
        if features and name not in features:
            continue
        items.append((name, count))
    items.sort(key=lambda x: (-x[1], x[0]))
    text = fmt_list(items, total, min_count)
    if text:
        answers[key] = text


def build_answers() -> dict[str, str]:
    answers: dict[str, str] = {}
    total = 148

    all_datasets = load_feature_csv("SLR(TAGGING-datasets)_feature_results.csv")
    bench_datasets = load_feature_csv("SLR(TAGGING-benchmark datasets)_feature_results.csv")
    if all_datasets:
        total = int(all_datasets[0]["Lacznie_artykulow"])

    answers["RQ1-A1"] = build_count_answer_with_details(
        items_from_feature_rows(all_datasets), total
    )
    answers["RQ1-A2"] = build_count_answer_with_details(
        items_from_feature_rows(bench_datasets), total
    )
    merged = merge_dataset_feature_counts(all_datasets, bench_datasets)
    answers["RQ1-A3"] = build_rq1_a3_html(all_datasets, bench_datasets, total, min_count=2)

    apps = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-applications).csv"))
    group_counts = count_articles_per_group(apps)
    answers["RQ1-A4"] = fmt_list(group_counts.most_common(), total)

    desc = load_feature_csv("SLR(TAGGING - Data description)_feature_results.csv")
    desc_map = {r["Cecha"]: int(r["Liczba"]) for r in desc}
    desc_path = os.path.join(INPUT_DIR, "SLR(TAGGING - Data description).csv")

    dim_parts = [(n, desc_map[n]) for n in ("Multivariate", "Univariate") if n in desc_map]
    if dim_parts:
        answers["RQ2-A1"] = render_count_lines(dim_parts, total)

    freq_names = [f for f in FREQUENCY_ORDER if f in desc_map and desc_map[f] > 0]
    answers["RQ2-A2"] = fmt_list([(n, desc_map[n]) for n in freq_names], total)

    rq2_a3 = build_rq2_a3_html(desc_path)
    if rq2_a3:
        answers["RQ2-A3"] = rq2_a3

    trans_parts = [(n, desc_map[n]) for n in ("Transformed", "Raw") if n in desc_map]
    if trans_parts:
        answers["RQ2-A4"] = render_count_lines(trans_parts, total)

    trans_desc = parse_transformation_descriptions(desc_path)
    rq2_a5 = build_rq2_a5_html(trans_desc, total)
    if rq2_a5:
        answers["RQ2-A5"] = rq2_a5

    degrad = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING - Data degrad type).csv"))
    degrad_rows = load_feature_csv("SLR(TAGGING - Data degrad type)_feature_results.csv")

    set_from_csv(answers, "RQ3-A1", "SLR(TAGGING - Data degrad type)_feature_results.csv", total, features=DEGRADATION_WAY)

    gen_counts = count_general_categories(degrad)
    a2_items = count_rq3_a2_categories(degrad)
    if a2_items:
        answers["RQ3-A2"] = render_count_lines(a2_items, total)
    elif gen_counts:
        answers["RQ3-A2"] = render_count_lines(
            sorted(gen_counts.items(), key=lambda x: (-x[1], x[0])), total
        )

    set_from_csv(answers, "RQ3-A3", "SLR(TAGGING - Data degrad type)_feature_results.csv", total, features=NOISE_TYPES)
    set_from_csv(answers, "RQ3-A4", "SLR(TAGGING - Data degrad type)_feature_results.csv", total, features=MISSINGNESS_TYPES)
    set_from_csv(answers, "RQ3-A5", "SLR(TAGGING - Data degrad type)_feature_results.csv", total, features=STRUCTURAL_DEGRADATION)

    drift_counts = Counter({f: sum(1 for t in degrad.values() if f in t) for f in DRIFT_DEGRADATION})
    drift_counts = Counter({k: v for k, v in drift_counts.items() if v})
    if drift_counts:
        answers["RQ3-A6"] = fmt_counter(drift_counts, total)

    method_counts, method_comments = parse_degrad_method_with_comments(
        os.path.join(INPUT_DIR, "SLR(TAGGING- data degrad method).csv")
    )
    if method_counts:
        parts = [
            "<p><strong>Metody jawnie oznaczone w skoroszycie:</strong></p>",
            render_count_lines(method_counts.most_common(), total),
        ]
        if method_comments:
            parts.append("<p><strong>Uzupełnienia z kolumny Comments:</strong></p>")
            parts.append("<ul class=\"bullet-list\">" + "".join(
                f"<li>{html.escape(c)}</li>" for c in method_comments
            ) + "</ul>")
        answers["RQ3-A7"] = "\n".join(parts)

    dm = domain_general_degradation_matrix(apps, degrad)
    if dm:
        categories = heatmap_general_columns()
        domains = sorted(
            {g for g, _ in dm},
            key=lambda g: -sum(dm[(g, c)] for c in categories if (g, c) in dm),
        )
        answers["RQ3-A8"] = render_matrix_table(
            domains, categories, dm, total_articles=total,
            caption="Domena zastosowania × ogólna kategoria problemu z jakością",
        )
        struct_specific = structural_degradation_by_group(apps, degrad, STRUCTURAL_DEGRADATION)
        if struct_specific:
            lines = [f"{grp} + {deg}: {cnt}" for (grp, deg), cnt in struct_specific.most_common(15)]
            answers["RQ3-A8-detail"] = (
                "<p><em>Szczegółowe zniekształcenia strukturalne (podpytanie):</em></p>"
                + "<br>\n".join(html.escape(line) for line in lines)
            )

    for qid, label in DOMAIN_QUESTION_IDS.items():
        if not qid.startswith("RQ3-"):
            continue
        groups = resolve_domain_groups(label)
        if not groups:
            continue
        ans = domain_quality_answer(apps, degrad, groups, total)
        if ans:
            answers[qid] = ans

    recon = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-models data reconstruct).csv"))

    macro = macro_category_totals(recon)
    if macro:
        answers["RQ4-A1"] = render_count_lines(
            sorted(macro.items(), key=lambda x: (-x[1], x[0])), total
        )

    for qid in RQ4_CATEGORY_TO_QUESTION.values():
        cat_counts = counts_for_question(recon, qid)
        if cat_counts:
            answers[qid] = fmt_counter(cat_counts, total)

    top_recon = load_feature_csv("SLR(TAGGING-models data reconstruct)_feature_results.csv")[:5]
    if top_recon:
        names = ", ".join(f"{r['Cecha']} ({r['Liczba']})" for r in top_recon)
        answers["RQ4"] = (
            f"W {total} artykułach zidentyfikowano wiele metod rekonstrukcji; "
            f"najczęściej tagowane: {names}."
        )

    set_from_csv(answers, "RQ4-A12", "SLR(TAGGING-bench model data recon)_feature_results.csv", total)

    for qid, label in DOMAIN_QUESTION_IDS.items():
        if not qid.startswith("RQ4-"):
            continue
        groups = resolve_domain_groups(label)
        parts = []
        for group in groups:
            c = recon_models_for_group(apps, recon, group)
            if c:
                parts.append(f"<p><strong>{html.escape(group)}</strong></p>")
                parts.append(fmt_counter(c, total))
        if parts:
            answers[qid] = "\n".join(parts)

    task_cats = ["Prediction", "Classification", "Anomaly detection", "other"]
    cat_lines = [(cat, sum(1 for tags in apps.values() if cat in tags)) for cat in task_cats]
    cat_lines = [(c, n) for c, n in cat_lines if n]
    if cat_lines:
        answers["RQ5-A1"] = render_count_lines(cat_lines, total)

    task_models = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-task models).csv"))
    pred_type_counts = parse_prediction_type_counts(desc_path)
    if pred_type_counts:
        answers["RQ5-A3"] = render_count_lines(pred_type_counts.most_common(), total)

    for qid, cat in (
        ("RQ5-A2", "Prediction"),
        ("RQ5-A4", "Classification"),
        ("RQ5-A5", "Anomaly detection"),
    ):
        c = models_for_task_category(apps, task_models, cat)
        if not c:
            continue
        if qid == "RQ5-A2":
            answers[qid] = build_count_answer_with_details(c.most_common(), total)
        else:
            answers[qid] = fmt_counter(c, total)

    other_models = models_for_task_category(apps, task_models, "other")
    if other_models:
        answers["RQ5-A6"] = fmt_counter(other_models, total)

    metrics_rows = load_feature_csv("SLR(TAGGING-metrics used)_feature_results.csv")
    answers["RQ6-A1"] = build_count_answer_with_details(
        items_from_feature_rows(metrics_rows), total
    )
    answers["RQ6-A2"] = build_rq6_a2_html(apps, total)
    answers["RQ6-A3"] = read_rq6_a3_summary()
    answers["RQ6-A4"] = build_rq6_a4_answer()
    answers["RQ6-A5"] = build_rq6_a5_answer()
    answers["RQ6-A6"] = build_rq6_a6_answer()

    if all_datasets:
        answers["RQ1"] = (
            f"Przeanalizowano {total} artykułów. "
            f"Zidentyfikowano {len(all_datasets)} nazw zbiorów danych "
            f"i {len(bench_datasets)} nazw benchmarkowych "
            f"({len(merged)} unikalnych po połączeniu)."
        )
    if dim_parts:
        answers["RQ2"] = (
            f"Szeregi czasowe są przede wszystkim wielowymiarowe "
            f"({desc_map.get('Multivariate', 0)} artykułów) i często transformowane "
            f"({desc_map.get('Transformed', 0)} artykułów)."
        )
    synthetic_n = next((int(r["Liczba"]) for r in degrad_rows if r["Cecha"] == "Synthetic"), 0)
    answers["RQ3"] = (
        f"Dominuje degradacja syntetyczna ({synthetic_n} artykułów). "
        "Najczęstsze ogólne kategorie: missingness, sposób powstawania degradacji."
    )
    if cat_lines:
        answers["RQ5"] = f"Najczęstsze zadanie końcowe: {cat_lines[0][0]} ({cat_lines[0][1]} artykułów)."
    answers["RQ6"] = (
        "Najczęściej raportowane metryki rekonstrukcji: RMSE, MSE, MAE. "
        "Ranking zwycięzców i kombinacje konfiguracyjne — patrz RQ6-A2."
    )

    return answers
