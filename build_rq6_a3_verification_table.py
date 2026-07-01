"""
Łączy pliki SLR z input_csv w tabele do ręcznej weryfikacji RQ6-A3.

Wyjście (folder rq6_a3_verification/):
  - verification_all_articles.csv      — wszystkie artykuły (bez REJECTED)
  - verification_candidates.csv        — priorytetowi kandydaci do weryfikacji
  - verification_configurations.csv    — wiersze z best data recon models (szczegóły)
  - README.txt                         — opis kolumn i kolejności pracy
"""

import csv
import os
from collections import Counter, defaultdict

from missing_rate_utils import classify_missing_rate

INPUT_DIR = "input_csv"
OUTPUT_DIR = "rq6_a3_verification"
REJECT_KEYWORD = "REJECTED"

RECON_METRICS = {
    "RMSE", "MAPE", "SMAPE", "MAE", "NRMSE", "MSE", "MAD",
    "Log-likelihood", "Log-likelihood ratio", "CORR", "RSE", "MASE",
    "Dstat", "BIC", "PCC (COR)", "STD", "R2", "LMI", "TIC", "IA", "ND",
    "WMAPE", "MSD", "Maximum Absolute Error", "RSquare",
}
TASK_METRICS = {
    "AUC", "F1-Score", "Accuracy", "Precision", "Specifity",
    "Sensivity", "Recall", "AUPRC", "Model size", "Runtime", "Other",
}

TAGGING_FILES = {
    "task_models": "SLR(TAGGING-task models).csv",
    "recon_models": "SLR(TAGGING-models data reconstruct).csv",
    "metrics_used": "SLR(TAGGING-metrics used).csv",
    "applications": "SLR(TAGGING-applications).csv",
    "degrad_type": "SLR(TAGGING - Data degrad type).csv",
    "degrad_method": "SLR(TAGGING- data degrad method).csv",
    "datasets": "SLR(TAGGING-datasets).csv",
    "benchmark_datasets": "SLR(TAGGING-benchmark datasets).csv",
}

BEST_RECON_MODELS = "SLR(TAGGING-best data recon models).csv"
BEST_RECON_SUMMARY = "SLR(TAGGING-best data recon SUMMARY).csv"

MANUAL_COLUMNS = [
    "manual_verified",
    "manual_paired_recon_task",
    "manual_has_numeric_table",
    "manual_pipeline_documented",
    "manual_notes",
]


def find_header(reader):
    for row in reader:
        if row and row[0].strip() == "ID":
            return [h.strip() for h in row]
    return None


def is_rejected(row) -> bool:
    return any(REJECT_KEYWORD in (cell or "").upper() for cell in row)


def parse_tagging_file(path: str) -> dict[str, dict]:
    """Zwraca {article_id: {title, tags: set[str]}}."""
    if not os.path.exists(path):
        return {}

    result: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = find_header(reader)
        if not headers:
            return {}

        col_idx = {h: i for i, h in enumerate(headers) if h}
        feature_cols = [
            h for h in headers
            if h and h not in {"ID", "Title", "Tagger", "Comment", "Comments"}
        ]

        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if is_rejected(row):
                continue

            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue

            title = row[col_idx["Title"]].strip() if "Title" in col_idx else ""
            tags = {
                fc for fc in feature_cols
                if row[col_idx[fc]].strip().lower() == "x"
            }
            result[aid] = {"title": title, "tags": tags}

    return result


def parse_summary_winners(path: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = find_header(reader)
        if not headers:
            return result

        col_idx = {h: i for i, h in enumerate(headers) if h}
        metric_cols = [h for h in headers if h and h not in {"ID", "Title", "Tagger"}]

        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if is_rejected(row):
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


def split_metrics(tags: set[str]) -> tuple[list[str], list[str], list[str]]:
    recon, task, other = [], [], []
    for m in sorted(tags):
        if m in RECON_METRICS:
            recon.append(m)
        elif m in TASK_METRICS:
            task.append(m)
        else:
            other.append(m)
    return recon, task, other


def format_winners(winners: dict[str, str], metric_set: set[str]) -> str:
    parts = [
        f"{k}={v}"
        for k, v in sorted(winners.items())
        if k in metric_set
    ]
    return "; ".join(parts)


def parse_best_recon_configurations(path: str) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """
    Zwraca:
      - per article: lista konfiguracji z best data recon models
      - per article: tytuł
    """
    if not os.path.exists(path):
        return {}, {}

    configs_by_article: dict[str, list[dict]] = defaultdict(list)
    titles: dict[str, str] = {}

    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = find_header(reader)
        if not headers:
            return {}, {}

        col_idx = {h: i for i, h in enumerate(headers) if h}
        metric_columns = [
            h for h in headers
            if h
            and h
            not in {
                "ID", "Title", "Tagger", "Dataset", "Task Models",
                "Order", "Timing", "Missing rate", "Batch size", "Comment",
            }
        ]

        current = {
            "article_id": "",
            "title": "",
            "dataset": "",
            "task_model": "",
            "missing_rate": "",
        }

        for raw_row in reader:
            if not raw_row:
                continue
            row = [c.strip() for c in raw_row]
            row.extend([""] * (len(headers) - len(row)))

            if is_rejected(row):
                aid = row[col_idx["ID"]] or current["article_id"]
                if aid:
                    configs_by_article.pop(aid, None)
                continue

            new_id = row[col_idx["ID"]]
            if new_id:
                if new_id != current["article_id"]:
                    current["dataset"] = ""
                    current["task_model"] = ""
                    current["missing_rate"] = ""
                current["article_id"] = new_id

            if row[col_idx["Title"]]:
                current["title"] = row[col_idx["Title"]]
            if row[col_idx["Dataset"]]:
                current["dataset"] = row[col_idx["Dataset"]]
            if row[col_idx["Task Models"]]:
                current["task_model"] = row[col_idx["Task Models"]]
            if row[col_idx["Missing rate"]]:
                current["missing_rate"] = row[col_idx["Missing rate"]]

            aid = current["article_id"]
            if not aid:
                continue

            if current["title"]:
                titles[aid] = current["title"]

            winners = {}
            for metric in metric_columns:
                idx = col_idx.get(metric)
                if idx is None:
                    continue
                value = row[idx].strip()
                if not value or len(value) > 80:
                    continue
                if REJECT_KEYWORD in value.upper():
                    continue
                winners[metric] = value

            if not winners:
                continue

            recon_w = {k: v for k, v in winners.items() if k in RECON_METRICS}
            task_w = {k: v for k, v in winners.items() if k in TASK_METRICS}

            rate, band, unknown_reason = classify_missing_rate(
                current["missing_rate"]
            )

            configs_by_article[aid].append(
                {
                    "article_id": aid,
                    "title": current["title"],
                    "dataset": current["dataset"],
                    "task_model": current["task_model"],
                    "missing_rate": current["missing_rate"],
                    "missing_rate_pct": rate if rate is not None else "",
                    "missing_rate_band": band,
                    "missing_rate_unknown_reason": unknown_reason,
                    "recon_winners": "; ".join(f"{k}={v}" for k, v in sorted(recon_w.items())),
                    "task_winners": "; ".join(f"{k}={v}" for k, v in sorted(task_w.items())),
                    "has_recon_winner": bool(recon_w),
                    "has_task_winner": bool(task_w),
                    "has_both_winners": bool(recon_w and task_w),
                }
            )

    return configs_by_article, titles


def join_tags(data: dict[str, dict]) -> set[str]:
    return set(data.keys())


def priority_score(row: dict) -> int:
    score = 0
    if row["has_recon_metrics"] and row["has_task_metrics"]:
        score += 40
    if row["has_task_models"] and row["has_recon_models"]:
        score += 25
    if row["summary_has_recon_and_task"]:
        score += 30
    if row["configs_with_both_winners_count"] > 0:
        score += 50
    if row["configs_count"] > 0:
        score += 5
    if row["application_prediction"] == "x":
        score += 3
    if row["application_classification"] == "x":
        score += 3
    return score


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, INPUT_DIR)
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    tagged: dict[str, dict[str, dict]] = {}
    for key, filename in TAGGING_FILES.items():
        tagged[key] = parse_tagging_file(os.path.join(input_dir, filename))

    summary = parse_summary_winners(os.path.join(input_dir, BEST_RECON_SUMMARY))
    configs_by_article, config_titles = parse_best_recon_configurations(
        os.path.join(input_dir, BEST_RECON_MODELS)
    )

    all_ids = set()
    for data in tagged.values():
        all_ids |= set(data.keys())
    all_ids |= set(summary.keys())
    all_ids |= set(configs_by_article.keys())

    article_rows = []
    config_rows = []

    for aid in sorted(all_ids, key=lambda x: int(x)):
        title = config_titles.get(aid, "")
        if not title:
            for key in tagged:
                entry = tagged[key].get(aid)
                if entry and entry.get("title"):
                    title = entry["title"]
                    break

        task_tags = tagged.get("task_models", {}).get(aid, {}).get("tags", set())
        recon_tags = tagged.get("recon_models", {}).get(aid, {}).get("tags", set())
        metric_tags = tagged.get("metrics_used", {}).get(aid, {}).get("tags", set())
        app_tags = tagged.get("applications", {}).get(aid, {}).get("tags", set())
        degrad_tags = tagged.get("degrad_type", {}).get(aid, {}).get("tags", set())
        degrad_m_tags = tagged.get("degrad_method", {}).get(aid, {}).get("tags", set())
        dataset_tags = tagged.get("datasets", {}).get(aid, {}).get("tags", set())
        bench_tags = tagged.get("benchmark_datasets", {}).get(aid, {}).get("tags", set())

        recon_metrics, task_metrics, other_metrics = split_metrics(metric_tags)
        winners = summary.get(aid, {})
        summary_recon = format_winners(winners, RECON_METRICS)
        summary_task = format_winners(winners, TASK_METRICS)

        configs = configs_by_article.get(aid, [])
        configs_both = sum(1 for c in configs if c["has_both_winners"])
        configs_recon_only = sum(1 for c in configs if c["has_recon_winner"] and not c["has_task_winner"])
        configs_task_only = sum(1 for c in configs if c["has_task_winner"] and not c["has_recon_winner"])

        row = {
            "article_id": aid,
            "title": title,
            "task_models": "; ".join(sorted(task_tags)),
            "recon_models": "; ".join(sorted(recon_tags)),
            "metrics_reconstruction": "; ".join(recon_metrics),
            "metrics_task": "; ".join(task_metrics),
            "metrics_other": "; ".join(other_metrics),
            "has_task_models": "x" if task_tags else "",
            "has_recon_models": "x" if recon_tags else "",
            "has_recon_metrics": "x" if recon_metrics else "",
            "has_task_metrics": "x" if task_metrics else "",
            "application_prediction": "x" if "Prediction" in app_tags else "",
            "application_classification": "x" if "Classification" in app_tags else "",
            "application_anomaly": "x" if "Anomaly detection" in app_tags else "",
            "degradation_types": "; ".join(sorted(degrad_tags)),
            "degradation_methods": "; ".join(sorted(degrad_m_tags)),
            "datasets": "; ".join(sorted(dataset_tags)),
            "benchmark_datasets": "; ".join(sorted(bench_tags)),
            "summary_recon_winners": summary_recon,
            "summary_task_winners": summary_task,
            "summary_has_recon_and_task": "x" if summary_recon and summary_task else "",
            "configs_count": len(configs),
            "configs_with_both_winners_count": configs_both,
            "configs_recon_only_count": configs_recon_only,
            "configs_task_only_count": configs_task_only,
            "candidate_priority": 0,
            "verification_hint": "",
        }

        row["candidate_priority"] = priority_score(row)

        hints = []
        if configs_both > 0:
            hints.append("W best data recon models sa wiersze z zwyciezcami recon+task")
        elif row["has_recon_metrics"] and row["has_task_metrics"]:
            hints.append("Raportuje oba typy metryk - sprawdz PDF pod tabele")
        elif row["summary_has_recon_and_task"]:
            hints.append("SUMMARY ma zwyciezcow recon i task na poziomie artykulu")
        elif row["has_task_models"] and row["has_recon_models"]:
            hints.append("Uzywa modeli task i recon - brak par wynikow w CSV")
        if not hints:
            hints.append("Niski priorytet dla RQ6-A3")
        row["verification_hint"] = " | ".join(hints)

        for col in MANUAL_COLUMNS:
            row[col] = ""

        article_rows.append(row)
        config_rows.extend(configs)

    article_rows.sort(key=lambda r: (-r["candidate_priority"], int(r["article_id"])))
    candidates = [r for r in article_rows if r["candidate_priority"] > 0]

    article_fields = [
        "article_id",
        "title",
        "candidate_priority",
        "verification_hint",
        "has_recon_metrics",
        "has_task_metrics",
        "has_task_models",
        "has_recon_models",
        "summary_has_recon_and_task",
        "configs_count",
        "configs_with_both_winners_count",
        "configs_recon_only_count",
        "configs_task_only_count",
        "task_models",
        "recon_models",
        "metrics_reconstruction",
        "metrics_task",
        "summary_recon_winners",
        "summary_task_winners",
        "application_prediction",
        "application_classification",
        "application_anomaly",
        "degradation_types",
        "degradation_methods",
        "datasets",
        "benchmark_datasets",
        "metrics_other",
    ] + MANUAL_COLUMNS

    config_fields = [
        "article_id",
        "title",
        "dataset",
        "task_model",
        "missing_rate",
        "missing_rate_pct",
        "missing_rate_band",
        "missing_rate_unknown_reason",
        "recon_winners",
        "task_winners",
        "has_recon_winner",
        "has_task_winner",
        "has_both_winners",
        "manual_verified",
        "manual_paired_recon_task",
        "manual_notes",
    ]

    all_path = os.path.join(output_dir, "verification_all_articles.csv")
    cand_path = os.path.join(output_dir, "verification_candidates.csv")
    cfg_path = os.path.join(output_dir, "verification_configurations.csv")
    readme_path = os.path.join(output_dir, "README.txt")

    for path, fields, rows in (
        (all_path, article_fields, article_rows),
        (cand_path, article_fields, candidates),
        (cfg_path, config_fields, config_rows),
    ):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Tabele do recznej weryfikacji RQ6-A3\n"
            "==================================\n\n"
            "Pliki:\n"
            "  verification_candidates.csv  - zacznij tutaj (priorytet > 0)\n"
            "  verification_all_articles.csv - wszystkie artykuly\n"
            "  verification_configurations.csv - wiersze z best data recon models\n\n"
            "Kolumny do uzupelnienia recznie (verification_*):\n"
            "  manual_verified            - tak / nie / czesciowo\n"
            "  manual_paired_recon_task   - tak jesli w artykule ta sama konfiguracja\n"
            "                               ma wynik recon i wynik zadania\n"
            "  manual_has_numeric_table   - tak jesli sa liczby (nie tylko nazwy algorytmow)\n"
            "  manual_pipeline_documented - tak jesli opisany jest lancuch imputacja->task\n"
            "  manual_notes               - dowolne uwagi\n\n"
            "Kolumny obliczone:\n"
            "  candidate_priority - wyzszy = wazniejszy do weryfikacji\n"
            "  configs_with_both_winners_count - ile wierszy ma zwyciezcow recon i task\n"
            "  missing_rate_band - przedzial co 10%: 0-10%, ..., 90-100%\n"
            "  missing_rate_unknown_reason - gdy band=nieznany:\n"
            "      brak_wartosci | nienumeryczna_wartosc | ujemna_wartosc | poza_zakresem\n\n"
            "Zrodla: input_csv/SLR(TAGGING-*.csv)\n"
        )

    print(f"Zapisano: {all_path} ({len(article_rows)} artykulow)")
    print(f"Zapisano: {cand_path} ({len(candidates)} kandydatow)")
    print(f"Zapisano: {cfg_path} ({len(config_rows)} konfiguracji)")
    print(f"Zapisano: {readme_path}")
    print(f"\nTop 5 kandydatow (priority):")
    for r in candidates[:5]:
        print(f"  [{r['article_id']}] prio={r['candidate_priority']} - {r['verification_hint'][:70]}")


if __name__ == "__main__":
    main()
