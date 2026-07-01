import csv
import os
from collections import Counter, defaultdict


# Ustawienia globalne
print_results = True  # Jeśli True, zapisuje wyniki do pliku tekstowego
reject_keyword = "REJECTED"  # Artykuły z tym słowem są pomijane
input_dir = "input_csv"  # Folder z plikami CSV
output_dir = "txt_results"  # Folder na pliki z wynikami
file_name = "SLR(TAGGING-best data recon models).csv"  # Nazwa pliku do analizy
print_summary = False  # Jeśli True, wyświetla wyniki w konsoli
print_file_operations = True  # Jeśli True, wyświetla informacje o operacjach na plikach


def parse_features(csv_file):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Plik {csv_file} nie istnieje.")

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file, delimiter=";")

        headers = None
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                break

        if headers is None:
            raise ValueError("Nie znaleziono nagłówka zaczynającego się od 'ID'.")

        col_idx = {name: i for i, name in enumerate(headers) if name}
        required_cols = ["ID", "Title", "Dataset", "Task Models", "Missing rate"]
        missing_required = [c for c in required_cols if c not in col_idx]
        if missing_required:
            raise ValueError(
                f"Brak wymaganych kolumn w pliku {csv_file}: {', '.join(missing_required)}"
            )

        metric_columns = [
            h
            for h in headers
            if h
            and h
            not in {"ID", "Title", "Tagger", "Dataset", "Task Models", "Order", "Timing", "Missing rate", "Batch size", "Comment"}
        ]

        per_id = {}
        rejected_ids = set()
        current = {"id": "", "title": "", "dataset": "", "task_model": "", "missing_rate": ""}

        for raw_row in reader:
            if not raw_row:
                continue

            row = [cell.strip() for cell in raw_row]
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))

            new_id = row[col_idx["ID"]]
            if new_id:
                if new_id != current["id"]:
                    current["dataset"] = ""
                    current["task_model"] = ""
                    current["missing_rate"] = ""
                current["id"] = new_id
            if row[col_idx["ID"]]:
                current["id"] = row[col_idx["ID"]]
            if row[col_idx["Title"]]:
                current["title"] = row[col_idx["Title"]]
            if row[col_idx["Dataset"]]:
                current["dataset"] = row[col_idx["Dataset"]]
            if row[col_idx["Task Models"]]:
                current["task_model"] = row[col_idx["Task Models"]]
            if row[col_idx["Missing rate"]]:
                current["missing_rate"] = row[col_idx["Missing rate"]]

            if not current["id"]:
                continue

            if any(reject_keyword in cell.upper() for cell in row if cell):
                rejected_ids.add(current["id"])
                per_id.pop(current["id"], None)
                continue

            if current["id"] in rejected_ids:
                continue

            if current["id"] not in per_id:
                per_id[current["id"]] = {
                    "title": current["title"],
                    "datasets": set(),
                    "task_models": set(),
                    "missing_rates": set(),
                    "rows_with_data": 0,
                    "metrics_used": Counter(),
                    "winners_by_metric": defaultdict(Counter),
                }

            article = per_id[current["id"]]
            if current["title"] and not article["title"]:
                article["title"] = current["title"]

            meaningful_row = False

            if current["dataset"]:
                article["datasets"].add(current["dataset"])
                meaningful_row = True
            if current["task_model"]:
                article["task_models"].add(current["task_model"])
                meaningful_row = True
            if current["missing_rate"]:
                article["missing_rates"].add(current["missing_rate"])
                meaningful_row = True

            for metric in metric_columns:
                idx = col_idx.get(metric)
                if idx is None:
                    continue
                value = row[idx].strip()
                if value:
                    article["metrics_used"][metric] += 1
                    article["winners_by_metric"][metric][value] += 1
                    meaningful_row = True

            if meaningful_row:
                article["rows_with_data"] += 1

    return per_id


def format_features(per_id):
    lines = []
    for article_id in sorted(per_id, key=lambda x: int(x) if x.isdigit() else x):
        article = per_id[article_id]
        title = article["title"] or "Brak tytułu"

        lines.append(f"ID {article_id} | {title}")
        lines.append(f"- liczba konfiguracji (wierszy): {article['rows_with_data']}")
        lines.append(f"- liczba datasetów: {len(article['datasets'])}")
        if article["datasets"]:
            lines.append(f"  datasety: {', '.join(sorted(article['datasets']))}")

        lines.append(f"- liczba modeli bazowych: {len(article['task_models'])}")
        if article["task_models"]:
            lines.append(f"  modele: {', '.join(sorted(article['task_models']))}")

        lines.append(f"- liczba poziomów missing rate: {len(article['missing_rates'])}")
        if article["missing_rates"]:
            lines.append(f"  missing rate: {', '.join(sorted(article['missing_rates']))}")

        if article["metrics_used"]:
            metrics = ", ".join(
                f"{metric} ({count})"
                for metric, count in article["metrics_used"].most_common()
            )
            lines.append(f"- użyte metryki: {metrics}")
        else:
            lines.append("- użyte metryki: brak")

        for metric, winners in article["winners_by_metric"].items():
            top_winner, top_count = winners.most_common(1)[0]
            lines.append(f"  - top dla {metric}: {top_winner} ({top_count})")

        lines.append("")

    return lines


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("cp1250", errors="replace").decode("cp1250")
        print(fallback)


def analyze_file(csv_file, results_directory):
    output_file = os.path.join(
        results_directory,
        os.path.splitext(os.path.basename(csv_file))[0] + "_feature_results.txt",
    )

    per_id = parse_features(csv_file)
    results = format_features(per_id)

    if print_summary:
        for result in results:
            safe_print(result)

    if print_results:
        os.makedirs(results_directory, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result + "\n")
        if print_file_operations:
            print(f"Wyniki zapisane do {output_file}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_directory = os.path.join(script_dir, input_dir)
    results_directory = os.path.join(script_dir, output_dir)
    csv_file_path = os.path.join(csv_directory, file_name)

    if not os.path.exists(csv_file_path):
        print(f"Plik {csv_file_path} nie istnieje.")
    else:
        if print_file_operations:
            print(f"Przetwarzanie pliku: {file_name}")
        analyze_file(csv_file_path, results_directory)