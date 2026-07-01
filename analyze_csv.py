import csv
import os
from collections import Counter

# Ustawienia globalne
save_results = True  # Jeśli True, zapisuje wyniki do pliku tekstowego
reject_keyword = "REJECTED"  # Artykuły z tym słowem są pomijane
full_analysis_list_file = "full_analysis_files.txt"  # Lista plików z pełną analizą
# Foldery
input_dir = "input_csv"  # Folder z plikami CSV
output_dir = "txt_results"  # Folder na pliki z wynikami
# Wyświetlanie wyników i operacji na plikach
print_summary = False  # Jeśli True, wyświetla wyniki w konsoli
print_file_operations = False  # Jeśli True, wyświetla informacje o operacjach na plikach


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
        required_cols = ["ID", "Title"]
        missing_required = [c for c in required_cols if c not in col_idx]
        if missing_required:
            raise ValueError(
                f"Brak wymaganych kolumn w pliku {csv_file}: {', '.join(missing_required)}"
            )

        base_cols = {"ID", "Title", "Tagger", "Comment", "Comments"}
        feature_columns = [h for h in headers if h and h not in base_cols]

        per_id = {}
        rejected_ids = set()
        for raw_row in reader:
            if not raw_row:
                continue

            row = [cell.strip() for cell in raw_row]
            if len(row) < len(headers):
                row.extend([""] * (len(headers) - len(row)))

            article_id = row[col_idx["ID"]]
            if not article_id:
                continue

            if any(reject_keyword in cell.upper() for cell in row if cell):
                rejected_ids.add(article_id)
                per_id.pop(article_id, None)
                continue

            if article_id in rejected_ids:
                continue

            title = row[col_idx["Title"]].strip() or "Brak tytułu"
            if article_id not in per_id:
                per_id[article_id] = {
                    "title": title,
                    "matched_features": [],
                }

            article = per_id[article_id]
            if article["title"] == "Brak tytułu" and title != "Brak tytułu":
                article["title"] = title

            for feature in feature_columns:
                idx = col_idx[feature]
                if row[idx].lower() == "x":
                    article["matched_features"].append(feature)

    return per_id


def format_features(per_id):
    lines = []
    global_counter = Counter()

    for article in per_id.values():
        global_counter.update(article["matched_features"])

    total_articles = len(per_id)
    lines.append("Podsumowanie cech (globalnie)")
    lines.append(f"- liczba przeanalizowanych artykułów: {total_articles}")
    lines.append(f"- liczba unikalnych cech: {len(global_counter)}")
    lines.append(f"- łączna liczba oznaczeń cech (x): {sum(global_counter.values())}")
    if global_counter:
        for feature_name, count in global_counter.most_common():
            lines.append(f"- Cecha '{feature_name}' wystąpiła {count} razy")
    else:
        lines.append("- Brak wykrytych cech")
    lines.append("")

    for article_id in sorted(per_id, key=lambda x: int(x) if x.isdigit() else x):
        article = per_id[article_id]
        title = article["title"] or "Brak tytułu"
        counter = Counter(article["matched_features"])

        lines.append(f"ID {article_id} | {title}")
        lines.append(f"- liczba zaznaczonych cech (x): {sum(counter.values())}")
        lines.append(f"- liczba unikalnych cech: {len(counter)}")
        if counter:
            lines.append(
                "- cechy: "
                + ", ".join(
                    f"{name} ({count})" for name, count in counter.most_common()
                )
            )
        else:
            lines.append("- cechy: brak")
        lines.append("")

    return lines


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("cp1250", errors="replace").decode("cp1250")
        print(fallback)


def load_full_analysis_files(config_file_path):
    if not os.path.exists(config_file_path):
        return set()

    files = set()
    with open(config_file_path, "r", encoding="utf-8", errors="ignore") as config_file:
        for line in config_file:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            files.add(entry)
    return files


def analyze_file(csv_file, results_directory, full_analysis_files):
    csv_base_name = os.path.basename(csv_file)
    if csv_base_name in full_analysis_files:
        if print_file_operations:
            print(f"Pominięto pełną analizę pliku: {csv_base_name}")
        return

    output_file = os.path.join(
        results_directory, os.path.splitext(csv_base_name)[0] + "_feature_results.txt"
    )

    per_id = parse_features(csv_file)
    results = format_features(per_id)

    if print_summary:
        for result in results:
            safe_print(result)

    if save_results:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(result + "\n")
        if print_file_operations:
            print(f"Wyniki zapisane do {output_file}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_directory = os.path.join(script_dir, input_dir)
    results_directory = os.path.join(script_dir, output_dir)
    full_analysis_config_path = os.path.join(script_dir, full_analysis_list_file)
    full_analysis_files = load_full_analysis_files(full_analysis_config_path)

    if not os.path.isdir(csv_directory):
        raise FileNotFoundError(f"Folder z plikami CSV nie istnieje: {csv_directory}")
    os.makedirs(results_directory, exist_ok=True)

    for file in os.listdir(csv_directory):
        if not file.endswith(".csv"):
            continue
        
        if print_file_operations:
            print(f"Przetwarzanie pliku: {file}")
        analyze_file(
            os.path.join(csv_directory, file),
            results_directory,
            full_analysis_files,
        )