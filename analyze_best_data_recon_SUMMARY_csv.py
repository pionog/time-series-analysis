import csv
import os
from collections import Counter, defaultdict

# Ustawienia
input_folder = 'input_csv'  # Folder z plikami CSV do analizy
csv_output_folder = 'csv_results'  # Folder z plikami CSV do wykresów
txt_output_folder = 'txt_results'  # Folder na pliki tekstowe z wynikami
summary_keyword = 'SUMMARY'  # Nazwa plików do przetworzenia
file_name = "SLR(TAGGING-best data recon SUMMARY).csv"  # Nazwa pliku do analizy
sort_by_articles = False  # Jeśli True, sortuje według liczby artykułów i zamienia kolejność kolumn
# Wyświetlanie wyników i operacji na plikach
print_file_operations = False  # Jeśli True, wyświetla informacje o operacjach na plikach


def create_output_folder():
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)
        if print_file_operations:
            print(f"Utworzono folder: {csv_output_folder}")
    if not os.path.exists(txt_output_folder):
        os.makedirs(txt_output_folder)
        if print_file_operations:
            print(f"Utworzono folder: {txt_output_folder}")


def find_header_row(reader):
    for row in reader:
        if row and row[0].strip() == 'ID':
            return row
    return None


def analyze_summary_file(csv_file):
    input_path = os.path.join(input_folder, csv_file)
    output_file = os.path.splitext(csv_file)[0] + '_algorithm_summary.csv'
    csv_output_path = os.path.join(csv_output_folder, output_file)
    txt_output_path = os.path.join(txt_output_folder, output_file)

    if not os.path.exists(input_path):
        print(f"Plik {input_path} nie istnieje.")
        return

    with open(input_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        headers = find_header_row(reader)
        if headers is None:
            print(f"Nie znaleziono nagłówków w pliku {csv_file}.")
            return

        feature_columns = [h.strip() for h in headers[3:]]
        totals = Counter()
        per_col = defaultdict(Counter)
        article_rows = defaultdict(set)

        for row_index, row in enumerate(reader, start=1):
            if any('REJECTED:' in (cell or '').upper() for cell in row):
                continue

            row_key = row[0].strip() if row and row[0].strip() else f'row_{row_index}'
            for i in range(3, min(len(headers), len(row))):
                value = row[i].strip()
                if not value:
                    continue
                algorithm = value
                totals[algorithm] += 1
                column_name = headers[i].strip() or f'Column_{i}'
                per_col[algorithm][column_name] += 1
                article_rows[algorithm].add(row_key)

    if not totals:
        print(f"Brak algorytmów do zliczenia w pliku {csv_file}.")
        return

    sorted_algorithms = [alg for alg, _ in totals.most_common()]
    output_headers = ['Algorithm', 'Total', 'Articles'] + feature_columns

    with open(csv_output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_headers)
        writer.writeheader()
        for algorithm in sorted_algorithms:
            row = {
                'Algorithm': algorithm,
                'Total': totals[algorithm],
                'Articles': len(article_rows[algorithm]),
            }
            for col in feature_columns:
                row[col] = per_col[algorithm].get(col, 0)
            writer.writerow(row)

    txt_output_file = os.path.splitext(csv_file)[0] + '_algorithm_summary.txt'
    txt_output_path = os.path.join(txt_output_folder, txt_output_file)
    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Podsumowanie algorytmów z pliku: {csv_file}\n")
        f.write(f"Liczba unikalnych algorytmów: {len(sorted_algorithms)}\n")
        f.write(f"Łączna liczba wystąpień algorytmów: {sum(totals.values())}\n")
        f.write(f"Liczba artykułów z przetworzonych wierszy: {len({row_key for algorithm in article_rows for row_key in article_rows[algorithm]})}\n")
        f.write("\nAlgorytm; Łączne wystąpienia; Artykuły; liczba wystąpień w kolumnach\n")
        for algorithm in sorted_algorithms:
            counts_by_col = ', '.join(f"{col}: {per_col[algorithm].get(col, 0)}" for col in feature_columns if per_col[algorithm].get(col, 0) > 0)
            f.write(f"{algorithm}; {totals[algorithm]}; {len(article_rows[algorithm])}; {counts_by_col}\n")

    if print_file_operations:
        print(f"Plik wynikowy zapisano jako: {csv_output_path}")
        print(f"Plik tekstowy zapisano jako: {txt_output_path}")


if __name__ == '__main__':
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} nie istnieje.")
    else:
        create_output_folder()
        file = os.path.basename(file_name)
        if file in os.listdir(input_folder):
            analyze_summary_file(file)
