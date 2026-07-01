import os
import csv
import re

# Ustawienia
input_folder = "txt_results"
output_folder = "csv_results"
results_extension = "_feature_results.txt"


def create_output_folder():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Utworzono folder: {output_folder}")


def parse_total_articles(lines: list[str]) -> int | None:
    pattern = r"- liczba przeanalizowanych artykułów:\s*(\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            return int(match.group(1))
    return None


def convert_results_to_csv(results_file: str) -> None:
    input_path = os.path.join(input_folder, results_file)
    output_file = os.path.splitext(results_file)[0] + ".csv"
    output_path = os.path.join(output_folder, output_file)

    if not os.path.exists(input_path):
        print(f"Plik {input_path} nie istnieje.")
        return

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_articles = parse_total_articles(lines)
        data = []
        pattern = r"- Cecha '(.+?)' wystąpiła (\d+) razy"

        for line in lines:
            match = re.search(pattern, line)
            if match:
                cecha = match.group(1)
                liczba = int(match.group(2))
                procent = (
                    round(liczba / total_articles * 100, 2)
                    if total_articles
                    else ""
                )
                data.append(
                    {
                        "Cecha": cecha,
                        "Liczba": liczba,
                        "Procent": procent,
                        "Lacznie_artykulow": total_articles if total_articles else "",
                    }
                )

        if not data:
            print(f"Nie znaleziono danych w pliku {input_path}")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Cecha", "Liczba", "Procent", "Lacznie_artykulow"],
            )
            writer.writeheader()
            writer.writerows(data)

        print(f"Plik {results_file} został skonwertowany do {output_file}")

    except Exception as e:
        print(f"Błąd przy przetwarzaniu pliku {results_file}: {e}")


if __name__ == "__main__":
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} nie istnieje.")
    else:
        create_output_folder()
        for file in os.listdir(input_folder):
            if file.endswith(results_extension):
                print(f"Konwertowanie pliku: {file}")
                convert_results_to_csv(file)
