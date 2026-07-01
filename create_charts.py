import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from domain_groups import APPLICATION_DOMAIN_GROUPS, FEATURE_TO_GROUP, TASK_DOMAINS

# Ustawienia
input_folder = "csv_results"
output_folder = "charts"
csv_extension = "_feature_results.csv"

# Parametry wykresu
CHART_WIDTH = 14
CHART_HEIGHT = 7
COLOR = "#2E86AB"
OTHER_COLOR = "#A8BCC8"
EDGE_COLOR = "#1A3A52"
TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 10
LABEL_FONTSIZE = 11
ROTATION = 45
DPI = 120
FILE_FORMAT = "png"
TOP_N = 10
OTHER_LABEL = "Other"

PROBLEM_CATEGORY_GROUPS = {
    "Zadanie: Prognoza": ["Prediction"],
    "Zadanie: Klasyfikacja": ["Classification"],
    "Zadanie: Wykrywanie anomalii": ["Anomaly detection"],
    "Zadanie: Inne": ["other", "Other"],
}

def _application_domain_tags() -> set[str]:
    tags: set[str] = set()
    for features in APPLICATION_DOMAIN_GROUPS.values():
        tags.update(features)
    return tags


APPLICATION_DOMAIN_TAGS = _application_domain_tags()

FEATURE_TO_GROUP.update(
    {feature: group for group, features in PROBLEM_CATEGORY_GROUPS.items() for feature in features}
)


def create_output_folder():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Utworzono folder: {output_folder}")


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Procent" not in df.columns and "Lacznie_artykulow" in df.columns:
        total = df["Lacznie_artykulow"].iloc[0]
        if pd.notna(total) and total:
            df["Procent"] = (df["Liczba"] / float(total) * 100).round(2)
    return df


def get_total_articles(df: pd.DataFrame) -> int | None:
    if "Lacznie_artykulow" not in df.columns:
        return None
    value = df["Lacznie_artykulow"].iloc[0]
    if pd.isna(value) or value == "":
        return None
    return int(value)


def generalize_application_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Łączy szczegółowe domeny w szersze kategorie."""
    grouped_counts: dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        feature = row["Cecha"]
        count = int(row["Liczba"])
        group = FEATURE_TO_GROUP.get(feature, "Inne domeny")
        grouped_counts[group] += count

    total = get_total_articles(df)
    rows = []
    for name, count in grouped_counts.items():
        procent = round(count / total * 100, 2) if total else 0
        rows.append(
            {
                "Cecha": name,
                "Liczba": count,
                "Procent": procent,
                "Lacznie_artykulow": total,
            }
        )
    return pd.DataFrame(rows).sort_values("Procent", ascending=False)


def apply_top_n_with_other(
    df: pd.DataFrame, top_n: int = TOP_N
) -> pd.DataFrame:
    """Zostawia TOP N pozycji; resztę sumuje w Other."""
    df = df.sort_values("Procent", ascending=False).reset_index(drop=True)
    if len(df) <= top_n:
        return df

    top = df.iloc[:top_n].copy()
    rest = df.iloc[top_n:]
    other_row = {
        "Cecha": OTHER_LABEL,
        "Liczba": int(rest["Liczba"].sum()),
        "Procent": round(float(rest["Procent"].sum()), 2),
        "Lacznie_artykulow": df["Lacznie_artykulow"].iloc[0]
        if "Lacznie_artykulow" in df.columns
        else "",
    }
    return pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)


def plot_percentage_bars(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    *,
    total_articles: int | None,
    note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(CHART_WIDTH, CHART_HEIGHT), dpi=DPI)

    labels = df["Cecha"].tolist()
    percents = df["Procent"].tolist()
    counts = df["Liczba"].tolist()
    colors = [OTHER_COLOR if label == OTHER_LABEL else COLOR for label in labels]

    bars = ax.bar(
        labels,
        percents,
        color=colors,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
    )

    ax.set_ylabel("Udział w artykułach (%)", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Kategoria", fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=16)

    if total_articles:
        subtitle = f"Przeanalizowano łącznie {total_articles} artykułów"
        if note:
            subtitle += f" | {note}"
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=SUBTITLE_FONTSIZE,
            color="#444444",
        )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.xticks(rotation=ROTATION, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    ymax = max(percents) * 1.15 if percents else 1
    ax.set_ylim(0, ymax)

    for bar, pct, count in zip(bars, percents, counts):
        label = f"{pct:.1f}%"
        if total_articles:
            label += f"\n(n={int(count)})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    footer = (
        f"Wykres: TOP {TOP_N} + {OTHER_LABEL} (jeśli dotyczy). "
        "Pełne dane w tabeli csv_results."
    )
    fig.text(0.01, 0.01, footer, fontsize=8, color="#666666")

    plt.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def create_chart(csv_file: str) -> None:
    input_path = os.path.join(input_folder, csv_file)
    if not os.path.exists(input_path):
        print(f"Plik {input_path} nie istnieje.")
        return

    try:
        df = load_dataframe(input_path)
        total_articles = get_total_articles(df)

        note = ""
        if "applications" in csv_file.lower():
            df = generalize_application_domains(df)
            note = "domeny uogólnione"

        df = apply_top_n_with_other(df)

        title_base = os.path.splitext(csv_file)[0].replace("_feature_results", "")
        title = f"Rozkład cech (% artykułów): {title_base}"

        output_filename = os.path.splitext(csv_file)[0] + f".{FILE_FORMAT}"
        output_path = os.path.join(output_folder, output_filename)

        plot_percentage_bars(
            df,
            title,
            output_path,
            total_articles=total_articles,
            note=note,
        )
        print(f"Wykres zapisany: {output_path}")

    except Exception as e:
        print(f"Błąd przy tworzeniu wykresu dla {csv_file}: {e}")


def create_summary_chart() -> None:
    all_features: dict[str, int] = defaultdict(int)
    total_articles = None

    try:
        for file in os.listdir(input_folder):
            if not file.endswith(csv_extension):
                continue
            df = load_dataframe(os.path.join(input_folder, file))
            if total_articles is None:
                total_articles = get_total_articles(df)
            for _, row in df.iterrows():
                all_features[row["Cecha"]] += int(row["Liczba"])

        if not all_features:
            print("Brak danych do utworzenia łącznego wykresu.")
            return

        rows = []
        for name, count in all_features.items():
            procent = (
                round(count / total_articles * 100, 2)
                if total_articles
                else 0
            )
            rows.append(
                {
                    "Cecha": name,
                    "Liczba": count,
                    "Procent": procent,
                    "Lacznie_artykulow": total_articles,
                }
            )

        df = pd.DataFrame(rows)
        df = apply_top_n_with_other(df)

        output_path = os.path.join(output_folder, f"SUMMARY_all_features.{FILE_FORMAT}")
        plot_percentage_bars(
            df,
            "Podsumowanie cech — wszystkie pliki (% względem artykułów)",
            output_path,
            total_articles=total_articles,
            note="agregat wszystkich tagów",
        )
        print(f"Wykres podsumowania zapisany: {output_path}")

    except Exception as e:
        print(f"Błąd przy tworzeniu wykresu podsumowania: {e}")


if __name__ == "__main__":
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} nie istnieje.")
    else:
        create_output_folder()

        print("\nTworzenie indywidualnych wykresów...")
        for file in sorted(os.listdir(input_folder)):
            if file.endswith(csv_extension):
                print(f"Przetwarzanie: {file}")
                create_chart(file)

        print("\nTworzenie wykresu podsumowania...")
        create_summary_chart()

        print("\nGotowe! Wykresy zapisane w folderze:", output_folder)
