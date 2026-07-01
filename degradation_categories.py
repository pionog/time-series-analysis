"""Mapowanie szczegółowych tagów degradacji na ogólne kategorie."""

GENERAL_DEGRADATION_CATEGORIES: dict[str, set[str]] = {
    "Sposób powstawania degradacji": {"Natural", "Synthetic", "Unknown Degradation"},
    "Missingness": {"MNAR", "MAR", "MCAR", "Unknown missingness"},
    "Zniekształcenia strukturalne": {
        "Drift", "Interrupted time series", "Non-stationarity", "Changepoint",
        "Non-linear", "Outliers",
    },
    "Szum": {"Additive distortions", "White Noise", "Gaussian Noise", "Unknown noise"},
    "Misalignment": {"Misalignments"},
    "Metoda zakłócania (jawnie opisana)": {
        "Random pattern", "Downsampling", "Outliers removal", "Regular pattern",
    },
}

TAG_TO_GENERAL: dict[str, str] = {}
for category, tags in GENERAL_DEGRADATION_CATEGORIES.items():
    for tag in tags:
        TAG_TO_GENERAL[tag] = category


def heatmap_general_columns() -> list[str]:
    """Kolumny heatmap bez skrajnych kategorii (sposób powstawania, metoda zakłócania)."""
    keys = list(GENERAL_DEGRADATION_CATEGORIES.keys())
    return keys[1:-1] if len(keys) > 2 else keys


def general_categories_from_tags(tags: set[str]) -> set[str]:
    return {TAG_TO_GENERAL[t] for t in tags if t in TAG_TO_GENERAL}


def count_general_categories(degrad: dict[str, set[str]]) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in GENERAL_DEGRADATION_CATEGORIES}
    for tags in degrad.values():
        for cat in general_categories_from_tags(tags):
            counts[cat] += 1
    return {k: v for k, v in counts.items() if v > 0}
