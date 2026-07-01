"""Kategoryzacja opisów transformacji (RQ2-A5)."""
import re

TRANSFORMATION_CATEGORY_ORDER = (
    "Imputacja i uzupełnianie braków",
    "Interpolacja i wypełnianie szeregów",
    "Normalizacja i skalowanie",
    "Dekompozycja i różnicowanie",
    "Resampling i agregacja czasowa",
    "Transformacje logarytmiczne i nieliniowe",
    "Redukcja wymiarów i reprezentacja cech",
    "Inne / specyficzne dla domeny",
)

_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Imputacja i uzupełnianie braków",
        (
            r"\bimput",
            r"missing data",
            r"missing value",
            r"fill.*nan",
            r"nan.*fill",
            r"not-a-number",
            r"replace.*nan",
            r"bad data such as null",
        ),
    ),
    (
        "Interpolacja i wypełnianie szeregów",
        (r"interpolat", r"linear interpolation"),
    ),
    (
        "Normalizacja i skalowanie",
        (
            r"normaliz",
            r"standardiz",
            r"standardised",
            r"\bscaled\b",
            r"min.*max",
            r"unit variance",
            r"zero-mean",
            r"z-score",
        ),
    ),
    (
        "Dekompozycja i różnicowanie",
        (
            r"decompos",
            r"\bceemd",
            r"\bemd\b",
            r"\bssa\b",
            r"differenc",
            r"detrend",
            r"exponential smoothing",
            r"moving averages",
        ),
    ),
    (
        "Resampling i agregacja czasowa",
        (
            r"resampl",
            r"re-sampled",
            r"downsampl",
            r"aggregat",
            r"mean 24 hours",
            r"resolution data",
        ),
    ),
    (
        "Transformacje logarytmiczne i nieliniowe",
        (r"\blog\b", r"sigmoid", r"heteroscedasticity", r"ln data"),
    ),
    (
        "Redukcja wymiarów i reprezentacja cech",
        (
            r"vectoriz",
            r"kernel",
            r"granul",
            r"sparse representation",
            r"feature space",
            r"weighting.*state",
            r"fusion",
        ),
    ),
]


def categorize_transformation(text: str) -> str:
    lowered = text.lower()
    for category, patterns in _RULES:
        if any(re.search(p, lowered) for p in patterns):
            return category
    return "Inne / specyficzne dla domeny"
