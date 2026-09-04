"""Grupy domen zastosowań (applications) — wspólne dla analiz i wykresów."""
from collections import Counter

from degradation_categories import TAG_TO_GENERAL, general_categories_from_tags

TASK_DOMAINS = {
    "Prediction", "Classification", "Anomaly detection", "other", "Other",
}

APPLICATION_DOMAIN_GROUPS: dict[str, list[str]] = {
    "Środowisko i klimat": ["Climate", "Air pollution"],
    "Chemia": ["Chemical"],
    "Finanse i gospodarka": ["stock exchange", "economic", "Finance", "business"],
    "Energia": [
        "Electric Load (mo?e lepiej szerzej Electricity)",
        "Power Engineering",
    ],
    "Medycyna i zdrowie": ["medicine"],
    "Transport": ["Traffic"],
    "Geografia": ["Geographic"],
    "Przemysł": ["Industry"],
    "Przemysł wytrzymałościowy": ["Endurance"],
    "Społeczne i behawioralne": [
        "social network",
        "household expenditures",
        "purchase habits (behavioral data)",
    ],
}

# Etykiety z arkusza pytań → nazwa grupy w APPLICATION_DOMAIN_GROUPS
QUESTION_LABEL_TO_GROUP: dict[str, str] = {
    "Wytrzymałość": "Przemysł wytrzymałościowy",
}

DOMAIN_QUESTION_IDS: dict[str, str] = {
    # RQ3 — problemy z jakością per domena (A9–A18 po usunięciu starego A9)
    "RQ3-A9": "Środowisko i klimat",
    "RQ3-A10": "Chemia",
    "RQ3-A11": "Finanse i gospodarka",
    "RQ3-A12": "Medycyna i zdrowie",
    "RQ3-A13": "Transport",
    "RQ3-A14": "Geografia",
    "RQ3-A15": "Przemysł",
    "RQ3-A16": "Wytrzymałość",
    "RQ3-A17": "Energia",
    "RQ3-A18": "Społeczne i behawioralne",
    # RQ4 — modele rekonstrukcji per domena
    "RQ4-A13": "Środowisko i klimat",
    "RQ4-A14": "Chemia",
    "RQ4-A15": "Finanse i gospodarka",
    "RQ4-A16": "Medycyna i zdrowie",
    "RQ4-A17": "Transport",
    "RQ4-A18": "Geografia",
    "RQ4-A19": "Przemysł",
    "RQ4-A20": "Wytrzymałość",
    "RQ4-A21": "Energia",
    "RQ4-A22": "Społeczne i behawioralne",
}

FEATURE_TO_GROUP: dict[str, str] = {}
for group, features in APPLICATION_DOMAIN_GROUPS.items():
    for feature in features:
        FEATURE_TO_GROUP[feature] = group


def resolve_domain_groups(label: str) -> list[str]:
    label = label.strip()
    if label in QUESTION_LABEL_TO_GROUP:
        return [QUESTION_LABEL_TO_GROUP[label]]
    if label in APPLICATION_DOMAIN_GROUPS:
        return [label]
    return []


def normalize_tags(tags: set[str]) -> set[str]:
    return {t.strip() for t in tags}


def article_in_group(app_tags: set[str], group: str) -> bool:
    tags = normalize_tags(app_tags)
    return any(f in tags or f.strip() in tags for f in APPLICATION_DOMAIN_GROUPS[group])


def count_articles_per_group(apps: dict[str, set[str]]) -> Counter:
    counts: Counter = Counter()
    for tags in apps.values():
        for group in APPLICATION_DOMAIN_GROUPS:
            if article_in_group(tags, group):
                counts[group] += 1
    return counts


def degradation_for_group(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    group: str,
) -> Counter:
    counts: Counter = Counter()
    for aid in set(apps) & set(degrad):
        if not article_in_group(apps[aid], group):
            continue
        for feat in degrad[aid]:
            counts[feat] += 1
    return counts


def general_degradation_for_group(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    group: str,
) -> Counter:
    counts: Counter = Counter()
    for aid in set(apps) & set(degrad):
        if not article_in_group(apps[aid], group):
            continue
        for cat in general_categories_from_tags(degrad[aid]):
            counts[cat] += 1
    return counts


def general_degradation_for_groups(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    groups: list[str],
) -> Counter:
    """Zlicza tagi degradacji w grupach, zagregowane do ogólnych kategorii."""
    combined: Counter = Counter()
    for group in groups:
        for tag, count in degradation_for_group(apps, degrad, group).items():
            category = TAG_TO_GENERAL.get(tag)
            if category:
                combined[category] += count
    return combined


def recon_models_for_group(
    apps: dict[str, set[str]],
    recon: dict[str, set[str]],
    group: str,
) -> Counter:
    counts: Counter = Counter()
    for aid in set(apps) & set(recon):
        if not article_in_group(apps[aid], group):
            continue
        for model in recon[aid]:
            counts[model] += 1
    return counts


def domain_general_degradation_matrix(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
) -> Counter:
    """Domena × ogólna kategoria degradacji (zliczenie tagów, spójne ze szczegółami)."""
    counts: Counter = Counter()
    for aid in set(apps) & set(degrad):
        groups = [g for g in APPLICATION_DOMAIN_GROUPS if article_in_group(apps[aid], g)]
        for group in groups:
            for tag in degrad[aid]:
                category = TAG_TO_GENERAL.get(tag)
                if category:
                    counts[(group, category)] += 1
    return counts


def structural_degradation_by_group(
    apps: dict[str, set[str]],
    degrad: dict[str, set[str]],
    structural: set[str],
) -> Counter:
    counts: Counter = Counter()
    for aid in set(apps) & set(degrad):
        groups = [g for g in APPLICATION_DOMAIN_GROUPS if article_in_group(apps[aid], g)]
        struct = {f for f in degrad[aid] if f in structural}
        for g in groups:
            for s in struct:
                counts[(g, s)] += 1
    return counts
