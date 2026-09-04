"""Wczytywanie mapowania etykiet EN→PL z pliku tekstowego."""
from __future__ import annotations

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABELS_FILE = os.path.join(SCRIPT_DIR, "edytowalne", "etykiety_pl.txt")

_cache: dict[str, str] | None = None
_cache_lower: dict[str, str] | None = None


def load_label_mapping(path: str | None = None) -> dict[str, str]:
    """Zwraca słownik {etykieta_źródłowa: etykieta_polska}."""
    global _cache, _cache_lower
    path = path or DEFAULT_LABELS_FILE
    if _cache is not None and path == DEFAULT_LABELS_FILE:
        return _cache

    mapping: dict[str, str] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ";" not in line:
                    continue
                source, target = line.split(";", 1)
                source = source.strip()
                target = target.strip()
                if source and target:
                    mapping[source] = target

    if path == DEFAULT_LABELS_FILE:
        _cache = mapping
        _cache_lower = {k.lower(): v for k, v in mapping.items()}
    return mapping


def translate_label(label: str, mapping: dict[str, str] | None = None) -> str:
    if mapping is None:
        if _cache is not None:
            if label in _cache:
                return _cache[label]
            if _cache_lower and label.lower() in _cache_lower:
                return _cache_lower[label.lower()]
            return label
        mapping = load_label_mapping()

    if label in mapping:
        return mapping[label]
    lower_map = {k.lower(): v for k, v in mapping.items()}
    return lower_map.get(label.lower(), label)


def translate_labels(labels: list[str], mapping: dict[str, str] | None = None) -> list[str]:
    return [translate_label(label, mapping) for label in labels]
