"""
Wspólna logika przedziałów missing rate dla analizy SLR (RQ6-A3, RQ6-A4).

Przedziały: co 10% od 0% do 100%:
  0-10%, 10-20%, 20-30%, ..., 80-90%, 90-100%

Przypisanie wartości (przykłady):
  0%   -> 0-10%
  9.9% -> 0-10%
  10%  -> 10-20%
  90%  -> 90-100%
  100% -> 90-100%

Nieznany procent (etykieta „nieznany”) — konfiguracja nie trafia do żadnego
przedziału 10%, gdy:
  - brak_wartosci          — pusta komórka Missing rate w CSV
  - nienumeryczna_wartosc  — tekst, którego nie da się zamienić na liczbę (%)
  - ujemna_wartosc         — wartość < 0%
  - poza_zakresem          — wartość > 100%
"""

BAND_STEP = 10
BAND_MIN = 0
BAND_MAX = 100
UNKNOWN_BAND = "nieznany"

MISSING_RATE_BANDS: tuple[tuple[str, float, float], ...] = tuple(
    (f"{low}-{low + BAND_STEP}%", float(low), float(low + BAND_STEP))
    for low in range(BAND_MIN, BAND_MAX, BAND_STEP)
)

BAND_LABELS: tuple[str, ...] = tuple(label for label, _, _ in MISSING_RATE_BANDS) + (
    UNKNOWN_BAND,
)


def parse_missing_rate(value: str) -> float | None:
    """Zamienia '25', '2,5', '12,5' na float (procent)."""
    if not value or not str(value).strip():
        return None
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def missing_rate_unknown_reason(raw_value: str, rate: float | None) -> str:
    """Powód klasyfikacji jako nieznany; pusty string gdy procent jest znany."""
    if not raw_value or not str(raw_value).strip():
        return "brak_wartosci"
    if rate is None:
        return "nienumeryczna_wartosc"
    if rate < BAND_MIN:
        return "ujemna_wartosc"
    if rate > BAND_MAX:
        return "poza_zakresem"
    return ""


def band_label_for_rate(rate: float) -> str:
    """Mapuje znaną wartość procentową [0, 100] na przedział 10%."""
    if rate == BAND_MAX:
        return MISSING_RATE_BANDS[-1][0]
    bucket_low = int(rate // BAND_STEP) * BAND_STEP
    if bucket_low >= BAND_MAX:
        bucket_low = BAND_MAX - BAND_STEP
    return f"{bucket_low}-{bucket_low + BAND_STEP}%"


def classify_missing_rate(
    raw_value: str,
) -> tuple[float | None, str, str]:
    """
    Zwraca (rate_float_lub_None, band_label, unknown_reason).
    unknown_reason jest pusty, gdy band_label != nieznany.
    """
    rate = parse_missing_rate(raw_value)
    reason = missing_rate_unknown_reason(raw_value, rate)
    if reason:
        return rate, UNKNOWN_BAND, reason
    assert rate is not None
    return rate, band_label_for_rate(rate), ""


def missing_rate_band(raw_value: str) -> str:
    """Zwraca etykietę przedziału na podstawie surowej wartości z CSV."""
    return classify_missing_rate(raw_value)[1]
