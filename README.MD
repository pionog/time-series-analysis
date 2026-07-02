# Analiza skoroszytów CSV (SLR)

Projekt służy do analizy otagowanych artykułów z przeglądu literatury: zliczanie cech, generowanie wykresów, reguł RQ6 oraz strony HTML z odpowiedziami na pytania badawcze.

## Wymagania

- Windows, Python **3.12**
- Zależności: `pandas`, `matplotlib` (plik `requirements.txt`)

## Środowisko wirtualne (`.venv`)

Po sklonowaniu lub przeniesieniu projektu utwórz lub odtwórz venv w katalogu projektu:

```powershell
cd ścieżka\do\Analiza
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Sprawdzenie:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import pandas, matplotlib; print('OK')"
```

Wszystkie poniższe komendy zakładają, że jesteś w katalogu projektu i używasz interpretera z `.venv`:

```powershell
.\.venv\Scripts\python.exe nazwa_skryptu.py
```

## Struktura folderów

| Folder | Zawartość |
|--------|-----------|
| `input_csv/` | Skoroszyty źródłowe (tagowanie artykułów) |
| `txt_results/` | Wyniki tekstowe analiz |
| `csv_results/` | Wyniki skonwertowane do CSV (dla wykresów) |
| `charts/` | Wykresy PNG |
| `rules_results/` | Reguły i dane RQ6 (A3, A4) |
| `rq6_a3_verification/` | Tabele do ręcznej weryfikacji RQ6-A3 |
| `html/` | Pytania badawcze, strona wyników, status pytań |

## Kolejność uruchamiania (pełny pipeline)

Po zmianie plików w `input_csv/` uruchom skrypty w tej kolejności:

```powershell
.\.venv\Scripts\python.exe analyze_csv.py
.\.venv\Scripts\python.exe analyze_best_data_recon_models_csv.py
.\.venv\Scripts\python.exe analyze_best_data_recon_SUMMARY_csv.py
.\.venv\Scripts\python.exe convert_results_to_csv.py
.\.venv\Scripts\python.exe create_charts.py
.\.venv\Scripts\python.exe extract_rules_from_best_data_recon.py
.\.venv\Scripts\python.exe build_rules_graph.py
.\.venv\Scripts\python.exe build_rq6_a3_verification_table.py
.\.venv\Scripts\python.exe build_research_html.py
.\.venv\Scripts\python.exe export_questions_status.py
```

Opcjonalnie (raport diagnostyczny na konsoli):

```powershell
.\.venv\Scripts\python.exe analyze_rq6_a3_gap.py
```

## Opis skryptów

### Analiza podstawowa

| Skrypt | Wejście | Wyjście | Opis |
|--------|---------|---------|------|
| `analyze_csv.py` | `input_csv/*.csv` | `txt_results/*_feature_results.txt` | Zlicza oznaczenia `x` w kolumnach cech dla wszystkich skoroszytów tagowania. Pomija artykuły z `REJECTED`. |
| `analyze_best_data_recon_models_csv.py` | `input_csv/SLR(TAGGING-best data recon models).csv` | `txt_results/…_feature_results.txt` | Szczegółowa analiza per artykuł (konfiguracje, datasety, missing rate). |
| `analyze_best_data_recon_SUMMARY_csv.py` | `input_csv/SLR(TAGGING-best data recon SUMMARY).csv` | `csv_results/…_algorithm_summary.csv`, `txt_results/…` | Ranking algorytmów według kolumn metryk. |
| `convert_results_to_csv.py` | `txt_results/*_feature_results.txt` | `csv_results/*.csv` | Konwersja wyników tekstowych do CSV. **Uwaga:** plik `best data recon models` ma inny format i nie jest konwertowany. |
| `create_charts.py` | `csv_results/*_feature_results.csv` | `charts/*.png` | Wykresy słupkowe + wykres podsumowania. Wymaga `pandas` i `matplotlib`. |

### Reguły i weryfikacja RQ6

| Skrypt | Wejście | Wyjście | Opis |
|--------|---------|---------|------|
| `extract_rules_from_best_data_recon.py` | `input_csv/…best data recon models.csv` | `rules_results/` | Reguły RQ6-A3/A4, `evidence_long.csv`, `configurations.csv`. |
| `build_rq6_a3_verification_table.py` | `input_csv/` | `rq6_a3_verification/` | Tabele kandydatów do ręcznej weryfikacji RQ6-A3. |
| `analyze_rq6_a3_gap.py` | `input_csv/` | konsola | Podsumowanie luk danych dla RQ6-A3. |

### Pytania badawcze (HTML)

| Skrypt | Wejście | Wyjście | Opis |
|--------|---------|---------|------|
| `build_research_html.py` | `html/Pytania badawcze(…).csv`, `csv_results/`, `input_csv/`, `rules_results/` | `html/wyniki_badawcze.html` | Strona z odpowiedziami na pytania badawcze i analityczne. |
| `export_questions_status.py` | (jak wyżej) | `html/status_pytan.txt` | Lista pytań ze statusem: [TAK] / [CZĘŚCIOWO] / [NIE]. |
| `build_rules_graph.py` | `rules_results/` | `html/reguly_graf.html`, `edytowalne/reguly_eksperyment.csv` | Interaktywne drzewo decyzyjne i graf reguł RQ6-A4. |

### Moduły pomocnicze (nie uruchamiać bezpośrednio)

| Plik | Rola |
|------|------|
| `missing_rate_utils.py` | Klasyfikacja poziomów missing rate |
| `domain_groups.py` | Grupy domen zastosowań (applications) |

## Pliki konfiguracyjne

- `full_analysis_files.txt` — lista skoroszytów z rozszerzoną analizą (używane przez `analyze_csv.py`)
- `html/Pytania badawcze(Pytania badawcze).csv` — pytania badawcze i analityczne

## Wyniki do przeglądu

- **HTML:** `html/wyniki_badawcze.html` — otwórz w przeglądarce
- **Graf reguł:** `html/reguly_graf.html` — drzewo decyzyjne i graf RQ6-A4
- **Status pytań:** `html/status_pytan.txt`
- **Wykresy:** `charts/`

## Konwencje analizy (`analyze_csv.py`)

- Dla plików **bez** prefiksu `basic_` — analiza zawartości komórek (np. datasety, modele, missing rate).
- Dla plików **z** prefiksem `basic_` — proste zliczanie oznaczeń `x`.
- Artykuły zawierające `REJECTED` są pomijane.

## Git i GitHub

Repozytorium jest przygotowane do wersjonowania: katalog `.venv/` jest ignorowany (każdy klon odtwarza środowisko lokalnie).

### Pierwszy commit (lokalnie)

Repozytorium jest już zainicjowane (`git init`). Wystarczy dodać pliki i zatwierdzić stan:

```powershell
cd ścieżka\do\Analiza
git add .
git status
git commit -m "Początkowy stan projektu analizy SLR"
```

### Publikacja na GitHub

1. Utwórz puste repozytorium na GitHub (bez README, bez `.gitignore` — są już w projekcie).
2. Podłącz zdalne i wypchnij:

```powershell
git branch -M main
git remote add origin https://github.com/TWOJ_USER/NAZWA_REPO.git
git push -u origin main
```

Z CLI GitHub (`gh`):

```powershell
gh repo create NAZWA_REPO --public --source=. --remote=origin --push
```

### Po sklonowaniu na innym komputerze

```powershell
git clone https://github.com/TWOJ_USER/NAZWA_REPO.git
cd NAZWA_REPO
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Wyniki w `html/`, `charts/` itd. są w repozytorium jako snapshot bieżącego stanu. Po zmianie plików w `input_csv/` uruchom ponownie pipeline (sekcja wyżej).
