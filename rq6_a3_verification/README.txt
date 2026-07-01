Tabele do recznej weryfikacji RQ6-A3
==================================

Pliki:
  verification_candidates.csv  - zacznij tutaj (priorytet > 0)
  verification_all_articles.csv - wszystkie artykuly
  verification_configurations.csv - wiersze z best data recon models

Kolumny do uzupelnienia recznie (verification_*):
  manual_verified            - tak / nie / czesciowo
  manual_paired_recon_task   - tak jesli w artykule ta sama konfiguracja
                               ma wynik recon i wynik zadania
  manual_has_numeric_table   - tak jesli sa liczby (nie tylko nazwy algorytmow)
  manual_pipeline_documented - tak jesli opisany jest lancuch imputacja->task
  manual_notes               - dowolne uwagi

Kolumny obliczone:
  candidate_priority - wyzszy = wazniejszy do weryfikacji
  configs_with_both_winners_count - ile wierszy ma zwyciezcow recon i task
  missing_rate_band - przedzial co 10%: 0-10%, ..., 90-100%
  missing_rate_unknown_reason - gdy band=nieznany:
      brak_wartosci | nienumeryczna_wartosc | ujemna_wartosc | poza_zakresem

Zrodla: input_csv/SLR(TAGGING-*.csv)
