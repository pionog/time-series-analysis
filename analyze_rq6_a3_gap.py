"""Analiza luk danych dla RQ6-A3 na podstawie plików input_csv."""
import csv
import os

RECON_METRICS = {
    "RMSE", "MAPE", "SMAPE", "MAE", "NRMSE", "MSE", "MAD",
    "Log-likelihood", "Log-likelihood ratio", "CORR", "RSE", "MASE",
    "Dstat", "BIC", "PCC (COR)", "STD", "R2", "LMI", "TIC", "IA", "ND",
    "WMAPE", "MSD", "Maximum Absolute Error", "RSquare",
}
TASK_METRICS = {
    "AUC", "F1-Score", "Accuracy", "Precision", "Specifity",
    "Sensivity", "Recall", "AUPRC", "Model size", "Runtime", "Other",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_csv")


def parse_tagging(path: str) -> dict[str, set[str]]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers:
            return {}

        feature_cols = [
            h for h in headers
            if h and h not in {"ID", "Title", "Tagger", "Comment", "Comments"}
        ]
        result: dict[str, set[str]] = {}
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            tags = {
                fc for fc in feature_cols
                if row[col_idx[fc]].strip().lower() == "x"
            }
            result[aid] = tags
        return result


def parse_summary_winners(path: str) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = None
        col_idx = {}
        for row in reader:
            if row and row[0].strip() == "ID":
                headers = [h.strip() for h in row]
                col_idx = {h: i for i, h in enumerate(headers) if h}
                break
        if not headers:
            return {}

        metric_cols = [h for h in headers if h and h not in {"ID", "Title", "Tagger"}]
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            if not row:
                continue
            row = row + [""] * (len(headers) - len(row))
            if any("REJECTED" in (c or "").upper() for c in row):
                continue
            aid = row[col_idx["ID"]].strip()
            if not aid or not aid.isdigit():
                continue
            winners = {
                h: row[col_idx[h]].strip()
                for h in metric_cols
                if row[col_idx[h]].strip()
            }
            if winners:
                result[aid] = winners
        return result


def metric_kind(name: str) -> str:
    if name in RECON_METRICS:
        return "rekonstrukcja"
    if name in TASK_METRICS:
        return "zadanie"
    return "inne"


def has_recon_in_tags(tags: set[str]) -> bool:
    return any(
        any(x in m for x in ("RMSE", "MAE", "MSE", "MAPE", "NRMSE", "SMAPE", "MAD", "CORR", "RSE", "R2", "MSD"))
        for m in tags
    )


def has_task_in_tags(tags: set[str]) -> bool:
    return any(
        any(x in m for x in ("AUC", "F1", "Accuracy", "Precision", "Recall", "Specifity", "Sensivity"))
        for m in tags
    )


def main():
    task = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-task models).csv"))
    recon_models = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-models data reconstruct).csv"))
    metrics = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-metrics used).csv"))
    apps = parse_tagging(os.path.join(INPUT_DIR, "SLR(TAGGING-applications).csv"))
    summary = parse_summary_winners(
        os.path.join(INPUT_DIR, "SLR(TAGGING-best data recon SUMMARY).csv")
    )

    both_models = sorted(
        aid for aid in set(task) & set(recon_models)
        if task.get(aid) and recon_models.get(aid)
    )
    both_metrics = sorted(aid for aid, t in metrics.items() if has_recon_in_tags(t) and has_task_in_tags(t))
    summary_both = []
    for aid, winners in summary.items():
        recon_w = {k: v for k, v in winners.items() if k in RECON_METRICS}
        task_w = {k: v for k, v in winners.items() if k in TASK_METRICS}
        if recon_w and task_w:
            summary_both.append((aid, recon_w, task_w))

    print("=== RQ6-A3: dostepnosc danych w input_csv ===\n")
    print(f"Artykuly z modelami zadaniowymi (task models): {sum(1 for t in task.values() if t)}")
    print(f"Artykuly z modelami rekonstrukcji: {sum(1 for t in recon_models.values() if t)}")
    print(f"Artykuly uzywajace OBOMA (task + recon): {len(both_models)}")
    print(f"Artykuly raportujace metryki recon+task (metrics used): {len(both_metrics)}")
    if both_metrics:
        print(f"  ID: {', '.join(both_metrics)}")
    print(f"\nSUMMARY - zwyciezca w kolumnach recon ORAZ task: {len(summary_both)}")
    for aid, rw, tw in summary_both:
        print(f"  [{aid}] recon={rw} | task={tw}")

    print("\n=== Czego BRAKUJE do pelnej odpowiedzi RQ6-A3 ===")
    gaps = [
        "1. Powiazanie per eksperyment: ten sam (dataset, missing_rate, task_model, metoda imputacji) -> wynik recon + wynik task",
        "2. Wartosci liczbowe metryk (w CSV sa glownie nazwy zwyciezcow / tagi x, nie RMSE=0.12)",
        "3. Jawny pipeline: imputacja metoda X -> model zadaniowy Y -> metryka Z (kolejnosc krokow)",
        "4. Porownanie wielu metod rekonstrukcji na TEJ SAMEJ metryce zadania (tabela w artykule)",
    ]
    for g in gaps:
        print(f"  - {g}")

    print("\n=== Co MOZNA wykorzystac posrednio ===")
    uses = [
        f"task models + models data reconstruct ({len(both_models)} artykulow): potwierdza ze praca ma oba etapy, ale nie korelacje jakosci",
        f"metrics used ({len(both_metrics)} artykulow): potwierdza raportowanie obu typow metryk",
        f"best data recon SUMMARY ({len(summary_both)} artykulow): najlepszy algorytm per kolumna metryki na poziomie artykulu (bez warunkow eksperymentu)",
        "best data recon models: szczegoly per missing_rate/dataset, ale prawie wylacznie metryki rekonstrukcji",
        "rules_rq6_a3_metric_consistency: proxy - spojnosc wielu metryk imputacji w jednej konfiguracji",
    ]
    for u in uses:
        print(f"  + {u}")


if __name__ == "__main__":
    main()
