"""Eksportuje listę pytań badawczych i analitycznych ze statusem odpowiedzialności."""
import os
import re
from datetime import date

from build_research_html import parse_questions_csv, resolve_answer
from research_answers import build_answers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(SCRIPT_DIR, "html", "status_pytan.txt")


def strip_html(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<br>\n", "\n")
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def main():
    answers = build_answers()
    blocks = parse_questions_csv()

    lines = [
        "STATUS PYTAŃ BADAWCZYCH I ANALITYCZNYCH",
        f"Data: {date.today().isoformat()}",
        "",
        "Legenda:",
        "  [TAK]       — na pytanie można już odpowiedzieć na podstawie danych w projekcie",
        "  [NIE]       — brak danych; wymaga ręcznej pracy lub nowych źródeł",
        "  [CZĘŚCIOWO] — dostępne są dane pomocnicze, ale odpowiedź nie jest pełna",
        "",
        "Źródła danych: input_csv/, csv_results/, rules_results/",
        "",
        "=" * 72,
        "",
    ]

    total_aq = 0
    answerable = 0
    partial = 0
    not_answerable = 0

    for block in blocks:
        rq_id = block["rq_id"]
        rq_text = block["rq_text"]
        rq_answer = strip_html(answers.get(rq_id, ""))

        lines.append(f"{rq_id}: {rq_text}")
        lines.append("-" * 72)
        if rq_answer:
            lines.append(f"  Podsumowanie RQ [{rq_id}]: [TAK]")
            lines.append(f"  {rq_answer.replace(chr(10), chr(10) + '  ')}")
        else:
            lines.append(f"  Podsumowanie RQ [{rq_id}]: [NIE]")
        lines.append("")

        for aq in block["analytical"]:
            total_aq += 1
            aq_id = aq.get("id") or "(bez ID)"
            aq_text = aq["text"]

            if aq.get("manual_only"):
                status = "NIE"
                note = "oznaczone jako do uzupełnienia w pliku CSV"
                not_answerable += 1
            else:
                raw = resolve_answer(aq.get("id"), aq_text, answers)
                content = strip_html(raw)
                if content:
                    status = "TAK"
                    answerable += 1
                    note = "dane gotowe — patrz html/wyniki_badawcze.html"
                elif aq.get("details_hint"):
                    status = "CZĘŚCIOWO"
                    partial += 1
                    note = f"w CSV jest szkic odpowiedzi (kolumna Szczegóły), ale brak automatycznej agregacji"
                else:
                    status = "NIE"
                    not_answerable += 1
                    note = "brak danych w projekcie"

            label = f"{aq_id}" if aq_id != "(bez ID)" else aq_text[:50]
            lines.append(f"  [{status}]  {label}")
            lines.append(f"           {aq_text}")
            if status == "TAK" and not aq.get("manual_only"):
                preview = content.split("\n")[0]
                if len(preview) > 100:
                    preview = preview[:97] + "..."
                lines.append(f"           → {preview}")
            else:
                lines.append(f"           → {note}")
            lines.append("")

        lines.append("=" * 72)
        lines.append("")

    lines.extend([
        "PODSUMOWANIE",
        "-" * 72,
        f"Pytania badawcze (RQ):           {len(blocks)}",
        f"Pytania analityczne łącznie:     {total_aq}",
        f"  [TAK]       — gotowe:          {answerable}",
        f"  [CZĘŚCIOWO] — wymaga doprac.:  {partial}",
        f"  [NIE]       — do zrobienia:    {not_answerable}",
        "",
        "Priorytet pracy: pytania [NIE] i [CZĘŚCIOWO].",
        "Pytania [TAK] są już w html/wyniki_badawcze.html — można je przejrzeć i sformułować narrację.",
    ])

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Zapisano: {OUTPUT}")
    print(f"TAK: {answerable}, CZĘŚCIOWO: {partial}, NIE: {not_answerable}")


if __name__ == "__main__":
    main()
