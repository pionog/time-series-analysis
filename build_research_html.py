"""Generuje stronę HTML z odpowiedziami na pytania badawcze na podstawie danych SLR."""
import csv
import html
import os
from datetime import date

from research_answers import build_answers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(SCRIPT_DIR, "html")
QUESTIONS_CSV = os.path.join(HTML_DIR, "Pytania badawcze(Pytania badawcze).csv")
OUTPUT_HTML = os.path.join(HTML_DIR, "wyniki_badawcze.html")


def row_is_koniec(row: list[str]) -> bool:
    return any(cell.strip().upper() == "KONIEC" for cell in row if cell and cell.strip())


def parse_questions_csv() -> list[dict]:
    blocks: list[dict] = []
    current: dict | None = None

    with open(QUESTIONS_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader, None)
        for row in reader:
            row = row + [""] * (10 - len(row))
            if row_is_koniec(row):
                break

            rq_num = row[0].strip()
            rq_text = row[1].strip()
            aq_num = row[2].strip()
            aq_text = row[3].strip()
            details = row[4].strip()

            if rq_num and rq_text:
                current = {"rq_id": rq_num, "rq_text": rq_text, "analytical": []}
                blocks.append(current)

            if current is None or (not aq_text and not aq_num):
                continue
            if "eksperyment" in (aq_text + aq_num).lower():
                continue
            if not aq_num and (
                aq_text.startswith("....")
                or aq_text.strip() == "..."
            ):
                continue
            if aq_num in ("...",) or aq_text.lower().startswith("jakie kategorie... <do uzupe"):
                current["analytical"].append({
                    "id": aq_num or "...",
                    "text": aq_text or "Jakie kategorie... (do uzupełnienia)",
                    "manual_only": True,
                })
                continue

            current["analytical"].append({
                "id": aq_num or None,
                "text": aq_text,
                "details_hint": details,
                "manual_only": False,
            })

    return blocks


def resolve_answer(qid: str | None, aq_text: str, answers: dict[str, str]) -> str:
    if not qid:
        return ""

    content = answers.get(qid, "")
    if qid == "RQ3-A8" and answers.get("RQ3-A8-detail"):
        content = content + answers["RQ3-A8-detail"]
    return content


def render_answer_cell(content: str, empty_label: str = "— do uzupełnienia ręcznie —") -> str:
    if not content or not content.strip():
        return f'<p class="empty">{html.escape(empty_label)}</p>'
    return f"""<details open>
  <summary>Pokaż / ukryj odpowiedź</summary>
  <div class="answer-body">{content}</div>
</details>"""


def render_html(blocks: list[dict], answers: dict[str, str]) -> str:
    sections = []
    for block in blocks:
        rq_id = block["rq_id"]
        analytical = block["analytical"]

        rq_text = html.escape(block["rq_text"])
        rq_answer = answers.get(rq_id, "")
        rows = [
            f'<tr><th colspan="1">Pytanie badawcze ({html.escape(rq_id)})</th></tr>',
            f"<tr><td><h2>{rq_text}</h2></td></tr>",
            "<tr><th>Odpowiedź na pytanie badawcze</th></tr>",
            f"<tr><td>{render_answer_cell(rq_answer)}</td></tr>",
        ]
        for aq in analytical:
            aq_id = aq.get("id") or ""
            aq_text = html.escape(aq["text"])
            label = f"{html.escape(aq_id)}: {aq_text}" if aq_id and aq_id != "..." else aq_text
            content = "" if aq.get("manual_only") else resolve_answer(aq_id or None, aq["text"], answers)
            rows.append(f"<tr><th>{label}</th></tr>")
            rows.append(f"<tr><td>{render_answer_cell(content)}</td></tr>")

        sections.append(
            f'<section class="rq-block" id="{html.escape(rq_id)}">\n'
            f'  <table class="rq-table">\n    ' + "\n    ".join(rows) + "\n  </table>\n</section>"
        )

    body = "\n".join(sections)
    today = date.today().isoformat()
    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wyniki pytań badawczych — SLR</title>
  <style>
    :root {{
      --bg: #f4f6f8; --card: #fff; --border: #c5d0dc; --text: #1a2a3a;
      --muted: #5a6a7a; --accent: #2e6b8a; --th-bg: #e8eef3; --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 2rem 1rem; font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5; }}
    .page {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.5rem; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
    .rq-block {{ margin-bottom: 2.5rem; }}
    .rq-table {{ width: 100%; border-collapse: collapse; background: var(--card);
      box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .rq-table th, .rq-table td {{ border: 1px solid var(--border); padding: 0.85rem 1rem;
      text-align: left; vertical-align: top; }}
    .rq-table th {{ background: var(--th-bg); font-weight: 600; font-size: 0.95rem; }}
    .rq-table td h2 {{ margin: 0; font-size: 1.15rem; }}
    details {{ margin: 0; }}
    summary {{ cursor: pointer; color: var(--accent); font-size: 0.85rem; margin-bottom: 0.5rem; }}
    .answer-body {{ font-size: 0.95rem; overflow-x: auto; }}
    .empty {{ color: var(--muted); font-style: italic; margin: 0; }}
    .sample-warn {{ color: var(--warn); font-size: 0.82em; font-style: italic; }}
    .bullet-list {{ margin: 0.4rem 0 0.8rem 1.2rem; padding: 0; }}
    .sub-answer {{ margin-top: 0.75rem; padding: 0.5rem 0.75rem;
      border-left: 3px solid var(--accent); background: #f8fbfd; }}
    .data-matrix {{ border-collapse: collapse; margin: 0.75rem 0; font-size: 0.88rem; }}
    .data-matrix th, .data-matrix td {{ border: 1px solid var(--border); padding: 0.4rem 0.55rem;
      text-align: center; min-width: 2.5rem; }}
    .data-matrix th {{ background: var(--th-bg); }}
    .data-matrix caption {{ caption-side: top; text-align: left; font-size: 0.85rem;
      color: var(--muted); margin-bottom: 0.35rem; }}
    .heat-legend {{ display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0 1rem;
      font-size: 0.82rem; color: var(--muted); flex-wrap: wrap; }}
    .heat-legend-bar {{ display: flex; border: 1px solid var(--border); border-radius: 3px;
      overflow: hidden; }}
    .heat-legend-swatch {{ display: block; width: 2rem; height: 0.85rem; }}
    .heat-legend-label {{ white-space: nowrap; }}
    nav.toc {{ background: var(--card); border: 1px solid var(--border);
      padding: 1rem 1.25rem; margin-bottom: 2rem; }}
    nav.toc ul {{ margin: 0.5rem 0 0; padding-left: 1.25rem; }}
    nav.toc a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Wyniki pytań badawcze</h1>
    <p class="meta">Wygenerowano: {today} · Źródła: input_csv, csv_results, rules_results</p>
    <nav class="toc">
      <strong>Spis pytań badawczych</strong>
      <ul>
        {''.join(f'<li><a href="#{html.escape(b["rq_id"])}">{html.escape(b["rq_id"])} — {html.escape(b["rq_text"][:80])}{"…" if len(b["rq_text"]) > 80 else ""}</a></li>' for b in blocks)}
      </ul>
    </nav>
    {body}
  </div>
</body>
</html>
"""


def main():
    os.makedirs(HTML_DIR, exist_ok=True)
    answers = build_answers()
    blocks = parse_questions_csv()
    content = render_html(blocks, answers)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Zapisano: {OUTPUT_HTML}")
    filled = sum(
        1 for b in blocks for a in b["analytical"]
        if not a.get("manual_only") and resolve_answer(a.get("id"), a["text"], answers).strip()
    )
    print(f"Odpowiedzi analityczne z danymi: {filled}")


if __name__ == "__main__":
    main()
