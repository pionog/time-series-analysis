"""
Generuje interaktywną stronę HTML z drzewem decyzyjnym i grafem reguł RQ6-A4.

Wejście:  rules_results/*.csv
Wyjście:  html/reguly_graf.html
          edytowalne/reguly_eksperyment.csv
"""
from __future__ import annotations

import csv
import html
import json
import os
import re
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(SCRIPT_DIR, "rules_results")
HTML_DIR = os.path.join(SCRIPT_DIR, "html")
EDITABLE_DIR = os.path.join(SCRIPT_DIR, "edytowalne")
OUTPUT_HTML = os.path.join(HTML_DIR, "reguly_graf.html")
EXPERIMENT_CSV = os.path.join(EDITABLE_DIR, "reguly_eksperyment.csv")

BAND_ORDER = [
    "0-10%",
    "10-20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
    "nieznany",
]

CONTEXT_LABELS = {
    "dataset": "Zbiór danych",
    "dataset_domain": "Domena",
    "dataset_features": "Cechy",
    "task_model": "Model zadania",
    "order": "Kolejność",
    "timing": "Timing",
    "prediction_horizon": "Horyzont",
    "batch_size": "Batch size",
    "missing_rate_band": "Pasmo missing rate",
}


def load_csv(name: str) -> list[dict]:
    path = os.path.join(RULES_DIR, name)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def band_from_condition(condition: str) -> str:
    if "missing_rate_band = " in condition:
        return condition.split("missing_rate_band = ", 1)[1].split(" AND ", 1)[0].strip()
    m = re.search(r"missing_rate ∈ \[([^\]]+)\]", condition)
    return m.group(1) if m else "nieznany"


def fmt_pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def rule_tooltip(rule: dict, rule_type: str) -> str:
    support = rule.get("support") or rule.get("article_votes") or "?"
    total = (
        rule.get("total_configurations")
        or rule.get("total_configurations_in_band")
        or rule.get("total_observations")
        or rule.get("total_articles_in_band")
        or "?"
    )
    conf = rule.get("confidence", "")
    conf_s = fmt_pct(float(conf)) if conf else "—"
    ua = rule.get("unique_articles", "—")
    articles = rule.get("article_ids", "")
    warn = (
        '<br><span style="color:#b45309">Uwaga: tylko 1 artykuł w SLR</span>'
        if str(ua) == "1"
        else ""
    )
    return (
        f"<b>{html.escape(rule_type)}</b><br>"
        f"{html.escape(rule.get('rule_text', rule.get('condition', '')))}"
        f"<br>support: {html.escape(str(support))}/{html.escape(str(total))}"
        f" · confidence: {conf_s}"
        f" · artykuły: {html.escape(str(ua))}"
        f"{('<br>ID artykułów: ' + html.escape(articles)) if articles else ''}"
        f"{warn}"
    )


def normalize_rules() -> list[dict]:
    """Jednolita lista reguł do wizualizacji."""
    items: list[dict] = []

    for rule in load_csv("rules_rq6_a4.csv"):
        band = band_from_condition(rule["condition"])
        items.append(
            {
                "rule_id": rule["rule_id"],
                "rule_type": "pasmo",
                "band": band,
                "context_field": None,
                "context_dimension": None,
                "context_value": None,
                "metric": None,
                "condition_label": f"missing rate ∈ [{band}]",
                "algorithm": rule["recommended_algorithm"],
                "support": int(rule["support"]),
                "total": int(rule["total_configurations_in_band"]),
                "confidence": float(rule["confidence"]),
                "unique_articles": int(rule.get("unique_articles") or 0),
                "article_ids": rule.get("article_ids", ""),
                "rule_text": rule["rule_text"],
                "source": "rules_rq6_a4.csv",
            }
        )

    for rule in load_csv("rules_rq6_a4_by_context_and_band.csv"):
        band = band_from_condition(rule["condition"])
        field = rule["context_field"]
        items.append(
            {
                "rule_id": rule["rule_id"],
                "rule_type": "kontekst×pasmo",
                "band": band,
                "context_field": field,
                "context_dimension": rule["context_dimension"],
                "context_value": rule["context_value"],
                "metric": None,
                "condition_label": (
                    f"{CONTEXT_LABELS.get(field, field)} = {rule['context_value']} "
                    f"· {band}"
                ),
                "algorithm": rule["recommended_algorithm"],
                "support": int(rule["support"]),
                "total": int(rule["total_configurations"]),
                "confidence": float(rule["confidence"]),
                "unique_articles": int(rule.get("unique_articles") or 0),
                "article_ids": rule.get("article_ids", ""),
                "rule_text": rule["rule_text"],
                "source": "rules_rq6_a4_by_context_and_band.csv",
            }
        )

    for rule in load_csv("rules_rq6_a4_by_metric.csv"):
        band = band_from_condition(rule["condition"])
        items.append(
            {
                "rule_id": rule["rule_id"],
                "rule_type": "pasmo×metryka",
                "band": band,
                "context_field": None,
                "context_dimension": None,
                "context_value": None,
                "metric": rule["metric"],
                "condition_label": f"missing rate ∈ [{band}] · metryka {rule['metric']}",
                "algorithm": rule["recommended_algorithm"],
                "support": int(rule["support"]),
                "total": int(rule["total_observations"]),
                "confidence": float(rule["confidence"]),
                "unique_articles": int(rule.get("unique_articles") or 0),
                "article_ids": rule.get("article_ids", ""),
                "rule_text": rule["rule_text"],
                "source": "rules_rq6_a4_by_metric.csv",
            }
        )

    return items


def pick_experiment_rules(rules: list[dict], limit: int = 8) -> list[dict]:
    """Wybiera zróżnicowany zestaw reguł pod przyszły eksperyment."""
    priority_ids = [
        "A4-ctx-task_model-LSSVM-band-20-30pct",
        "A4-ctx-dataset_domain-Climate-band-0-10pct",
        "A4-ctx-task_model-FIR-band-0-10pct",
        "A4-ctx-task_model-Transformer-band-0-10pct",
        "A4-20-30pct-MAE",
        "A4-30-40pct-SMAPE",
        "A4-90-100pct-RMSE",
        "A4-ctx-task_model-ASTGCN-band-0-10pct",
    ]
    by_id = {r["rule_id"]: r for r in rules}
    picked: list[dict] = []

    for rid in priority_ids:
        if rid in by_id and rid not in {p["rule_id"] for p in picked}:
            picked.append(by_id[rid])

    if len(picked) < limit:
        ctx_rules = [r for r in rules if r["rule_type"] == "kontekst×pasmo"]
        ctx_rules.sort(key=lambda r: (-r["support"], -r["confidence"]))
        seen_algo: set[str] = set()
        for r in ctx_rules:
            if r["rule_id"] in {p["rule_id"] for p in picked}:
                continue
            if r["algorithm"] in seen_algo:
                continue
            picked.append(r)
            seen_algo.add(r["algorithm"])
            if len(picked) >= limit:
                break

    return picked[:limit]


def export_experiment_csv(rules: list[dict]) -> None:
    os.makedirs(EDITABLE_DIR, exist_ok=True)
    fields = [
        "priorytet",
        "rule_id",
        "rule_type",
        "condition_label",
        "recommended_algorithm",
        "support",
        "total",
        "confidence",
        "unique_articles",
        "article_ids",
        "wybrac_do_eksperymentu",
        "notatki",
    ]
    with open(EXPERIMENT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, rule in enumerate(rules, 1):
            writer.writerow(
                {
                    "priorytet": i,
                    "rule_id": rule["rule_id"],
                    "rule_type": rule["rule_type"],
                    "condition_label": rule["condition_label"],
                    "recommended_algorithm": rule["algorithm"],
                    "support": rule["support"],
                    "total": rule["total"],
                    "confidence": f"{rule['confidence']:.4f}",
                    "unique_articles": rule["unique_articles"],
                    "article_ids": rule["article_ids"],
                    "wybrac_do_eksperymentu": "TAK" if i <= 5 else "ROZWAZYC",
                    "notatki": (
                        "Jeden artykuł w SLR — wymaga walidacji eksperymentalnej"
                        if rule["unique_articles"] <= 1
                        else ""
                    ),
                }
            )


def build_graph_payload(rules: list[dict]) -> dict:
    """Buduje węzły i krawędzie dla widoku sieci (warunek → algorytm)."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_id = 0

    def ensure_node(key: str, label: str, group: str, value: int, title: str) -> str:
        if key not in nodes:
            nodes[key] = {
                "id": key,
                "label": label,
                "group": group,
                "value": max(value, 1),
                "title": title,
            }
        else:
            nodes[key]["value"] = max(nodes[key]["value"], value)
        return key

    for rule in rules:
        cond_key = f"cond::{rule['rule_id']}"
        algo_key = f"algo::{rule['algorithm']}"
        ensure_node(
            cond_key,
            rule["condition_label"],
            rule["rule_type"],
            rule["support"],
            rule_tooltip(
                {
                    "rule_text": rule["rule_text"],
                    "support": rule["support"],
                    "total_configurations": rule["total"],
                    "confidence": rule["confidence"],
                    "unique_articles": rule["unique_articles"],
                    "article_ids": rule["article_ids"],
                    "condition": rule["condition_label"],
                },
                rule["rule_type"],
            ),
        )
        ensure_node(
            algo_key,
            rule["algorithm"],
            "algorytm",
            rule["support"],
            f"<b>Algorytm</b><br>{html.escape(rule['algorithm'])}",
        )
        edge_id += 1
        edges.append(
            {
                "id": edge_id,
                "from": cond_key,
                "to": algo_key,
                "value": rule["support"],
                "title": (
                    f"support={rule['support']}/{rule['total']} "
                    f"({fmt_pct(rule['confidence'])})"
                ),
                "rule_id": rule["rule_id"],
                "rule_type": rule["rule_type"],
                "band": rule["band"],
                "support": rule["support"],
            }
        )

    return {"nodes": list(nodes.values()), "edges": edges}


def build_tree_payload(rules: list[dict]) -> dict:
    """
    Drzewo: korzeń → pasmo missing rate → kontekst → wartość → algorytm.
    Reguły ogólne (pasmo) i kontekst×pasmo w jednej hierarchii.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_id = 0

    def add_node(key: str, label: str, group: str, level: int, title: str, size: int = 8):
        if key not in nodes:
            nodes[key] = {
                "id": key,
                "label": label,
                "group": group,
                "level": level,
                "value": size,
                "title": title,
            }

    def add_edge(src: str, dst: str, label: str = ""):
        nonlocal edge_id
        edge_id += 1
        edges.append({"id": edge_id, "from": src, "to": dst, "label": label})

    root = "root"
    add_node(root, "Wybór modelu rekonstrukcji", "korzen", 0, "<b>Korzeń drzewa decyzyjnego</b>")

    band_rules = [r for r in rules if r["rule_type"] == "pasmo"]
    ctx_rules = [r for r in rules if r["rule_type"] == "kontekst×pasmo"]

    for band in BAND_ORDER:
        band_key = f"band::{band}"
        band_items = [r for r in band_rules if r["band"] == band]
        ctx_items = [r for r in ctx_rules if r["band"] == band]
        if not band_items and not ctx_items:
            continue

        band_support = max((r["support"] for r in band_items), default=0)
        add_node(
            band_key,
            band,
            "pasmo",
            1,
            f"<b>Pasmo missing rate</b><br>{html.escape(band)}",
            max(band_support, 4),
        )
        add_edge(root, band_key)

        for rule in band_items:
            algo_key = f"tree_algo::{rule['rule_id']}"
            add_node(
                algo_key,
                f"{rule['algorithm']}\n({rule['support']}/{rule['total']})",
                "rekomendacja_ogolna",
                2,
                rule_tooltip(
                    {
                        "rule_text": rule["rule_text"],
                        "support": rule["support"],
                        "total_configurations_in_band": rule["total"],
                        "confidence": rule["confidence"],
                        "unique_articles": rule["unique_articles"],
                        "article_ids": rule["article_ids"],
                    },
                    "pasmo (ogólne)",
                ),
                rule["support"],
            )
            add_edge(band_key, algo_key, "ogólne")

        ctx_by_field: dict[str, list[dict]] = {}
        for rule in ctx_items:
            field = rule["context_field"] or "inne"
            ctx_by_field.setdefault(field, []).append(rule)

        for field, field_rules in sorted(ctx_by_field.items()):
            field_label = CONTEXT_LABELS.get(field, field)
            field_key = f"field::{band}::{field}"
            add_node(
                field_key,
                field_label,
                "kontekst",
                2,
                f"<b>{html.escape(field_label)}</b>",
                6,
            )
            add_edge(band_key, field_key)

            field_rules.sort(key=lambda r: (-r["support"], -r["confidence"]))
            for rule in field_rules:
                val_key = f"val::{rule['rule_id']}"
                algo_key = f"tree_algo_ctx::{rule['rule_id']}"
                add_node(
                    val_key,
                    rule["context_value"] or "?",
                    "wartosc",
                    3,
                    html.escape(rule["condition_label"]),
                    rule["support"],
                )
                add_edge(field_key, val_key)
                add_node(
                    algo_key,
                    f"{rule['algorithm']}\n({rule['support']}/{rule['total']})",
                    "rekomendacja",
                    4,
                    rule_tooltip(
                        {
                            "rule_text": rule["rule_text"],
                            "support": rule["support"],
                            "total_configurations": rule["total"],
                            "confidence": rule["confidence"],
                            "unique_articles": rule["unique_articles"],
                            "article_ids": rule["article_ids"],
                        },
                        "kontekst × pasmo",
                    ),
                    rule["support"],
                )
                add_edge(val_key, algo_key, fmt_pct(rule["confidence"]))

    return {"nodes": list(nodes.values()), "edges": edges}


def render_experiment_table(rules: list[dict]) -> str:
    rows = []
    for i, rule in enumerate(rules, 1):
        warn = (
            ' <span class="sample-warn">(1 artykuł SLR)</span>'
            if rule["unique_articles"] <= 1
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(rule['rule_id'])}</code></td>"
            f"<td>{html.escape(rule['rule_type'])}</td>"
            f"<td>{html.escape(rule['condition_label'])}</td>"
            f"<td><strong>{html.escape(rule['algorithm'])}</strong></td>"
            f"<td>{rule['support']}/{rule['total']}</td>"
            f"<td>{fmt_pct(rule['confidence'])}{warn}</td>"
            "</tr>"
        )
    return (
        '<table class="data-table"><thead><tr>'
        "<th>#</th><th>ID</th><th>Typ</th><th>Warunek</th>"
        "<th>Algorytm</th><th>Support</th><th>Confidence</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_html(
    all_rules: list[dict],
    experiment_rules: list[dict],
    tree_data: dict,
    graph_data: dict,
) -> str:
    today = date.today().isoformat()
    counts = {
        "all": len(all_rules),
        "pasmo": sum(1 for r in all_rules if r["rule_type"] == "pasmo"),
        "kontekst": sum(1 for r in all_rules if r["rule_type"] == "kontekst×pasmo"),
        "metryka": sum(1 for r in all_rules if r["rule_type"] == "pasmo×metryka"),
    }
    tree_json = json.dumps(tree_data, ensure_ascii=False)
    graph_json = json.dumps(graph_data, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reguły RQ6 — drzewo i graf</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    :root {{
      --bg: #f4f6f8; --card: #fff; --border: #c5d0dc; --text: #1a2a3a;
      --muted: #5a6a7a; --accent: #2e6b8a; --th-bg: #e8eef3; --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 2rem 1rem; font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5; }}
    .page {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; margin: 0 0 0.5rem; color: var(--accent); }}
    h2 {{ font-size: 1.2rem; margin: 2rem 0 0.75rem; color: var(--accent); }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .meta a {{ color: var(--accent); }}
    .card {{ background: var(--card); border: 1px solid var(--border);
      box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 1rem 1.25rem; margin-bottom: 1.5rem; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0; }}
    .stat {{ background: var(--th-bg); border: 1px solid var(--border); border-radius: 6px;
      padding: 0.5rem 0.85rem; font-size: 0.9rem; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: 0.75rem 0; }}
    .controls label {{ font-size: 0.9rem; }}
    .controls select, .controls input[type=range] {{ margin-left: 0.35rem; }}
    .tabs {{ display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }}
    .tab {{ border: 1px solid var(--border); background: var(--th-bg); color: var(--text);
      padding: 0.45rem 0.9rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }}
    .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    #graph-network, #tree-network {{ width: 100%; height: 620px; border: 1px solid var(--border);
      background: #fbfcfd; border-radius: 4px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 0.65rem; margin-top: 0.75rem; font-size: 0.82rem; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
    .swatch {{ width: 0.85rem; height: 0.85rem; border-radius: 3px; border: 1px solid var(--border); }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    .data-table th, .data-table td {{ border: 1px solid var(--border); padding: 0.5rem 0.65rem;
      text-align: left; vertical-align: top; }}
    .data-table th {{ background: var(--th-bg); }}
    .sample-warn {{ color: var(--warn); font-size: 0.82em; font-style: italic; }}
    .note {{ color: var(--muted); font-size: 0.9rem; }}
    code {{ font-size: 0.85em; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Reguły wyboru modelu rekonstrukcji (RQ6-A4)</h1>
    <p class="meta">
      Wygenerowano: {today} · Źródło: rules_results/
      · <a href="wyniki_badawcze.html">← Wyniki pytań badawczych</a>
    </p>

    <div class="card">
      <p>Interaktywna wizualizacja reguł wyciągniętych z literatury (SLR).
         <strong>Drzewo decyzyjne</strong> pokazuje hierarchię: pasmo missing rate → kontekst → rekomendacja.
         <strong>Graf sieciowy</strong> łączy warunki z algorytmami (grubość krawędzi = support).</p>
      <div class="stats">
        <div class="stat"><strong>{counts["all"]}</strong> reguł łącznie</div>
        <div class="stat">{counts["pasmo"]} ogólnych (pasmo)</div>
        <div class="stat">{counts["kontekst"]} kontekst × pasmo</div>
        <div class="stat">{counts["metryka"]} pasmo × metryka</div>
      </div>
      <p class="note">Większość reguł opiera się na jednym artykule SLR (wiele konfiguracji).
         Kandydaci do eksperymentu są w tabeli poniżej i w pliku
         <code>edytowalne/reguly_eksperyment.csv</code>.</p>
    </div>

    <h2>Wizualizacja</h2>
    <div class="card">
      <div class="tabs">
        <button class="tab active" id="tab-tree" type="button">Drzewo decyzyjne</button>
        <button class="tab" id="tab-graph" type="button">Graf sieciowy</button>
      </div>
      <div class="controls" id="graph-controls" style="display:none">
        <label>Min. support: <span id="support-val">3</span>
          <input type="range" id="support-min" min="2" max="12" value="3">
        </label>
        <label>Typ reguły:
          <select id="rule-type">
            <option value="all">Wszystkie</option>
            <option value="pasmo">Pasmo</option>
            <option value="kontekst×pasmo">Kontekst × pasmo</option>
            <option value="pasmo×metryka">Pasmo × metryka</option>
          </select>
        </label>
      </div>
      <div id="tree-network"></div>
      <div id="graph-network" style="display:none"></div>
      <div class="legend">
        <span><i class="swatch" style="background:#2e6b8a"></i> pasmo</span>
        <span><i class="swatch" style="background:#5b8a72"></i> kontekst</span>
        <span><i class="swatch" style="background:#7a6b8a"></i> wartość</span>
        <span><i class="swatch" style="background:#b45309"></i> rekomendacja</span>
        <span><i class="swatch" style="background:#4a6fa5"></i> algorytm (graf)</span>
      </div>
    </div>

    <h2>Kandydaci do eksperymentu (top {len(experiment_rules)})</h2>
    <div class="card">
      {render_experiment_table(experiment_rules)}
    </div>
  </div>

  <script>
    const treeData = {tree_json};
    const graphData = {graph_json};

    const groupColors = {{
      korzen: "#1a2a3a",
      pasmo: "#2e6b8a",
      kontekst: "#5b8a72",
      wartosc: "#7a6b8a",
      rekomendacja: "#b45309",
      rekomendacja_ogolna: "#c77d38",
      algorytm: "#4a6fa5",
      "pasmo": "#2e6b8a",
      "kontekst×pasmo": "#5b8a72",
      "pasmo×metryka": "#6a7c9a",
    }};

    function colorNodes(nodes) {{
      return nodes.map(n => ({{
        ...n,
        color: {{ background: groupColors[n.group] || "#888", border: "#334" }},
        font: {{ color: n.group === "korzen" ? "#fff" : "#1a2a3a", size: 13 }},
        shape: n.group === "algorytm" ? "box" : "dot",
      }}));
    }}

    const treeOptions = {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 120,
          nodeSpacing: 140,
        }},
      }},
      physics: false,
      interaction: {{ hover: true, tooltipDelay: 120 }},
      edges: {{ arrows: {{ to: {{ enabled: true, scaleFactor: 0.55 }} }}, font: {{ size: 10, align: "middle" }} }},
      nodes: {{ borderWidth: 1, size: 16 }},
    }};

    const graphOptions = {{
      layout: {{ improvedLayout: true }},
      physics: {{
        enabled: true,
        barnesHut: {{ gravitationalConstant: -22000, springLength: 180, springConstant: 0.04 }},
        stabilization: {{ iterations: 180 }},
      }},
      interaction: {{ hover: true, tooltipDelay: 120 }},
      edges: {{
        smooth: {{ type: "dynamic" }},
        scaling: {{ min: 1, max: 10 }},
      }},
      nodes: {{ borderWidth: 1, font: {{ size: 12 }} }},
    }};

    const treeNet = new vis.Network(
      document.getElementById("tree-network"),
      {{ nodes: new vis.DataSet(colorNodes(treeData.nodes)), edges: new vis.DataSet(treeData.edges) }},
      treeOptions
    );

    let graphNet = null;
    function renderGraph() {{
      const minSupport = Number(document.getElementById("support-min").value);
      const type = document.getElementById("rule-type").value;
      document.getElementById("support-val").textContent = minSupport;
      const edges = graphData.edges.filter(e =>
        e.support >= minSupport && (type === "all" || e.rule_type === type)
      );
      const nodeIds = new Set();
      edges.forEach(e => {{ nodeIds.add(e.from); nodeIds.add(e.to); }});
      const nodes = graphData.nodes.filter(n => nodeIds.has(n.id));
      const container = document.getElementById("graph-network");
      if (!graphNet) {{
        graphNet = new vis.Network(
          container,
          {{ nodes: new vis.DataSet(colorNodes(nodes)), edges: new vis.DataSet(edges) }},
          graphOptions
        );
      }} else {{
        graphNet.setData({{ nodes: new vis.DataSet(colorNodes(nodes)), edges: new vis.DataSet(edges) }});
      }}
    }}

    document.getElementById("tab-tree").addEventListener("click", () => {{
      document.getElementById("tab-tree").classList.add("active");
      document.getElementById("tab-graph").classList.remove("active");
      document.getElementById("tree-network").style.display = "block";
      document.getElementById("graph-network").style.display = "none";
      document.getElementById("graph-controls").style.display = "none";
    }});
    document.getElementById("tab-graph").addEventListener("click", () => {{
      document.getElementById("tab-graph").classList.add("active");
      document.getElementById("tab-tree").classList.remove("active");
      document.getElementById("tree-network").style.display = "none";
      document.getElementById("graph-network").style.display = "block";
      document.getElementById("graph-controls").style.display = "flex";
      renderGraph();
      if (graphNet) graphNet.fit();
    }});
    document.getElementById("support-min").addEventListener("input", renderGraph);
    document.getElementById("rule-type").addEventListener("change", renderGraph);
  </script>
</body>
</html>
"""


def main():
    os.makedirs(HTML_DIR, exist_ok=True)
    all_rules = normalize_rules()
    if not all_rules:
        raise SystemExit("Brak reguł w rules_results/. Uruchom najpierw extract_rules_from_best_data_recon.py")

    experiment = pick_experiment_rules(all_rules)
    export_experiment_csv(experiment)

    tree_data = build_tree_payload(all_rules)
    graph_data = build_graph_payload(all_rules)

    content = render_html(all_rules, experiment, tree_data, graph_data)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Zapisano: {OUTPUT_HTML}")
    print(f"Zapisano: {EXPERIMENT_CSV}")
    print(
        f"Reguły: {len(all_rules)} · drzewo: {len(tree_data['nodes'])} węzłów · "
        f"graf: {len(graph_data['nodes'])} węzłów · eksperyment: {len(experiment)}"
    )


if __name__ == "__main__":
    main()
