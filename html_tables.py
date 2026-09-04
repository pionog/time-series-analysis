"""Generatory tabel i heatmap do strony HTML."""
import html
import math


def sample_note(count: int, total: int | None = None, threshold: int = 3) -> str:
    if count == 0:
        return ""
    parts = []
    if count < threshold:
        parts.append("znikoma próbka")
    if total and count / total < 0.02:
        parts.append(f"{round(count / total * 100, 1)}% artykułów")
    if not parts:
        return ""
    return f' <span class="sample-warn">({" · ".join(parts)})</span>'


def render_bullet_list(items: list[tuple[str, int]], total: int | None = None) -> str:
    if not items:
        return ""
    lis = []
    for name, count in items:
        note = sample_note(count, total)
        lis.append(f"<li>{html.escape(name)}: <strong>{count}</strong>{note}</li>")
    return f"<ul class=\"bullet-list\">{''.join(lis)}</ul>"


def render_count_lines(items: list[tuple[str, int]], total: int | None = None) -> str:
    lines = []
    for name, count in items:
        note = sample_note(count, total)
        lines.append(f"{html.escape(name)}: {count}{note}")
    return "<br>\n".join(lines)


def _heat_color(value: float, max_val: float) -> str:
    if max_val <= 0 or value <= 0:
        return "#f8fafb"
    t = value / max_val
    r = int(255 - t * 120)
    g = int(245 - t * 80)
    b = int(252 - t * 40)
    return f"rgb({r},{g},{b})"


def heat_rgb(value: float, max_val: float) -> tuple[float, float, float]:
    """Kolor heatmapy jako tuple 0–1 (matplotlib)."""
    if max_val <= 0 or value <= 0:
        return (248 / 255, 250 / 255, 251 / 255)
    t = value / max_val
    return (
        (255 - t * 120) / 255,
        (245 - t * 80) / 255,
        (252 - t * 40) / 255,
    )


def render_heat_legend(max_val: int, label_low: str = "mniej artykułów", label_high: str = "więcej artykułów") -> str:
    if max_val <= 0:
        return ""
    stops = []
    for i in range(6):
        t = i / 5
        val = max(1, round(max_val * t)) if i else 0
        color = _heat_color(val, max_val)
        stops.append(f'<span class="heat-legend-swatch" style="background:{color}"></span>')
    return (
        '<div class="heat-legend">'
        f'<span class="heat-legend-label">{html.escape(label_low)}</span>'
        f'<div class="heat-legend-bar">{"".join(stops)}</div>'
        f'<span class="heat-legend-label">{html.escape(label_high)}'
        f' (max {max_val})</span>'
        "</div>"
    )


def render_matrix_table(
    row_labels: list[str],
    col_labels: list[str],
    matrix: dict[tuple[str, str], int],
    total_articles: int | None = None,
    caption: str = "",
    *,
    show_legend: bool = True,
    row_keys: list[str] | None = None,
    col_keys: list[str] | None = None,
) -> str:
    row_keys = row_keys or row_labels
    col_keys = col_keys or col_labels
    if len(row_keys) != len(row_labels) or len(col_keys) != len(col_labels):
        raise ValueError("row_keys/col_keys muszą mieć tę samą długość co etykiety")

    max_val = max(matrix.values(), default=0)
    parts = ['<table class="data-matrix">']
    if caption:
        parts.append(f"<caption>{html.escape(caption)}</caption>")
    parts.append("<thead><tr><th></th>")
    for col in col_labels:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for row_lbl, row_key in zip(row_labels, row_keys):
        parts.append(f"<tr><th>{html.escape(row_lbl)}</th>")
        for col_lbl, col_key in zip(col_labels, col_keys):
            val = matrix.get((row_key, col_key), 0)
            bg = _heat_color(val, max_val)
            note = sample_note(val, total_articles)
            parts.append(
                f'<td class="heat" style="background:{bg}">{val}{note}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    if show_legend and max_val > 0:
        parts.append(render_heat_legend(max_val))
    return "".join(parts)
