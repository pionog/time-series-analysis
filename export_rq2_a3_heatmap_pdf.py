"""
Eksport heatmapy RQ2-A3 (wymiarowość × częstotliwość) do PDF/PNG pod Overleaf.

Wyjście: charts/rq2_a3_heatmap.pdf (+ .png)
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from html_tables import heat_rgb
from label_mapping import translate_labels
from research_answers import (
    DIMENSION_COLS,
    FREQUENCY_ORDER,
    INPUT_DIR,
    dimension_frequency_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
DATA_PATH = os.path.join(INPUT_DIR, "SLR(TAGGING - Data description).csv")
OUTPUT_PDF = os.path.join(CHARTS_DIR, "rq2_a3_heatmap.pdf")
OUTPUT_PNG = os.path.join(CHARTS_DIR, "rq2_a3_heatmap.png")

BORDER = "#c5d0dc"
TEXT = "#1a2a3a"
MUTED = "#5a6a7a"
WARN = "#b45309"


def draw_heatmap(
    matrix: dict,
    row_keys: list[str],
    col_keys: list[str],
    row_labels: list[str],
    col_labels: list[str],
    with_both: int,
    articles: int,
) -> plt.Figure:
    nrows = len(row_keys)
    ncols = len(col_keys)
    values = [
        [matrix.get((col, row), 0) for col in col_keys]
        for row in row_keys
    ]
    max_val = max((v for row in values for v in row), default=0)

    fig_w = 4.2 + 0.35 * max(len(c) for c in col_labels)
    fig_h = 0.28 * nrows + 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.55, ncols)
    # y rośnie w dół (invert_yaxis): nagłówki y<0, tabela 0..nrows, legenda i stopka poniżej
    ax.set_ylim(-1.15, nrows + 1.2)
    ax.invert_yaxis()
    ax.axis("off")

    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            val = values[i][j]
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=heat_rgb(val, max_val),
                    edgecolor=BORDER,
                    linewidth=0.8,
                )
            )
            label = str(val)
            if 0 < val < 3:
                label += "*"
            ax.text(
                j + 0.5,
                i + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=TEXT,
            )

    for j, col in enumerate(col_labels):
        ax.text(
            j + 0.5,
            -0.35,
            col,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT,
        )

    for i, row in enumerate(row_labels):
        ax.text(
            -0.12,
            i + 0.5,
            row,
            ha="right",
            va="center",
            fontsize=8.5,
            color=TEXT,
        )

    ax.text(
        ncols / 2,
        -1.2,
        "Oś X: wymiarowość",
        ha="center",
        va="top",
        fontsize=9,
        color=MUTED,
    )
    ax.text(
        -0.55,
        nrows / 2,
        "Oś Y: częstotliwość",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        rotation=90,
    )

    # legenda kolorów (jak w HTML) — pod tabelą
    legend_y = nrows + 0.55
    legend_x0 = 0.0
    legend_w = ncols
    n_stops = 6
    stop_w = legend_w / n_stops
    ax.text(
        legend_x0,
        legend_y - 0.08,
        "mniej artykułów",
        ha="left",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    ax.text(
        legend_x0 + legend_w,
        legend_y - 0.08,
        f"więcej artykułów (max {max_val})",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for k in range(n_stops):
        t = k / (n_stops - 1) if n_stops > 1 else 0
        val = 0 if k == 0 else max(1, round(max_val * t))
        ax.add_patch(
            Rectangle(
                (legend_x0 + k * stop_w, legend_y),
                stop_w,
                0.22,
                facecolor=heat_rgb(val, max_val),
                edgecolor=BORDER,
                linewidth=0.5,
            )
        )

    # stopka pod legendą (pasek kończy się ok. nrows + 0.77)
    ax.text(
        0,
        nrows + 0.9,
        f"RQ2-A3 · artykuły z obiema informacjami: {with_both}/{articles}   "
        f"* znikoma próbka (<3 artykuły)",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )

    fig.subplots_adjust(left=0.28, right=0.98, top=0.96, bottom=0.06)
    return fig


def main() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    matrix, with_both, articles = dimension_frequency_matrix(DATA_PATH)
    if not matrix:
        raise SystemExit("Brak danych do heatmapy RQ2-A3.")

    row_keys = [
        f for f in FREQUENCY_ORDER
        if any(matrix.get((d, f), 0) for d in DIMENSION_COLS)
    ]
    col_keys = list(DIMENSION_COLS)
    row_labels = translate_labels(row_keys)
    col_labels = translate_labels(col_keys)

    fig = draw_heatmap(
        matrix, row_keys, col_keys, row_labels, col_labels, with_both, articles
    )
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUTPUT_PNG, format="png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"Zapisano: {OUTPUT_PDF}")
    print(f"Zapisano: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
