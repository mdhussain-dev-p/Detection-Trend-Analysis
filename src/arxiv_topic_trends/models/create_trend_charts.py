from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = ROOT / "outputs" / "reports" / "topic_trends_2018_2024"

WIDE_PATH = OUTPUT_DIR / "yearly_topic_trends_wide.csv"
LONG_PATH = OUTPUT_DIR / "yearly_topic_trends_long.csv"
CORPUS_PATH = OUTPUT_DIR / "yearly_corpus_counts.csv"

YEARS = list(range(2018, 2025))

TOPIC_LABELS = {
    1: "Supervised learning",
    2: "Adversarial robustness",
    3: "Predictive and general ML",
    4: "Bandits and online optimisation",
    5: "Reinforcement learning",
    6: "Gradient-based optimisation",
    7: "Deep neural networks",
    8: "Federated learning",
    9: "Graph neural networks",
    10: "Large language models",
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    started = time.perf_counter()

    print("\n[1/4] Checking trend-analysis files...", flush=True)

    for path in (WIDE_PATH, LONG_PATH, CORPUS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    wide_rows = read_csv(WIDE_PATH)
    long_rows = read_csv(LONG_PATH)
    corpus_rows = read_csv(CORPUS_PATH)

    if len(wide_rows) != 7:
        raise ValueError("Expected exactly seven yearly wide records.")

    if len(long_rows) != 70:
        raise ValueError("Expected exactly 70 topic-year observations.")

    if len(corpus_rows) != 7:
        raise ValueError("Expected seven annual corpus-count records.")

    if [int(row["year"]) for row in wide_rows] != YEARS:
        raise ValueError("Unexpected year sequence in wide trend data.")

    if len(long_rows) != 70:
        raise ValueError("Trend rows are incomplete.")

    print("  Seven years verified.")
    print("  Seventy topic-year observations verified.")

    print("\n[2/4] Validating topic-share percentages...", flush=True)

    mean_matrix = []
    dominant_matrix = []

    for row in wide_rows:
        mean_values = []
        dominant_values = []

        for topic in range(1, 11):
            mean_values.append(
                float(row[f"topic_{topic:02d}_mean_activation_share_pct"])
            )
            dominant_values.append(
                float(row[f"topic_{topic:02d}_dominant_share_pct"])
            )

        if abs(sum(mean_values) - 100.0) > 0.002:
            raise ValueError(f"Mean topic shares do not sum to 100 in {row['year']}.")

        if abs(sum(dominant_values) - 100.0) > 0.002:
            raise ValueError(f"Dominant shares do not sum to 100 in {row['year']}.")

        mean_matrix.append(mean_values)
        dominant_matrix.append(dominant_values)

    corpus_counts = [
        int(row["total_papers"]) for row in corpus_rows
    ]

    mean_matrix = list(map(list, zip(*mean_matrix)))
    dominant_matrix = list(map(list, zip(*dominant_matrix)))

    print("  Both topic-share measures pass validation.")

    chart_paths = [
        OUTPUT_DIR / "yearly_topic_composition_stacked.png",
        OUTPUT_DIR / "yearly_dominant_topic_trends.png",
        OUTPUT_DIR / "yearly_mean_topic_share_trends.png",
    ]

    for path in chart_paths:
        if path.exists():
            raise FileExistsError(
                f"Chart already exists. Refusing to overwrite: {path}"
            )

    print("\n[3/4] Creating three publication-quality charts...", flush=True)

    # Chart 1 — Stacked area of mean topic shares.
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.stackplot(
        YEARS,
        mean_matrix,
        labels=[TOPIC_LABELS[i] for i in range(1, 11)],
        alpha=0.85,
    )

    ax.set_title(
        "Evolution of Machine Learning Research Topics (2018–2024)",
        fontsize=15,
    )
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Mean normalised topic activation share (%)")
    ax.set_xlim(2018, 2024)
    ax.set_xticks(YEARS)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Research topics",
    )

    fig.tight_layout()
    fig.savefig(chart_paths[0], dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("  Chart 1/3 completed: stacked topic composition.")

    # Chart 2 — Dominant-topic share by year.
    fig, ax = plt.subplots(figsize=(14, 7))

    for index, values in enumerate(dominant_matrix):
        ax.plot(
            YEARS,
            values,
            marker="o",
            linewidth=1.8,
            label=TOPIC_LABELS[index + 1],
        )

    ax.set_title(
        "Dominant Research Topic Share by Year",
        fontsize=15,
    )
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Share of modelled papers (%)")
    ax.set_xticks(YEARS)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Research topics",
    )

    fig.tight_layout()
    fig.savefig(chart_paths[1], dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("  Chart 2/3 completed: dominant-topic trends.")

    # Chart 3 — Mean topic activation share.
    fig, ax = plt.subplots(figsize=(14, 7))

    for index, values in enumerate(mean_matrix):
        ax.plot(
            YEARS,
            values,
            marker="o",
            linewidth=1.8,
            label=TOPIC_LABELS[index + 1],
        )

    ax.set_title(
        "Mean Topic Activation Share by Year",
        fontsize=15,
    )
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Mean normalised activation share (%)")
    ax.set_xticks(YEARS)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="Research topics",
    )

    fig.tight_layout()
    fig.savefig(chart_paths[2], dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("  Chart 3/3 completed: mean activation trends.")

    print("\n[4/4] Saving chart-generation record...", flush=True)

    chart_summary = {
        "phase": "Phase 10C - chart generation",
        "status": "completed",
        "years": YEARS,
        "topic_count": 10,
        "charts_created": [
            path.name for path in chart_paths
        ],
        "topic_labels": TOPIC_LABELS,
        "label_status": (
            "Working labels derived from full-corpus topic terms; "
            "independent human confirmation is still pending."
        ),
        "source_files": [
            WIDE_PATH.name,
            LONG_PATH.name,
            CORPUS_PATH.name,
        ],
        "corpus_paper_counts": corpus_counts,
        "elapsed_minutes": round(
            (time.perf_counter() - started) / 60, 2
        ),
    }

    with (OUTPUT_DIR / "chart_generation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(chart_summary, handle, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("PHASE 10C COMPLETED")
    print("============================================================")
    print("Charts generated: 3/3")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Elapsed: {chart_summary['elapsed_minutes']:.2f} minutes")

    for path in chart_paths:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")

    print("[==============================] 100% — charts saved.")

    return OUTPUT_DIR


if __name__ == "__main__":
    main()
