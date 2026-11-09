from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = (
    ROOT / "outputs" / "models" /
    "final_full_corpus_nmf_k10"
)

METADATA_PATH = MODEL_DIR / "corpus_metadata.csv"
PROPORTIONS_PATH = MODEL_DIR / "document_topic_proportions.npy"
MASK_PATH = MODEL_DIR / "modelled_document_mask.npy"
TERMS_PATH = MODEL_DIR / "final_nmf_k10_topic_terms.csv"

OUTPUT_DIR = ROOT / "outputs" / "reports" / "topic_trends_2018_2024"

YEARS = list(range(2018, 2025))
TOPIC_COUNT = 10


def progress(step, total, label, started):
    width = 28
    fraction = step / total
    filled = int(width * fraction)
    bar = "=" * filled + "-" * (width - filled)
    elapsed = (time.perf_counter() - started) / 60

    print(
        f"[{bar}] {step}/{total} ({fraction * 100:.0f}%) "
        f"{label} | elapsed: {elapsed:.2f} min",
        flush=True,
    )


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_boolean(value):
    value = str(value).strip().lower()

    if value in ("true", "1", "yes"):
        return True

    if value in ("false", "0", "no"):
        return False

    raise ValueError(f"Unexpected Boolean value in metadata: {value!r}")


def calculate_year_metrics(year, years, modelled_mask, proportions):
    """
    Calculate two annual topic prevalence measures.

    1. Dominant-topic share:
       Fraction of modelled papers whose highest activation belongs to a topic.

    2. Mean topic activation share:
       Average normalised NMF activation for a topic across modelled papers.
       These values sum to 100% per year.

    Neither measure is a calibrated probability.
    """
    year_mask = years == year
    eligible_mask = year_mask & modelled_mask

    total_papers = int(year_mask.sum())
    modelled_papers = int(eligible_mask.sum())
    unassigned_papers = total_papers - modelled_papers

    if modelled_papers == 0:
        raise ValueError(f"No modelled papers found for year {year}.")

    annual_proportions = proportions[eligible_mask]

    if not np.isfinite(annual_proportions).all():
        raise ValueError(f"Non-finite topic values found in {year}.")

    row_sums = annual_proportions.sum(axis=1)

    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            f"Document topic shares do not sum to one in {year}."
        )

    # Average share across documents.
    mean_share_pct = annual_proportions.mean(axis=0) * 100

    # Dominant topic of each paper.
    dominant_indices = np.argmax(annual_proportions, axis=1)

    dominant_counts = np.bincount(
        dominant_indices,
        minlength=TOPIC_COUNT,
    )

    dominant_share_pct = (
        dominant_counts / modelled_papers
    ) * 100

    return {
        "total_papers": total_papers,
        "modelled_papers": modelled_papers,
        "unassigned_papers": unassigned_papers,
        "dominant_counts": dominant_counts,
        "dominant_share_pct": dominant_share_pct,
        "mean_share_pct": mean_share_pct,
    }


def read_topic_terms():
    terms_by_topic = {topic: [] for topic in range(1, TOPIC_COUNT + 1)}

    with TERMS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            topic = int(row["topic_number"])
            rank = int(row["rank"])

            if topic not in terms_by_topic:
                raise ValueError(f"Unexpected topic number: {topic}")

            if rank <= 10:
                terms_by_topic[topic].append(
                    (rank, row["term"])
                )

    result = {}

    for topic, terms in terms_by_topic.items():
        ordered = [
            word for _, word in sorted(terms)
        ]

        if len(ordered) != 10:
            raise ValueError(
                f"Topic {topic} does not contain 10 top terms."
            )

        result[topic] = ordered

    return result


def create_charts(years, soft_share_matrix, dominant_share_matrix):
    """
    Create visualisations if matplotlib is installed.
    The CSV and JSON reports are produced independently.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Matplotlib is not installed. CSV and JSON reports "
            "will still be available.",
            flush=True,
        )
        return []

    chart_files = []

    labels = [f"Topic {i}" for i in range(1, TOPIC_COUNT + 1)]

    # Stacked area: mean normalised topic activation shares.
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.stackplot(
        years,
        soft_share_matrix.T,
        labels=labels,
        alpha=0.85,
    )

    ax.set_title(
        "Research Topic Composition in arXiv cs.LG Papers (2018–2024)"
    )
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Mean normalised topic activation share (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(years)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="NMF topics",
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "yearly_topic_composition_stacked.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_files.append(path.name)

    # Line chart: dominant topic shares.
    fig, ax = plt.subplots(figsize=(12, 6))

    for index in range(TOPIC_COUNT):
        ax.plot(
            years,
            dominant_share_matrix[:, index],
            marker="o",
            linewidth=1.7,
            label=f"Topic {index + 1}",
        )

    ax.set_title("Dominant Research Topic Share by Year (2018–2024)")
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Share of modelled papers (%)")
    ax.set_xticks(years)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="NMF topics",
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "yearly_dominant_topic_trends.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_files.append(path.name)

    # Line chart: mean normalised topic activation shares.
    fig, ax = plt.subplots(figsize=(12, 6))

    for index in range(TOPIC_COUNT):
        ax.plot(
            years,
            soft_share_matrix[:, index],
            marker="o",
            linewidth=1.7,
            label=f"Topic {index + 1}",
        )

    ax.set_title("Mean Topic Activation Share by Year (2018–2024)")
    ax.set_xlabel("Initial submission year")
    ax.set_ylabel("Mean normalised activation share (%)")
    ax.set_xticks(years)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        title="NMF topics",
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "yearly_mean_topic_share_trends.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    chart_files.append(path.name)

    return chart_files


def main():
    started = time.perf_counter()

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {OUTPUT_DIR}"
        )

    for path in (
        METADATA_PATH,
        PROPORTIONS_PATH,
        MASK_PATH,
        TERMS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("\nPHASE 10B — YEARLY TOPIC TREND ANALYSIS", flush=True)

    # --------------------------------------------------------
    # Step 1: Load and validate data.
    # --------------------------------------------------------
    print("\n[1/5] Loading final model outputs...", flush=True)

    proportions = np.load(
        PROPORTIONS_PATH,
        allow_pickle=False,
    )

    modelled_mask = np.asarray(
        np.load(MASK_PATH, allow_pickle=False),
        dtype=bool,
    ).reshape(-1)

    metadata_rows = []

    with METADATA_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            metadata_rows.append(row)

    document_count = len(metadata_rows)

    if proportions.shape != (document_count, TOPIC_COUNT):
        raise ValueError(
            f"Unexpected document-topic shape: {proportions.shape}"
        )

    if modelled_mask.shape[0] != document_count:
        raise ValueError("Modelled-document mask length does not match metadata.")

    if not np.isfinite(proportions).all():
        raise ValueError("Non-finite topic proportions found.")

    row_indices = np.array(
        [int(row["row_index"]) for row in metadata_rows]
    )

    if not np.array_equal(row_indices, np.arange(document_count)):
        raise ValueError("Metadata row indices are not in original dataset order.")

    years = np.array(
        [int(row["year"]) for row in metadata_rows],
        dtype=np.int16,
    )

    saved_mask = np.array(
        [parse_boolean(row["is_modelled"]) for row in metadata_rows],
        dtype=bool,
    )

    if not np.array_equal(saved_mask, modelled_mask):
        raise ValueError("Modelled mask does not match corpus metadata.")

    categories = {row["primary_category"] for row in metadata_rows}

    if categories != {"cs.LG"}:
        raise ValueError(f"Unexpected primary categories: {categories}")

    if set(np.unique(years)) - set(YEARS):
        raise ValueError("Unexpected submission years found.")

    modelled_values = proportions[modelled_mask]
    unassigned_values = proportions[~modelled_mask]

    if not np.allclose(modelled_values.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Modelled document proportions are not normalised.")

    if unassigned_values.size and not np.allclose(unassigned_values, 0.0):
        raise ValueError("Unassigned documents must have all-zero topic shares.")

    terms_by_topic = read_topic_terms()

    print(f"  Documents: {document_count:,}")
    print(f"  Modelled: {modelled_mask.sum():,}")
    print(f"  Unassigned: {(~modelled_mask).sum():,}")
    print(f"  Topics: {TOPIC_COUNT}")

    # --------------------------------------------------------
    # Step 2: Calculate year-specific measures.
    # --------------------------------------------------------
    print("\n[2/5] Calculating annual topic trends...", flush=True)

    long_rows = []
    wide_rows = []
    corpus_rows = []

    soft_share_matrix = []
    dominant_share_matrix = []

    annual_summary = {}

    for year in YEARS:
        metrics = calculate_year_metrics(
            year,
            years,
            modelled_mask,
            proportions,
        )

        soft_shares = metrics["mean_share_pct"]
        dominant_shares = metrics["dominant_share_pct"]
        dominant_counts = metrics["dominant_counts"]

        if not np.isclose(soft_shares.sum(), 100.0, atol=1e-4):
            raise ValueError(f"Mean topic shares do not sum to 100% in {year}.")

        if not np.isclose(dominant_shares.sum(), 100.0, atol=1e-4):
            raise ValueError(f"Dominant topic shares do not sum to 100% in {year}.")

        soft_share_matrix.append(soft_shares)
        dominant_share_matrix.append(dominant_shares)

        corpus_rows.append({
            "year": year,
            "total_papers": metrics["total_papers"],
            "modelled_papers": metrics["modelled_papers"],
            "unassigned_papers": metrics["unassigned_papers"],
            "modelled_coverage_pct": (
                metrics["modelled_papers"] /
                metrics["total_papers"] * 100
            ),
        })

        wide_row = {
            "year": year,
            "total_papers": metrics["total_papers"],
            "modelled_papers": metrics["modelled_papers"],
            "unassigned_papers": metrics["unassigned_papers"],
        }

        for topic_index in range(TOPIC_COUNT):
            topic_number = topic_index + 1

            mean_share = float(soft_shares[topic_index])
            dominant_share = float(dominant_shares[topic_index])
            dominant_count = int(dominant_counts[topic_index])

            terms = ", ".join(terms_by_topic[topic_number])

            long_rows.append({
                "year": year,
                "total_papers_in_year": metrics["total_papers"],
                "modelled_papers_in_year": metrics["modelled_papers"],
                "unassigned_papers_in_year": metrics["unassigned_papers"],
                "topic_number": topic_number,
                "top_10_terms": terms,
                "dominant_topic_papers": dominant_count,
                "dominant_topic_share_pct": round(dominant_share, 6),
                "mean_topic_activation_share_pct": round(mean_share, 6),
            })

            wide_row[f"topic_{topic_number:02d}_dominant_share_pct"] = round(
                dominant_share,
                6,
            )

            wide_row[f"topic_{topic_number:02d}_mean_activation_share_pct"] = round(
                mean_share,
                6,
            )

        wide_rows.append(wide_row)

        dominant_topic_number = int(np.argmax(soft_shares)) + 1

        annual_summary[str(year)] = {
            "total_papers": metrics["total_papers"],
            "modelled_papers": metrics["modelled_papers"],
            "unassigned_papers": metrics["unassigned_papers"],
            "largest_mean_share_topic": dominant_topic_number,
            "largest_mean_share_pct": float(
                soft_shares[dominant_topic_number - 1]
            ),
            "dominant_topic_paper_counts": {
                str(index + 1): int(count)
                for index, count in enumerate(dominant_counts)
            },
        }

        print(
            f"  {year}: {metrics['total_papers']:,} papers; "
            f"{metrics['modelled_papers']:,} modelled; "
            f"largest mean-share topic: {dominant_topic_number}",
            flush=True,
        )

    soft_share_matrix = np.asarray(soft_share_matrix)
    dominant_share_matrix = np.asarray(dominant_share_matrix)

    # --------------------------------------------------------
    # Step 3: Save CSV outputs.
    # --------------------------------------------------------
    print("\n[3/5] Saving yearly CSV reports...", flush=True)

    long_path = OUTPUT_DIR / "yearly_topic_trends_long.csv"

    write_csv(
        long_path,
        long_rows,
        [
            "year",
            "total_papers_in_year",
            "modelled_papers_in_year",
            "unassigned_papers_in_year",
            "topic_number",
            "top_10_terms",
            "dominant_topic_papers",
            "dominant_topic_share_pct",
            "mean_topic_activation_share_pct",
        ],
    )

    wide_path = OUTPUT_DIR / "yearly_topic_trends_wide.csv"

    wide_fields = [
        "year",
        "total_papers",
        "modelled_papers",
        "unassigned_papers",
    ]

    for topic in range(1, TOPIC_COUNT + 1):
        wide_fields.extend([
            f"topic_{topic:02d}_dominant_share_pct",
            f"topic_{topic:02d}_mean_activation_share_pct",
        ])

    write_csv(wide_path, wide_rows, wide_fields)

    corpus_path = OUTPUT_DIR / "yearly_corpus_counts.csv"

    write_csv(
        corpus_path,
        corpus_rows,
        [
            "year",
            "total_papers",
            "modelled_papers",
            "unassigned_papers",
            "modelled_coverage_pct",
        ],
    )

    # --------------------------------------------------------
    # Step 4: Produce visualisations.
    # --------------------------------------------------------
    print("\n[4/5] Creating charts...", flush=True)

    chart_files = create_charts(
        YEARS,
        soft_share_matrix,
        dominant_share_matrix,
    )

    # --------------------------------------------------------
    # Step 5: Save methodology and summary.
    # --------------------------------------------------------
    print("\n[5/5] Writing summary and methodology...", flush=True)

    summary = {
        "phase": "Phase 10B - yearly topic trend analysis",
        "status": "completed",
        "dataset": {
            "source": "Filtered arXiv cs.LG metadata",
            "years": YEARS,
            "total_documents": int(document_count),
            "modelled_documents": int(modelled_mask.sum()),
            "unassigned_documents": int((~modelled_mask).sum()),
            "topic_count": TOPIC_COUNT,
        },
        "methodology": {
            "dominant_topic_share": (
                "Percentage of modelled papers in a year whose largest "
                "normalised NMF activation belongs to that topic."
            ),
            "mean_topic_activation_share": (
                "Average normalised NMF activation for a topic across "
                "all modelled papers in a year. The ten topic shares "
                "sum to 100 percent for each year."
            ),
            "normalisation": (
                "Topic activations are normalised within each document. "
                "They are not calibrated probabilities."
            ),
            "unassigned_handling": (
                "Zero-feature documents are retained in overall corpus "
                "counts but excluded from topic-share denominators."
            ),
            "scope": (
                "Retrospective descriptive trends. These results do not "
                "establish causation or predict future research activity."
            ),
            "topic_numbering": (
                "Topic numbers refer to the final full-corpus NMF K=10 model, "
                "not the earlier pilot model."
            ),
        },
        "annual_summary": annual_summary,
        "outputs": [
            long_path.name,
            wide_path.name,
            corpus_path.name,
            *chart_files,
        ],
        "charts_created": chart_files,
    }

    summary_path = OUTPUT_DIR / "topic_trend_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    report_path = OUTPUT_DIR / "topic_trend_methodology.txt"

    report_path.write_text(
        "TOPIC DETECTION AND TREND ANALYSIS\n"
        "YEARLY RESEARCH TRENDS — 2018 TO 2024\n"
        "=====================================\n\n"
        "Final model: NMF, K=10, trained on the full filtered corpus.\n"
        "Topic numbers refer to the full-corpus model.\n\n"
        "Two annual measures are calculated:\n"
        "1. Dominant-topic share: percentage of papers assigned primarily\n"
        "   to a topic in a particular year.\n"
        "2. Mean topic activation share: average normalised NMF activation\n"
        "   for each topic across modelled papers in that year.\n\n"
        "Both measures are calculated using modelled papers as the topic\n"
        "denominator. Unassigned papers remain in annual corpus counts but\n"
        "are excluded from topic-share calculations.\n\n"
        "The mean activation shares sum to 100 percent for each year.\n"
        "NMF activations are not calibrated probabilities.\n\n"
        "This is retrospective descriptive analysis, not causal inference\n"
        "or forecasting.\n",
        encoding="utf-8",
    )

    elapsed = (time.perf_counter() - started) / 60

    print("\n============================================================")
    print("PHASE 10B COMPLETED")
    print("============================================================")
    print(f"Years analysed: {YEARS[0]}–{YEARS[-1]}")
    print(f"Total papers: {document_count:,}")
    print(f"Modelled papers: {modelled_mask.sum():,}")
    print(f"Topics analysed: {TOPIC_COUNT}")
    print(f"Elapsed time: {elapsed:.2f} minutes")
    print(f"\nOutput folder: {OUTPUT_DIR}")
    print(f"Long report: {long_path}")
    print(f"Wide report: {wide_path}")
    print(f"Corpus counts: {corpus_path}")
    print(f"Summary: {summary_path}")
    print(f"Charts: {len(chart_files)}")

    print("\nYEARLY LEADING TOPICS:")

    for year in YEARS:
        result = annual_summary[str(year)]
        print(
            f"  {year}: Topic {result['largest_mean_share_topic']} "
            f"({result['largest_mean_share_pct']:.2f}% mean share)"
        )

    progress(5, 5, "All trend reports and available charts saved", started)


if __name__ == "__main__":
    main()
