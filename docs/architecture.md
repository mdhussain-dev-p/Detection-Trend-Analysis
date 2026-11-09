# Project architecture

The project follows a staged pipeline so that data ingestion, text preparation, modelling, and reporting can be checked separately.

```text
arXiv metadata snapshot
        |
        v
Streaming inspection and filtering
(primary cs.LG; first submission 2018–2024)
        |
        v
Filtered JSONL + ingestion manifest
        |
        v
Exploratory checks and abstract cleaning
        |
        v
Sparse TF–IDF and count representations
        |                         |
        v                         v
      NMF                       LDA
        |                         |
        +---- model comparison ---+
                    |
                    v
  Leakage-free validation and stability checks
                    |
                    v
     Final full-corpus NMF, K=10
                    |
                    v
       Per-document topic activations
                    |
                    v
      Yearly topic-share calculations
                    |
                    v
     Tables, figures, report and review
```

## Main package areas

| Area | Responsibility | Representative modules |
|---|---|---|
| `data/` | Inspect the source snapshot, filter records, and explore the selected corpus | `ingest.py`, `inspect_snapshot.py`, `explore_filtered.py` |
| `preprocessing/` | Clean abstracts and retain tokenised text | `clean_abstracts.py` |
| `features/` | Build sparse TF–IDF and count matrices | `vectorize.py` |
| `models/` | Pilot, validate, compare, fit the final model, and calculate trends | `train_pilot.py`, `evaluate_pilot.py`, `compare_topics_phase7.py`, `leakage_free_validation.py`, `train_final_full_corpus.py`, `analyze_topic_trends.py`, `create_trend_charts.py` |
| `tests/` | Test data handling, transformations, model metrics, and annual calculations | `test_*.py` |
| `docs/` | Record decisions, definitions, caveats, and reproducibility steps | Markdown project documents |

## Data flow and outputs

Large inputs and intermediate matrices remain local under `data/`. Final model artefacts are stored under `outputs/models/final_full_corpus_nmf_k10/`. Annual trend tables, JSON summaries, methodology notes, and charts are stored under `outputs/reports/topic_trends_2018_2024/`.

The final trend calculations use the same full-corpus model for all seven years. This preserves a common topic numbering across years. The charts use working labels based on the final model's top terms; these labels remain subject to independent human review.
