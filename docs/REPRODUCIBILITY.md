# Reproducibility guide

This guide records the environment, inputs, seeds, commands, and outputs needed to repeat the analysis. Reproduction requires the source arXiv metadata snapshot and the project source code; the raw dataset, sparse matrices, and trained model files are intentionally not part of this documentation package.

## 1. Recorded environment

The project was run on Windows 11 using PowerShell in VS Code. The recorded laptop environment had an Intel Core i5-1135G7 processor and 8 GB RAM. The analysis used sparse matrices and memory-conscious, incremental processing because available memory was limited.

Recorded software versions from the project logs:

| Component | Recorded version |
|---|---:|
| Python | 3.13.7 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Gensim | 4.4.0 |
| Matplotlib | 3.11.2 |
| NLTK | 3.9 or later was installed; exact patch version not retained in the final log |

The exact installed versions of every transitive dependency were not preserved in a lockfile. Before presenting the repository as fully reproducible, save a fresh environment record:

```powershell
python --version
python -m pip freeze | Set-Content requirements-lock.txt -Encoding UTF8
```

Review the generated lock file before committing it.

## 2. Environment setup

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test]"
```

The project requires NumPy, pandas, scikit-learn, SciPy, NLTK, Gensim, Matplotlib, and pytest. The preprocessing stage uses NLTK's English stopwords and WordNet resources. The code checks for those resources and downloads missing NLTK data when network access is available.

For memory-constrained runs, the following environment settings were used to limit numerical-library threading:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

## 3. Input data

Place the Kaggle metadata file at:

```text
data/raw/arxiv-metadata-oai-snapshot.json
```

The recorded snapshot is dataset version 311. Its SHA-256 digest is:

```text
97a7fee00eac38ee5ea92d5cddae351cf3cbb444775a397ac60efb773f217b10
```

Check the file against the digest before processing:

```powershell
Get-FileHash "data/raw/arxiv-metadata-oai-snapshot.json" -Algorithm SHA256
```

The raw file and generated datasets should remain local and should not be committed to GitHub. The ingestion process creates the filtered JSONL dataset and an ingestion manifest under `data/processed/`.

## 4. Deterministic settings

| Stage | Setting |
|---|---|
| Model-selection sample | 8,000 documents |
| Train/validation split | 6,400 / 1,600 |
| Split seed | 2026 |
| Candidate topic counts | 5, 8, 10, 12 |
| NMF validation seeds | 42, 13, 21, 34, 55 |
| NMF validation initialisation | random |
| NMF validation iteration limit | 200 |
| LDA seed | 42 |
| Coherence reference sample | Up to 4,000 non-empty training documents |
| Final NMF topic count | 10 |
| Final NMF seed | 42 |
| Final NMF initialisation | random |
| Final NMF iteration limit | 300 |
| Final NMF observed iterations | 142 |

The validation vectorisers were fitted on the training subset only. The final full-corpus model used the full-corpus feature matrix for retrospective topic description.

## 5. Useful verification commands

Run the automated tests:

```powershell
python -m pytest -q
```

The last recorded complete test output before the chart-generation stage was:

```text
46 passed in 6.17s
```

The chart script was separately syntax-checked with `python -m py_compile` and executed successfully. The chart-generation log records three PNG outputs. Rerun the test suite on the current checkout before publishing.

Final trend calculations and chart creation:

```powershell
python -m arxiv_topic_trends.models.analyze_topic_trends
python -m arxiv_topic_trends.models.create_trend_charts
```

These modules expect the final full-corpus model artefacts and annual input data to exist. Do not rerun if the output directory already exists without first checking the script's overwrite protections and the current saved outputs.

## 6. Expected key outputs

Filtered and preprocessed data:

```text
data/processed/cs_lg_2018_2024.jsonl
data/processed/cs_lg_2018_2024_preprocessed.jsonl
data/processed/ingestion_manifest.json
```

Feature artefacts:

```text
data/features/tfidf_matrix.npz
data/features/count_matrix.npz
data/features/tfidf_vectorizer.joblib
data/features/count_vectorizer.joblib
```

Final model artefacts:

```text
outputs/models/final_full_corpus_nmf_k10/final_nmf_k10_seed42.joblib
outputs/models/final_full_corpus_nmf_k10/document_topic_proportions.npy
outputs/models/final_full_corpus_nmf_k10/document_topic_weights.npy
outputs/models/final_full_corpus_nmf_k10/modelled_document_mask.npy
outputs/models/final_full_corpus_nmf_k10/corpus_metadata.csv
outputs/models/final_full_corpus_nmf_k10/final_nmf_k10_topic_terms.csv
outputs/models/final_full_corpus_nmf_k10/document_dominant_topics.csv
outputs/models/final_full_corpus_nmf_k10/final_model_summary.json
```

Trend-analysis outputs:

```text
outputs/reports/topic_trends_2018_2024/yearly_topic_trends_long.csv
outputs/reports/topic_trends_2018_2024/yearly_topic_trends_wide.csv
outputs/reports/topic_trends_2018_2024/yearly_corpus_counts.csv
outputs/reports/topic_trends_2018_2024/topic_trend_summary.json
outputs/reports/topic_trends_2018_2024/topic_trend_methodology.txt
outputs/reports/topic_trends_2018_2024/yearly_topic_composition_stacked.png
outputs/reports/topic_trends_2018_2024/yearly_dominant_topic_trends.png
outputs/reports/topic_trends_2018_2024/yearly_mean_topic_share_trends.png
outputs/reports/topic_trends_2018_2024/chart_generation_summary.json
```

## 7. Protect the source data and models

The `.gitignore` keeps raw data, processed data, feature matrices, and model files out of Git. The curated annual CSVs and three final charts are useful portfolio outputs and should be selectively allowed through the ignore rules. The updated `.gitignore` supplied with this package does that for the `topic_trends_2018_2024` report directory, while leaving large model artefacts and private local data excluded.

## 8. Reproduction caveat

Re-running the same code on a different snapshot, a different dependency stack, or a different numerical-library build can produce small changes. The source checksum, parameters, seeds, and model outputs should be kept together when results are shared. The version date of the selected Kaggle snapshot was not confirmed in the project record and remains a metadata item to verify if exact temporal provenance is required.
