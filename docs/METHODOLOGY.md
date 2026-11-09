# Methodology

This document records the decisions used to turn arXiv metadata into a set of yearly topic-share measures. It is intended to make the analysis understandable and repeatable, including the choices that could affect the result.

## 1. Research scope

The corpus is restricted to records whose primary category is exactly `cs.LG` and whose first-submission year is from 2018 through 2024. The unit of analysis is one arXiv metadata record / abstract. The model does not use full paper PDFs, citations, author networks, or external impact measures.

## 2. Data source and extraction

- Source: Cornell University arXiv metadata snapshot distributed through Kaggle.
- Recorded dataset version: 311.
- Source filename: `arxiv-metadata-oai-snapshot.json`.
- Recorded source SHA-256: `97a7fee00eac38ee5ea92d5cddae351cf3cbb444775a397ac60efb773f217b10`.
- Approximate source size: 5.67 GB.
- Filtered records: 87,945.

A record is selected when the first category token is `cs.LG`. The year is based on the first-submission date, not the latest metadata update. The exact Kaggle publication date for version 311 was not confirmed in the project record, so it is intentionally not guessed.

## 3. Validation of filtered data

The filtered corpus was checked for valid records, unique arXiv IDs, missing titles and abstracts, and agreement with the ingestion manifest. The recorded checks found 87,945 records, 87,945 unique IDs, and no missing titles or abstracts.

## 4. Abstract cleaning

The cleaning pipeline is applied to each abstract and stores the original abstract with its derived token list:

1. HTML entity unescaping;
2. lowercasing;
3. alphanumeric token extraction;
4. removal of single-character and numeric-only tokens;
5. removal of NLTK English stopwords;
6. WordNet noun lemmatisation.

Domain-specific stopword candidates are reviewed, but no domain-specific terms are automatically removed in the recorded run. The purpose is to avoid stripping words that may distinguish machine-learning topics.

The preprocessing result contains 87,945 rows, 9,997,305 tokens, a mean of 113.68 tokens per abstract, a median of 113, and 85,573 unique tokens before vectoriser filtering. One record (`1902.02322`) has an empty token list after cleaning. It remains in the corpus metadata and is marked unassigned in the final model and trend calculations.

## 5. Feature representations

Two sparse feature matrices were generated, each with 87,945 document rows and 8,000 feature columns:

- TF–IDF for NMF, with `max_features=8000`, `min_df=5`, and sublinear term frequency;
- term counts for LDA, with the matching token vocabulary settings.

Sparse matrices were retained to reduce memory use. The original full-corpus vectorisers were used for the final retrospective model. The validation experiment below used fresh vectoriser clones fitted only on training documents.

## 6. Leakage-free model-selection experiment

A deterministic sample of 8,000 documents was used for model comparison. The sample was divided into:

- training: 6,400 documents;
- validation: 1,600 documents.

The split seed was 2026. Vectorisers were cloned from the agreed configurations, then fitted on training documents. Validation documents were transformed without refitting. The resulting TF–IDF training vocabulary contained 6,257 terms. Coherence used up to 4,000 non-empty training documents and a dictionary built from those training texts.

This procedure prevents validation documents from affecting the vocabulary or IDF estimates used in the validation experiment. It does not make the validation split chronological; papers from different years can occur in both subsets.

## 7. Candidate models and metrics

Topic counts tested: `K = 5, 8, 10, 12`.

NMF was run at each K with five random seeds: `42, 13, 21, 34, 55`. The validation NMF runs used random initialisation and a maximum of 200 iterations. The reported NMF metrics are mean validation reconstruction error over seeds, mean top-15 lexical stability, and c_v coherence for the seed-42 model.

Lexical stability was calculated by comparing each pair of NMF runs at the same K. Topics were optimally matched using the Hungarian algorithm, and Jaccard similarity was calculated over top-15 term sets. The average pairwise similarity is a measure of top-word overlap, not proof of semantic equivalence.

LDA was run once for each K with seed 42. It was evaluated using held-out perplexity and c_v coherence. LDA perplexity and NMF reconstruction error are not directly comparable because they measure different aspects of different models.

### Recorded validation results

| K | NMF validation error (mean) | NMF top-15 stability (mean) | NMF c_v | LDA held-out perplexity | LDA c_v |
|---:|---:|---:|---:|---:|---:|
| 5 | 38.7803 | 0.8288 | 0.5687 | 1,925.2945 | 0.4039 |
| 8 | 38.5987 | 0.7613 | 0.5788 | 2,027.7662 | 0.4178 |
| 10 | 38.4972 | 0.9195 | 0.5816 | 2,099.7763 | 0.4400 |
| 12 | 38.4154 | 0.7672 | 0.5906 | 2,120.5340 | 0.4573 |

K=10 was selected for its strongest observed NMF lexical stability and a c_v score close to the K=12 result. K=12's coherence was higher, but its lexical stability was lower. The choice reflects a balance between stability, interpretability, and topic granularity. It is not a claim that K=10 is objectively optimal for every use.

## 8. Final NMF model

The selected K=10 NMF configuration was refitted on all 87,944 records with non-zero feature rows using the full-corpus TF–IDF matrix. The final fit used:

- components: 10;
- initialisation: random;
- random seed: 42;
- maximum iterations: 300;
- observed iterations: 142;
- training reconstruction error: 285.671509.

No convergence warning was recorded for this final fit. One document with an empty cleaned token list was not passed to NMF. Its metadata row was retained and its modelled flag was set to false.

The final outputs include the fitted model, raw document-topic weights, normalised document-topic proportions, a modelled-document mask, corpus metadata, topic terms, and dominant topic assignments.

## 9. Topic interpretation

The final model's topic numbers and top terms are the reference for all trend charts. The working labels in the README and report were assigned from those final full-corpus top terms. They are not ground-truth labels and have not been independently confirmed.

An important distinction must be preserved: the existing 50-paper review CSV was created for the earlier validation model. The validation model's topic ordering differs from the final full-corpus model. Its topic numbers and labels must not be copied directly onto the final trend charts. A final-model review should use representative papers derived from the final model outputs and the review template in `docs/templates/`.

## 10. Annual topic measures

For each modelled document, the ten NMF activation weights were normalised to sum to one. The annual trend calculation then produced two measures.

### Dominant-topic share

For a given year, the dominant topic is the topic with the largest normalised activation in each document. The dominant-topic share is the percentage of modelled papers in that year assigned to each topic by this rule. Every modelled paper contributes to exactly one topic in this measure.

### Mean topic activation share

For a given year, the normalised activations are averaged separately for each topic across all modelled papers in that year. A document may contribute to multiple topics. The ten mean activation shares sum to 100% per year, within rounding tolerance.

The unassigned paper is retained in the total paper count and excluded from topic-share denominators. There is one such paper in 2019; all other yearly modelled counts equal their total paper counts.

## 11. Software and reproducibility controls

The recorded environment used Python 3.13.7, scikit-learn 1.9.1, Gensim 4.4.0, NumPy 2.5.4, and Matplotlib 3.11.2. The final trend analysis records its input files, method definitions, years, topic count, and output list in JSON and text summaries. Seeds are recorded for the sample split, model runs, and final model.

The scripts use sparse matrices and bounded/batched operations where practical. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for commands and expected outputs.
