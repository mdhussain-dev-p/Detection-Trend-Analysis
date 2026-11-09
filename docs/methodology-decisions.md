# Methodology decisions and final status

This file replaces the initial planning-stage checklist. It records the choices actually used in the completed analysis and keeps unresolved research decisions visible.

## Decisions used in the recorded run

1. Source: Cornell University arXiv metadata snapshot distributed through Kaggle; version 311 is recorded.
2. Source file: `arxiv-metadata-oai-snapshot.json`; SHA-256 `97a7fee00eac38ee5ea92d5cddae351cf3cbb444775a397ac60efb773f217b10`.
3. Include records whose primary category is exactly `cs.LG`.
4. Use the first-submission date to determine the year; analyse 2018–2024 inclusive.
5. Clean abstracts with lowercase alphanumeric tokenisation, NLTK English stopwords, and WordNet noun lemmatisation.
6. Do not automatically remove the reviewed domain-specific candidate terms.
7. Build TF–IDF and count matrices with up to 8,000 features and `min_df=5`.
8. Compare K=5, 8, 10, and 12.
9. Use a seeded 8,000-document model-selection sample with a 6,400/1,600 split and train-only vectoriser fitting.
10. Run five random NMF seeds per K and measure top-15 lexical stability after Hungarian matching.
11. Select NMF K=10 as a balance of observed stability, coherence, and topic detail.
12. Fit the final NMF K=10 model on all 87,944 documents with usable features, using random initialisation and seed 42.
13. Keep the one zero-token record in metadata and annual counts but mark it unassigned for topic calculations.
14. Calculate both dominant-topic share and mean normalised activation share by first-submission year.
15. Treat the topic labels as provisional until representative papers from the final model are reviewed.

## Items still unresolved

- The exact publication date of Kaggle dataset version 311 was not recorded.
- Independent human review of the final full-corpus topics is pending.
- No Cohen's kappa or other human agreement statistic is available until two independent sets of ratings exist.
- A chronological holdout experiment has not been run.
- A complete dependency lockfile has not been recorded.

## Important model-version note

The Phase 9 50-paper review sample and its original label template correspond to the validation model. The final full-corpus model has a different topic order. Use the new `docs/templates/final_model_topic_label_review_template.csv` for final-model topic labels and extract new final-model paper examples before making a final confirmation.
