# Limitations and interpretation boundaries

A topic model can make a large collection of abstracts easier to inspect, but it also compresses a complicated literature into a small number of word patterns. The figures in this project are useful summaries, not a complete or definitive account of machine-learning research. The following limitations should remain visible whenever the results are presented.

## 1. Corpus coverage

The analysis includes only records whose primary arXiv category is `cs.LG`. Work listed primarily under computer vision, artificial intelligence, statistics, signal processing, robotics, or other categories is not included unless its primary category is `cs.LG`. Cross-listed work for which `cs.LG` is not the primary category is also excluded. The results therefore describe this selected arXiv slice, not the entire machine-learning research landscape.

The source is an arXiv metadata snapshot. It is not a census of peer-reviewed journal and conference papers, industry reports, books, or research that is not deposited on arXiv. Different communities have different preprint practices.

## 2. Abstracts are a compressed view of papers

The model uses titles and abstracts rather than full papers. Abstracts are short summaries and may not mention every method, contribution, or application in a paper. A topic model built on abstracts can miss details that would be clear from methods, experiments, or full-text sections.

## 3. Bag-of-words features do not understand meaning

TF–IDF and count vectors represent text through token frequencies. This makes the model efficient and relatively easy to inspect, but it does not fully capture word order, context, negation, or semantic similarity. A word can be used in more than one sense. Related concepts can be described with different vocabulary, and a broad topic can contain several subthemes.

The chosen cleaning process removes common English stopwords and performs noun lemmatisation. It can also remove short tokens that may carry meaning in particular notation or abbreviations. Domain-specific stopwords were not automatically removed, which avoids one source of information loss but leaves some generic academic terms in the feature space.

## 4. Topic labels are interpretations

NMF discovers components from the feature matrix; it does not know what a research topic is. Human-readable labels were assigned by looking at top terms. The labels in this report and charts are working labels, not ground-truth categories. Topic boundaries may overlap, and a topic such as “predictive and general ML” is broad by design because its top terms do not point to one narrow method family.

Independent human confirmation of the **final full-corpus model** has not yet been completed. No independent agreement statistic, including Cohen's kappa, is available. The report does not invent one.

There is also a model-ordering issue to keep clear. The existing 50-paper sample and its review template were prepared from the earlier validation model. The final full-corpus NMF model has a different topic ordering. The earlier paper sample can be useful for reviewing the validation model, but it is not a direct validation sample for the final chart labels. A new representative-paper sample should be extracted from the final model outputs before final labels are confirmed.

## 5. Stability is lexical, not semantic

The reported NMF stability score is based on top-15 word-set overlap after optimal topic matching. A score of 0.9195 at K=10 indicates strong overlap in the selected top terms across the tested runs. It does not prove that the models assign the same documents to topics, nor that the topics have identical meaning. Conversely, two semantically related topics may have different top words and receive a lower lexical score.

## 6. Validation was random, not chronological

The model-selection experiment used a deterministic random split of 8,000 sampled documents into 6,400 training and 1,600 validation records. The vectorisers were fitted on training data only, which prevents validation vocabulary and IDF leakage within that experiment. However, the split was not ordered by submission year. It therefore does not test whether a model trained on earlier years would generalise to later years.

The final NMF model was subsequently fitted on all eligible documents. The annual trend analysis is a retrospective description of the same corpus used to fit that final model. It is not a future-year prediction or an out-of-time validation result.

## 7. NMF activations are not probabilities

Normalised topic activations are relative weights within a document. They are normalised to sum to 100% for reporting convenience, but they are not calibrated probabilities that a paper belongs to a topic. The share assigned to one topic depends on the other topics and on the fitted model.

Dominant-topic share simplifies each paper to one topic and can hide secondary themes. Mean topic activation share preserves mixed-topic contributions, but it can also be affected by small activations across many topics. Reporting both measures reduces dependence on one definition, but does not remove the limitations of the model.

## 8. Relative shares are not absolute counts

The filtered corpus grows from 4,736 records in 2018 to 20,076 in 2024. A topic's share can fall while the number of papers associated with it still rises. The endpoint comparisons in the report are therefore expressed in percentage points and should not be described as a decline or increase in absolute publication volume unless a separate count-based analysis supports that claim.

## 9. Descriptive results are not causal results

The project calculates yearly topic shares and visualises their movement. It does not estimate causal effects, test research-policy interventions, or identify why a topic's share changed. Changes may reflect many things: research interest, naming conventions, arXiv submission behaviour, category assignment, dataset coverage, or shifts in abstract-writing practices.

No inferential statistical tests or uncertainty intervals were produced for the annual shares. Differences are descriptive comparisons, not claims of statistical significance.

## 10. Snapshot timing and versioning

The project records Kaggle dataset version 311 and a source-file checksum, but the exact publication date for that version was not confirmed in the recorded materials. The 2024 subset is therefore the set of 2024 first-submission records present in that snapshot, not a statement that every possible 2024 arXiv record was captured.

## 11. One record was not modelled

One of the 87,945 filtered records (`1902.02322`) had an empty token list after preprocessing. It is kept in the corpus metadata and annual paper count, but it cannot receive an NMF topic activation. This is a small coverage difference, but it is documented so that the denominator is clear.

## 12. Practical next steps

Before treating topic names as final, extract representative papers from the final full-corpus model, ask a second reviewer to assess the same material independently, document any label changes, and calculate inter-rater agreement only from actual independent ratings. For a stronger temporal test, repeat model selection with a chronological holdout or fit on earlier years and evaluate later years using a fixed vocabulary and topic mapping.
