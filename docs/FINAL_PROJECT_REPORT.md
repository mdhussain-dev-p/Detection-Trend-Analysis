# Final Project Report

## Topic Detection and Trend Analysis in arXiv Machine-Learning Abstracts, 2018–2024

### Executive summary

This project examines how the mix of research topics represented in arXiv abstracts under the primary category `cs.LG` changed between 2018 and 2024. The working corpus contains 87,945 papers. After text cleaning and vectorisation, 87,944 records had usable features for the final topic model; one record was retained in the metadata but left unassigned because its cleaned token list was empty.

The modelling work compared Non-negative Matrix Factorisation (NMF) and Latent Dirichlet Allocation (LDA), and tested four topic counts: 5, 8, 10, and 12. A leakage-free validation experiment fitted its vectorisers on training documents only. Among the tested NMF configurations, K=10 gave the strongest observed top-term stability (mean matched Jaccard similarity 0.9195) while also producing a useful level of topic detail and a c_v coherence score of 0.5816. K=12 had a slightly higher NMF coherence score, but lower lexical stability. K=10 was therefore selected as a practical balance rather than because one score alone declared it the winner.

The final NMF model was then fitted to all 87,944 records with usable features. It converged in 142 iterations, with a training reconstruction error of 285.6715. The ten topics were used as a common reference for yearly trend analysis. The clearest movement is the increase in the large-language-model topic: its mean normalised activation share rose from 6.71% in 2018 to 17.99% in 2024. Predictive and general machine learning remained the largest topic by mean activation share in every year from 2019 onward. Deep neural networks declined as a proportion of the modelled topic mixture over the same period.

These results describe relative patterns in one arXiv category. They are not forecasts, causal findings, or a complete account of machine-learning research. Topic labels remain provisional pending a separate review of representative papers from the final full-corpus model.

## 1. Background and research question

Machine-learning research covers a wide range of methods and applications. The terminology used in abstracts changes as new methods become established, older methods are adapted, and areas of application develop. Reading thousands of abstracts individually is possible only with substantial effort, so this project uses unsupervised topic modelling to organise recurring terms into broad themes and then follows those themes over time.

The main research question is:

> **What research themes can be identified in arXiv papers whose primary category is `cs.LG`, and how does the relative share of those themes change between 2018 and 2024?**

The analysis addresses three supporting questions:

1. Can NMF and LDA produce interpretable topic structures from the abstracts?
2. How sensitive are the NMF topics to the selected number of topics and to random initialisation?
3. Which topics gain or lose relative share across the seven first-submission years?

The scope is intentionally narrow. The corpus contains arXiv metadata records assigned primarily to `cs.LG`; it does not cover all arXiv categories, journals, conferences, or machine-learning publications outside arXiv.

## 2. Data and corpus construction

### 2.1 Source

The source was the Cornell University arXiv metadata snapshot distributed through Kaggle:

<https://www.kaggle.com/datasets/Cornell-University/arxiv>

The project records dataset version 311. The source file was approximately 5.67 GB, and its SHA-256 digest was recorded as:

```text
97a7fee00eac38ee5ea92d5cddae351cf3cbb444775a397ac60efb773f217b10
```

The exact publication date associated with the selected Kaggle version was not confirmed in the project record, so it is not supplied here.

### 2.2 Inclusion rules

A record was retained when:

- its primary category was exactly `cs.LG`; and
- its initial submission year was between 2018 and 2024, inclusive.

The year field represents the first-submission date recorded for the paper. It is not the latest update date. This distinction matters because a paper may be revised years after it first appears.

The filtered JSONL dataset contains 87,945 records. The exploratory checks found 87,945 unique IDs and no missing titles or abstracts. The annual record counts are:

| First-submission year | Papers |
|---:|---:|
| 2018 | 4,736 |
| 2019 | 8,840 |
| 2020 | 11,097 |
| 2021 | 13,087 |
| 2022 | 13,754 |
| 2023 | 16,355 |
| 2024 | 20,076 |
| **Total** | **87,945** |

The number of records in the selected corpus grew over the period. For that reason, the topic-share percentages in this report should be interpreted as changes in composition, not as a direct measure of the absolute number of papers in a field.

### 2.3 Record structure

The filtered records retain paper identifiers, titles, abstracts, author information, category metadata, the primary category, the first-submission date and year, and the latest update date. The preprocessed records add cleaned tokens and a token count. The full field definitions are in [`data_dictionary.md`](data_dictionary.md).

## 3. Exploratory analysis and text preparation

The exploratory stage examined record validity, category selection, publication-year counts, abstract length, and frequent terms. The average abstract length was 146.37 words, the median was 145, and the 90th percentile was 202 words. The maximum observed abstract length was 405 words. No missing titles or abstracts were found in the filtered records.

Abstract cleaning was applied record by record. The process:

1. unescaped HTML entities and converted text to lowercase;
2. extracted alphanumeric tokens;
3. removed single-character tokens and numeric-only tokens;
4. removed NLTK English stopwords; and
5. lemmatised tokens using WordNet's noun lemmatiser.

Domain-specific terms were reviewed as candidates but were not automatically removed. This was a conservative choice: terms that appear frequently in machine-learning abstracts can still carry meaning for the topic model. The original abstract was retained alongside the cleaned tokens so that a topic could be checked against the source text later.

The preprocessing stage processed all 87,945 records and produced 9,997,305 tokens. The mean cleaned-token count was 113.68 per abstract, the median was 113, and the vocabulary contained 85,573 unique tokens before vectoriser filtering.

One record (`1902.02322`) had no tokens remaining after cleaning. It was retained in the metadata, but excluded from NMF because a zero-feature document cannot contribute to a topic activation. This was recorded as one unassigned document, rather than deleting the record from the corpus count.

## 4. Feature construction

Two sparse feature representations were prepared:

- **TF–IDF**, for NMF;
- **document-term counts**, for LDA.

The vectorisers were configured to retain up to 8,000 features, with a minimum document frequency of five. The TF–IDF configuration used sublinear term frequency. Both full-corpus matrices had 87,945 rows and 8,000 columns. Sparse storage was used to avoid unnecessarily expanding the mostly-zero document-term matrices into dense arrays.

The full-corpus feature space is used for the final descriptive model. Model-selection validation was handled separately, with fresh vectoriser instances fitted only on the training subset.

## 5. Model comparison and validation

### 5.1 Validation design

A deterministic sample of 8,000 documents was selected for model comparison. It was split into 6,400 training documents and 1,600 validation documents, using split seed 2026. The TF–IDF and count vectorisers were cloned from the agreed configurations, then fitted on training documents only. The validation documents were transformed using those training-fitted vectorisers. This prevents validation documents from influencing the vocabulary or inverse-document-frequency weights in the validation experiment.

The training-only TF–IDF vocabulary contained 6,257 terms. Coherence was calculated using up to 4,000 non-empty training documents. This was a pilot-sample validation experiment; it was not a chronological holdout and it did not test whether a model trained on earlier years could predict later years.

### 5.2 Candidate topic counts

The candidate topic counts were 5, 8, 10, and 12. NMF was run with five random seeds (42, 13, 21, 34, and 55) at each topic count, using random initialisation. LDA was run once per topic count with seed 42. The validation experiment comprised 24 model runs in total.

NMF was assessed using validation reconstruction error, c_v coherence, and lexical stability. Lexical stability was the mean pairwise Jaccard overlap of each model's top 15 terms after optimal topic matching with the Hungarian algorithm. It measures overlap between top-word lists; it does not prove that two runs have identical topic meaning.

LDA was assessed using held-out perplexity and c_v coherence. LDA perplexity and NMF reconstruction error are different quantities and should not be compared numerically against one another.

### 5.3 Validation results

| Topics (K) | NMF validation error (mean) | NMF top-15 lexical stability (mean) | NMF c_v coherence | LDA held-out perplexity | LDA c_v coherence |
|---:|---:|---:|---:|---:|---:|
| 5 | 38.7803 | 0.8288 | 0.5687 | 1,925.2945 | 0.4039 |
| 8 | 38.5987 | 0.7613 | 0.5788 | 2,027.7662 | 0.4178 |
| 10 | 38.4972 | **0.9195** | 0.5816 | 2,099.7763 | 0.4400 |
| 12 | **38.4154** | 0.7672 | **0.5906** | 2,120.5340 | **0.4573** |

K=12 had the lowest NMF validation reconstruction error and the highest NMF coherence among the tested values. However, K=10 had the highest mean lexical stability and slightly lower but still strong coherence. The ten-topic solution was selected as a balance between repeatability and a manageable level of detail. This is a modelling decision, not proof that ten is the uniquely correct number of topics.

Some K=5 NMF runs reached the configured maximum iteration limit during validation. That behaviour is one reason to consider convergence and stability alongside error and coherence scores rather than ranking topic counts on a single metric.

## 6. Final full-corpus topic model

After choosing K=10, the final model was trained on all 87,944 documents with non-zero feature rows. The final NMF used random initialisation, seed 42, and a maximum of 300 iterations. It converged in 142 iterations. Its training reconstruction error was 285.6715, and no convergence warning was recorded for this final fit.

The final model output includes the fitted NMF model, document-topic weights, normalised document-topic proportions, a modelled-document mask, corpus metadata, per-topic terms, and dominant-topic assignments. Topic numbers in the trend analysis refer to this final full-corpus model.

### 6.1 Working topic interpretations

The labels below were assigned by inspecting the final model's top terms. They remain working labels pending review against representative abstracts from this final model.

| Topic | Working label | Selected top terms |
|---:|---|---|
| 1 | Supervised learning | data, distribution, label, sample, class, supervised, classification |
| 2 | Adversarial robustness | adversarial, attack, robustness, perturbation, defense, robust |
| 3 | Predictive and general ML | data, prediction, machine, time, system, series, application |
| 4 | Bandits and online optimisation | bound, regret, algorithm, bandit, optimal, online |
| 5 | Reinforcement learning | policy, agent, reinforcement, environment, reward, action |
| 6 | Gradient-based optimisation | gradient, optimisation, convergence, descent, stochastic, convex |
| 7 | Deep neural networks | network, neural, deep, layer, architecture, convolutional |
| 8 | Federated learning | federated, client, privacy, communication, distributed, server |
| 9 | Graph neural networks | graph, node, GNN, representation, edge, embedding |
| 10 | Large language models | language, model, LLM, transformer, fine-tuning, code |

These labels summarise word patterns; they should not be read as mutually exclusive research disciplines. For example, a paper about graph neural networks may also activate the deep-learning topic.

## 7. Measuring topic trends

The same final full-corpus model was used for every year. This is important: topic numbers are comparable across years because they come from one common model, rather than from separately trained yearly models whose topic numbers could mean different things.

Each document's topic activations were normalised so that the ten topic shares summed to 100%. These values are relative weights within the document, not calibrated probabilities.

Two annual measures were calculated:

**Dominant-topic share.** For a given year, this is the percentage of modelled papers whose largest normalised topic activation belongs to a particular topic. Each modelled paper contributes to one topic in this measure.

**Mean topic activation share.** For a given year, this is the average normalised activation of a topic across all modelled papers in that year. A paper can contribute to several topics. The ten mean shares sum to 100% for each year, subject to rounding.

The single unassigned paper was retained in the yearly corpus counts but excluded from topic-share denominators. There were 8,839 modelled papers out of 8,840 in 2019; all other years had no unassigned papers.

## 8. Findings

### 8.1 Large language models gained the most share

Topic 10, labelled large language models, had the largest increase between the endpoints. Mean activation share rose from 6.71% in 2018 to 17.99% in 2024, an increase of 11.28 percentage points. Its dominant-topic share rose from 3.27% to 18.98%, a gain of 15.71 percentage points.

This is the clearest shift in the modelled topic mixture. It is consistent with the increased presence of language-model terminology in the selected abstracts, but the analysis does not identify the cause of that change.

### 8.2 Predictive and general ML remained prominent

Topic 7 (deep neural networks) had the largest mean activation share in 2018, at 18.41%. From 2019 through 2024, Topic 3 was the largest mean-share topic, rising from 17.23% in 2019 to 21.30% in 2024. Its dominant-topic share reached 26.60% in 2024.

The topic is broad. Its terms include prediction, machine learning, time, systems, and applications. It should not be interpreted as a sharply bounded subfield.

### 8.3 Deep neural networks declined as a relative share

Topic 7's mean activation share fell from 18.41% in 2018 to 10.50% in 2024, a decrease of 7.90 percentage points. Its dominant-topic share moved from 22.53% to 10.54%. This is a relative decline inside the selected topic mixture; because the corpus itself grew, the percentages alone do not show that the absolute number of papers on neural networks fell.

### 8.4 Federated and graph learning gained share

Federated learning increased from 3.07% to 6.79% mean activation share, while graph neural networks increased from 4.84% to 7.44%. Both also gained dominant-topic share. The model therefore indicates that these themes occupied a larger relative part of the selected `cs.LG` abstracts in 2024 than in 2018.

### 8.5 Endpoint comparison for all topics

The following table compares the two endpoint years. Changes are expressed in percentage points (pp), not percentage growth rates.

| Topic | Mean share 2018 | Mean share 2024 | Change (pp) | Dominant share 2018 | Dominant share 2024 |
|---|---:|---:|---:|---:|---:|
| 1. Supervised learning | 12.06% | 9.30% | −2.75 | 9.38% | 6.17% |
| 2. Adversarial robustness | 6.79% | 4.78% | −2.01 | 7.03% | 3.73% |
| 3. Predictive and general ML | 16.91% | 21.30% | +4.39 | 20.19% | 26.60% |
| 4. Bandits and online optimisation | 9.60% | 6.50% | −3.10 | 7.96% | 4.93% |
| 5. Reinforcement learning | 8.52% | 6.71% | −1.81 | 10.60% | 7.48% |
| 6. Gradient-based optimisation | 13.09% | 8.69% | −4.40 | 13.07% | 6.57% |
| 7. Deep neural networks | 18.41% | 10.50% | −7.90 | 22.53% | 10.54% |
| 8. Federated learning | 3.07% | 6.79% | +3.71 | 1.63% | 6.69% |
| 9. Graph neural networks | 4.84% | 7.44% | +2.60 | 4.35% | 8.30% |
| 10. Large language models | 6.71% | 17.99% | **+11.28** | 3.27% | 18.98% |

The complete annual values are available in `outputs/reports/topic_trends_2018_2024/yearly_topic_trends_long.csv` and the wide-format companion CSV.

## 9. Quality checks and deliverables

The project includes automated tests for ingestion, preprocessing, feature construction, model validation, topic stability, and trend calculations. The Phase 10B log records 46 passing tests in 6.17 seconds. The chart-generation stage separately checked that the annual trend files contained seven years and 70 topic-year observations, confirmed that the two topic-share measures summed to 100% for every year within the configured tolerance, compiled the chart script, and generated all three PNG files.

The principal trend outputs are:

- `yearly_topic_trends_long.csv` — one row per year and topic (70 rows);
- `yearly_topic_trends_wide.csv` — one row per year with all topic measures (7 rows);
- `yearly_corpus_counts.csv` — annual corpus and modelled-paper counts;
- `topic_trend_summary.json` — machine-readable trend summary;
- `topic_trend_methodology.txt` — concise calculation notes;
- three PNG figures — stacked composition, dominant-topic share, and mean activation share.

The final model and document-topic arrays are stored separately under `outputs/models/final_full_corpus_nmf_k10/` and are intentionally not included in the documentation package.

## 10. Limitations

The results are shaped by the scope and design of the data and model. They cover one arXiv primary category, use abstracts rather than full papers, and depend on a bag-of-words representation and one fitted topic model. Topic labels are interpretive, not ground truth. The random validation split does not test future-year generalisation, and the final annual trends are descriptive. The NMF activation values are not probabilities. Independent human confirmation for the final model's topic labels has not yet been completed.

A fuller account is provided in [`LIMITATIONS.md`](LIMITATIONS.md).

## 11. Conclusion

The final model provides a useful high-level view of how the selected arXiv `cs.LG` abstract corpus changed from 2018 to 2024. The most visible pattern is the growth in the large-language-model topic, alongside continued prominence of broader predictive and general machine-learning work. Federated learning and graph neural networks also occupy a larger relative share in 2024, while deep neural network and several optimisation-related topics occupy smaller shares of the modelled mixture.

The project should be treated as a reproducible exploratory analysis rather than a final map of machine-learning research. The next research-quality step is to review representative papers from the **final full-corpus model** with an independent human rater, update the topic labels where necessary, and record inter-rater agreement without inventing a score where no second rating exists.
