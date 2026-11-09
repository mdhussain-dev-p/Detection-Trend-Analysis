# Topic review protocol

The final NMF topics were labelled by inspecting their top terms. Those names are working labels, not verified categories. This document sets out how to review them without confusing the final model with the earlier validation model.

## 1. Keep the two topic sets separate

The project has two different ten-topic outputs:

- The **validation model** was fitted on the training portion of the 8,000-document model-selection sample. The existing 50-paper file and `topic_label_review_template.csv` were created for that model.
- The **final full-corpus model** was fitted later on 87,944 documents with usable features. Its topics are the ones used in the yearly trend CSVs and charts.

The topic order changed between these models. The earlier review CSV is not a topic-label key for the final charts. Do not copy its topic numbers or labels into the final report.

A separate template for the final full-corpus model is provided at:

```text
docs/templates/final_model_topic_label_review_template.csv
```

## 2. What needs to be reviewed

For each final-model topic, review:

1. the top terms from the final model;
2. several representative abstracts selected from the final model's document-topic activations;
3. whether the provisional label captures the common theme;
4. whether the topic is distinct enough from the other nine topics; and
5. any terms or abstracts that appear inconsistent with the proposed label.

The final model artefacts include `document_topic_proportions.npy`, `document_topic_weights.npy`, `corpus_metadata.csv`, and `document_dominant_topics.csv`. Representative papers for the final topic numbering should be extracted from these final-model files. The existing Phase 9 validation-sample abstracts should not be used as if they were drawn from the final model.

## 3. Independent reviewer procedure

For a meaningful inter-rater check:

1. Prepare one fixed review set from the final model. Include the same paper IDs, titles, abstracts, and topic codebook for both reviewers.
2. Give each reviewer the same written definitions and instructions. Do not show one reviewer's decisions to the other.
3. Ask each reviewer to assign the best-fitting topic to each abstract, or select `Other/unclear` when none fits. Ask them to note uncertainty and suggest a better label where needed.
4. Collect the two completed files before discussing disagreements.
5. Calculate Cohen's kappa from the two independent categorical assignments. Keep the raw anonymised ratings and the script or calculation used.
6. Review disagreements, revise topic descriptions where justified, and document any label changes.

Cohen's kappa measures agreement beyond chance under its assumptions. It does not prove that the model is correct, and its value can be affected by class prevalence and the chosen sample. The review sample is purposively selected to include high-activation examples, so its agreement score should not be presented as an estimate of accuracy on every paper in the corpus.

## 4. Do not invent human validation

Until a second person has independently rated the same final-model review set, the project must state that independent human review is pending. An AI-generated interpretation, the project author's own second pass, or two passes by the same person do not count as independent inter-rater agreement. Do not report a Cohen's kappa value until two real sets of independent ratings exist.

## 5. Review outcome to record

Record:

- date of review;
- version of the final model and its topic-term file;
- number of abstracts reviewed;
- reviewer identifiers in an anonymised form;
- final label selected for each topic;
- inter-rater statistic and confidence interval, if calculated with a suitable method;
- disagreements and any subsequent label changes; and
- remaining ambiguous topics.

The template starts with the ten working labels used in the charts. Reviewers should be free to revise or reject them.
