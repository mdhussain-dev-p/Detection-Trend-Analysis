# Data dictionary

This project stores the filtered corpus as JSON Lines: one JSON object per paper. Preprocessing adds token fields while retaining the original abstract and metadata.

## Filtered paper record

| Field | Type | Meaning |
|---|---|---|
| `id` | string | arXiv identifier |
| `title` | string | Paper title |
| `abstract` | string | Original abstract text, with whitespace normalised during extraction |
| `authors` | string or source-preserved text | Author information from the metadata record |
| `categories` | string | Full category list from the source |
| `primary_category` | string | Primary category used for the project filter; `cs.LG` for included rows |
| `submitted_date` | date-like string | First-submission date used to assign the paper to a year |
| `submission_year` | integer | Year extracted from `submitted_date` |
| `update_date` | date-like string or null | Latest update date supplied in the source metadata |

## Preprocessed record additions

| Field | Type | Meaning |
|---|---|---|
| `tokens` | array of strings | Cleaned abstract tokens used by vectorisation |
| `token_count` | integer | Number of cleaned tokens in the record |

A record with `tokens=[]` remains in the preprocessed metadata but has no usable TF–IDF feature row and is excluded from NMF fitting. In this corpus, one record (`1902.02322`) has an empty token list.

## Annual trend tables

`yearly_topic_trends_long.csv` contains one row for each year/topic combination (70 rows total):

| Field | Meaning |
|---|---|
| `year` | First-submission year, 2018–2024 |
| `total_papers_in_year` | All filtered records in the year |
| `modelled_papers_in_year` | Records with non-zero feature rows and topic activations |
| `unassigned_papers_in_year` | Records retained in the corpus but not modelled |
| `topic_number` | Topic index in the final full-corpus NMF K=10 model |
| `top_10_terms` | Top terms for the final model topic |
| `dominant_topic_papers` | Modelled papers whose largest activation is this topic |
| `dominant_topic_share_pct` | Dominant-topic papers as a percentage of modelled papers in that year |
| `mean_topic_activation_share_pct` | Mean normalised activation share for this topic in the year |

`yearly_topic_trends_wide.csv` stores the same topic-share measures with one row per year. `yearly_corpus_counts.csv` records total, modelled, unassigned, and coverage counts by year.
