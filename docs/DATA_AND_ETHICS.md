# Data source, licensing, and responsible use

## Source and scope

The project uses the Cornell University arXiv metadata snapshot distributed through Kaggle:

<https://www.kaggle.com/datasets/Cornell-University/arxiv>

The project record identifies the selected snapshot as version 311. The source metadata file was approximately 5.67 GB, with SHA-256 digest:

```text
97a7fee00eac38ee5ea92d5cddae351cf3cbb444775a397ac60efb773f217b10
```

Only records whose primary category is exactly `cs.LG` and whose initial submission year is 2018–2024 are used in the analysis.

## Licence note

The Kaggle dataset page identifies the arXiv metadata dataset as CC0 1.0. This project uses the metadata and abstracts supplied in that dataset. That licence statement should not be extended to full paper PDFs or to other content linked from arXiv; individual papers may be subject to their own licences and terms. The project does not download, bundle, or redistribute full papers.

Anyone redistributing this project should check the dataset page and applicable licence information again at the time of publication, and should preserve the source attribution and snapshot provenance.

## Data minimisation

The analysis needs paper identifiers, titles, abstracts, category information, and submission dates. It does not need author contact details or personal profiles. The project does not attempt to infer sensitive personal attributes about authors.

The raw 5.67 GB source snapshot, filtered corpus, feature matrices, and fitted model binaries are kept outside the repository. The public-facing project should include code, documentation, selected aggregate outputs, and appropriately limited review material rather than the entire raw dataset.

## Interpretation and communication

Topic labels are human interpretations of model terms. They should be described as provisional until reviewed against representative abstracts. The yearly percentages describe the composition of this selected corpus; they do not measure research quality, societal importance, funding, or the number of publications across all venues.

The analysis is exploratory and descriptive. It should not be used to rank researchers, institutions, or individual papers, nor to make decisions about people.
