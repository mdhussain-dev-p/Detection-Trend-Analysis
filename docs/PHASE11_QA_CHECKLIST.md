# Phase 11 quality-assurance checklist

This checklist separates completed work from checks that still need to be performed on the user's current Windows checkout.

## Documented and delivered

- [x] Replace the early-stage README with a description of the completed project.
- [x] Add a final project report with data, methods, validation results, findings, and caveats.
- [x] Add a standalone methodology document.
- [x] Add a standalone limitations document.
- [x] Add reproducibility instructions and recorded environment versions.
- [x] Add data-source, licensing, and responsible-use notes.
- [x] Update the architecture and data dictionary to reflect the implemented pipeline.
- [x] Add a protocol and a new template for reviewing the final model's ten topics.
- [x] Document that the older Phase 9 review file belongs to the validation model, not the final full-corpus model.
- [x] Update Git ignore rules so the three final charts and their supporting trend tables can be committed without exposing large raw data or model binaries.

## Checks to run on the project computer before public release

- [ ] Apply the documentation package to the project root and inspect the backup of any replaced documentation.
- [ ] Run `python -m pytest -q` in the project's virtual environment and save the fresh output.
- [ ] Confirm that the three PNG figures open from the exact paths linked in the README.
- [ ] Confirm that the two trend CSVs, annual count CSV, JSON summary, and methodology text are present.
- [ ] Confirm that `git status --short` does not show `data/raw/`, `data/processed/`, `data/features/`, `.venv/`, or `outputs/models/` being staged.
- [ ] Confirm that the three PNG files and the intended trend CSV/summary files are not ignored by Git.
- [ ] Review the final topic-label template and keep the labels marked provisional until final-model representative papers have been reviewed.
- [ ] Capture `python --version` and `python -m pip freeze` into a lock/environment file if exact environment reproduction is required.

## Research validation still pending

- [ ] Extract a representative-paper sample from the **final full-corpus model**.
- [ ] Obtain an independent second human rating of the same review set.
- [ ] Calculate and report agreement only after both independent ratings are available.
- [ ] Revise final topic labels and report text where the review warrants it.

## Release wording

Until the pending human review is complete, use wording such as:

> “Topic labels are provisional interpretations of the final NMF model's top terms. Independent human review of representative papers from the final full-corpus model remains future work.”

Do not describe the labels as independently validated, and do not invent a reviewer-agreement score.
