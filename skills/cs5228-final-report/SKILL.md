---
name: cs5228-final-report
description: Use when writing, editing or reviewing this CS5228 HDB price-prediction project's final report and submission materials. Check course requirements, research content, the eight-page limit, assessment criteria and reproducibility. Applies to this project's report work, not unrelated papers or ordinary code changes.
---

[Chinese version](SKILL.zh-CN.md)

# CS5228 final project report

Use course requirements as acceptance criteria for a scientific-paper-style final report and reproducible submission. Plan, write or review according to the current request. Reading or reviewing requirements alone does not authorize rewriting the whole report, running all experiments or submitting materials.

## Mandatory course requirements

- Submit a PDF with at most **eight main-text pages**, including main-text tables, figures and plots; references and appendices are excluded.
- The main text must stand alone and fully explain methods and key conclusions without relying on appendices.
- Use appendices sparingly for supplementary EDA, evaluation figures/tables and team contributions.
- Include team name and all members' names/student IDs; provide a contribution overview in the appendix, optionally as a table or Gantt chart.
- Section structure is flexible but must cover the four core areas below.
- Submit the report PDF and source-code archive through Canvas; a GitHub repository link may replace the code archive.
- The original deadline is a placeholder; the actual date is unknown. Do not infer a date or treat the placeholder as valid.

## Core content and writing evidence

### Motivation

State the research question, goals, practical value, beneficiaries and challenges. Ground motivation in actual HDB price-prediction uses, such as budgeting or factor analysis, as optional directions consistent with completed work. Kaggle ranking alone is insufficient motivation.

### Exploratory data analysis and preprocessing

Explain data understanding and how EDA supports preprocessing. Cover applicable missing-value handling, categorical processing, reduction, transformations, outliers and feature generation, grounded in actual data; using every technique is unnecessary. Connect observations → treatment choices → rationale.

### Data-mining methods

Explain methods, suitability, actual implementation and hyperparameter selection/tuning. Discuss reasonable alternatives and key design decisions. Avoid reteaching common algorithms; cite relevant literature for advanced methods or models.

### Evaluation and interpretation

Use task-appropriate settings and metrics, compare methods and explain which work best and why. Analyze failures, common errors, fundamental limitations and future improvements. Distinguish supported conclusions from untested explanations.

## Assessment criteria

| Dimension | Weight | Review focus |
| --- | --- | --- |
| Method quality | 60% | EDA-supported preprocessing; correct missing/categorical handling; appropriate figures; sound method selection/implementation; explained alternatives/designs; valid experiments/metrics; errors and limitations |
| Report quality | 30% | Clear, concise, complete methods/results; appropriate related-work citations; main text within eight pages |
| Reproducibility | 10% | Complete, organized, readable, documented code enabling external readers to trace steps and reproduce results |

## Recommendations for the HDB project

These are project recommendations for meeting assessment criteria, not additional mandatory course rules.

- Inspect code, configurations, logs, figures and results before writing so the report matches implementation.
- Justify splits through the prediction scenario; inspect leakage through time, duplicates, target-derived features and preprocessing fitting.
- Compare models under comparable splits/metrics and distinguish training, validation, tuning and final testing.
- Where data allow, analyze errors by flat type, region, price range or transaction time; select groups that explain the issue.
- Record dependencies, data sources/acquisition, processing, commands, configuration, seeds and output paths needed for reproduction.
- Keep figures supporting main arguments in the body and detailed supplements in appendices; do not hide essential methods there.

## Workflow and delivery checks

1. Inspect available materials and gaps for the current request. Mark missing team details, contributions, results or sources explicitly; do not invent them.
2. Organize claims around real evidence, prioritizing method-quality gaps before compression and polishing. Respect requests limited to local edits.
3. Prefer the project's existing final-PDF template. Course Word/LaTeX templates exist but their contents have not been read; do not infer fonts, columns, margins or other details.
4. Inspect the rendered final PDF for main-text page count, reference/appendix boundaries, figure readability, layout, citations and team details. Source files alone do not establish PDF acceptance.
5. Check that reproduction instructions let external readers obtain data, run code and verify main results. Clearly state checks not actually executed.
6. Reviews should identify specific issues, evidence locations and proposed edits, distinguishing unmet course requirements, methodological risks and optional improvements.

## Course template links

User-provided Canvas links may require login. Check for downloaded project templates first; if unavailable, state that template contents remain unverified.

- [Template page](https://canvas.nus.edu.sg/courses/93866/files/9565679?wrap=1)
- [Template download](https://canvas.nus.edu.sg/courses/93866/files/9565679/download?download_frd=1)
