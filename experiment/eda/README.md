[Chinese version](README.zh-CN.md)

# EDA directory

This directory collects exploratory analysis for the midterm report's first part. Prefer citing
[`report.md`](report.md), `figures/`, `tables/` and [`summary.json`](summary.json).

## Directory structure

```text
experiment/eda/
├── report.md                 # Integrated EDA findings for the midterm
├── notes.md                  # Detailed analysis notes
├── progress_report.pdf       # 15-page progress report
├── summary.json              # Machine-readable integrated EDA summary
├── figures/                  # 11 main EDA figures
├── tables/                   # 18 statistical tables
└── supplementary/
    ├── train_test/           # Main-data audit,five compact plots and summary
    └── auxiliary/            # Six auxiliary-file audits,three plots and summary
```

## Two statistical definitions

Integrated and supplementary EDA answer different questions; retain both explicitly labelled definitions:

- Integrated EDA joins facility distances to every rental record for descriptive rent associations; primary schools have `MAINLEVEL_CODE` starting with `primary`.
- The auxiliary analysis measures distances per unique HDB building to audit facility coverage; primary schools include both `primary` and mixed-level `p1` schools.

School counts and distance distributions in the two summaries are therefore not interchangeable. The main report uses the integrated EDA
definitions; the supplementary report provides building coverage and per-file quality checks.

## Regeneration

```powershell
conda activate cs5228
python -m analysis.eda.run
```

The same entry point is available through `python -m analysis.main eda`. The script reads
raw CSV files in `data/` and checks source files before and after running; all outputs go to this directory.
