[Chinese version](README.zh-CN.md)

# EDA directory

This directory holds frozen exploratory evidence. The main workflow should prioritize
[`report.md`](report.md), `figures/`, `tables/` and [`summary.json`](summary.json).

## Directory structure

```text
experiment/main/M01/eda/
├── report.md                 # Integrated EDA findings for the midterm
├── notes.md                  # Detailed analysis notes
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
raw CSVs and check source files before and after execution. New outputs default to ignored `output/eda/`, leaving this frozen evidence intact.

The old `progress_report.pdf` was a 15-page interim presentation built from the same figures and has been removed from main. Run `python -m analysis.eda.pdf_report` to generate `output/reports/progress_report.pdf` from this directory; specify `--eda-dir output/eda` for new EDA. It is not the final eight-page scientific paper.
