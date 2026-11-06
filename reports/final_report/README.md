# Final report draft

The manuscript follows the pasted IEEEtran conference template. The compiled draft has seven main-content pages, one references page, and seven appendix pages. Appendix A holds team/workload and reproduction notes; Appendix B contains supplementary validation diagnostics; Appendix C reports TabPFN fine-tuning and support-sampling experiments; Appendix D compares small Transformers trained from scratch with TabPFN; Appendix E interprets TabPFN's advantage; Appendix F reports auxiliary temporal-data experiments with COE and stock inputs. The main text remains self-contained.

Read the [compiled report](final_report.pdf) or edit `final_report.tex` with its bundled figures and tables. The report package is self-contained for LaTeX compilation.

The original experiments include `experiment/tabpfn-anniversary-history` at `91599ab18029345de3f12203d0de1da9b528af68`. Main now preserves the TabPFN code under `workflows/tabpfn/` and exposes isolated runners through `run_tabpfn_best.py` and `run_tabpfn_ensemble.py`. Rebuilding the paper's evidence uses locally retained row predictions; these detailed outputs remain ignored, while final CSVs and training audit records are tracked.

Team name, member names/student IDs, and actual workload ownership intentionally remain placeholders, as requested. The report prioritizes chronological validation: 451.03/441.83 annual/recent RMSE for the tree/Transformer pipeline, and 440.95/440.77 recent RMSE for original/grouped-flat-model tree/TabPFN blends. Table III distinguishes pipeline comparisons from model-family baselines; Table V reports all eight exact grouped-ensemble members, standalone errors and final blend weights. Archived Kaggle values appear only as supplementary context in the appendix.

From the repository root:

```powershell
.venv/Scripts/python.exe reports/final_report/build_report.py
$env:TECTONIC_CACHE_DIR = (Resolve-Path .cache/report-tools/tex-cache).Path
Push-Location reports/final_report
$env:TECTONIC_CACHE_DIR = "$PWD/../../.cache/report-tools/tex-cache"
../../.cache/report-tools/tectonic.exe final_report.tex --outdir ../../output/pdf --keep-logs
Pop-Location
```

Alternatively compile `final_report.tex` with `latexmk -pdf`. IEEEtran and the listed LaTeX packages are required; the figures are vector PDFs generated with matplotlib. `evidence_audit.json` records checks and hashes. The builder does not train models, alter source data, or upload submissions.

Run `.venv/Scripts/python.exe reports/final_report/verify_pdf.py` after compilation to verify the eight-page main-content limit, check text/page boundaries, and render all pages to `tmp/pdfs/final_report/` for visual inspection. Rendering uses PyMuPDF because Poppler is not installed here. The source package can be compiled independently with its bundled figures and tables; rebuilding the evidence and figures requires the repository data, pandas, NumPy, and matplotlib.

The report builder does not train models. It re-scores recent saved predictions, verifies hashes and row alignment, and checks the recorded fresh F1 fixed-recipe validation results. Separate tree/TabPFN validation/full-data refits and submission hashes are in `review/checks/tabpfn-ensemble/`; the exact component calculations are exported to `ensemble_member_scores.csv`. Archived screenshot values remain distinct from the new, unscored Kaggle submission CSVs.

The supplied bibliography is preserved in `reference_pool.bib`; it is a source pool, not a list of implemented methods. The revised paper adds three selected references on Singapore automated valuation, tree/neural tabular benchmarks, and chronological forecast evaluation. `reference_notes.md` records the primary-source checks and one published-title correction.

Supplementary figures show group-deletion effects, cohort support, rent-band error concentration, and monthly effects of training-only filtering. The paired interval table uses 5,000 circular three-month-block resamples of the same ten validation months (seed 42), without refitting or selecting models. `supplementary_paired_intervals.csv` and `supplementary_rent_band_scores.csv` contain the numerical outputs. Intervals are conditional descriptive sensitivity measures and do not account for adaptive model selection.

`build_neural_appendix.py`, called by the main builder, regenerates the four neural appendix tables and `supplementary_neural_scores.csv` from saved predictions. It checks every prediction's unique IDs, full recent-window coverage, months, original labels and finite values, then independently recomputes RMSE against the archived scores. The evidence audit hashes all source predictions and settings. Fine-tuning differences use matched frozen inference views; scratch-model seed averages are not seed-ensemble scores. Original 77-input, ABCDE and later anniversary-feature TabPFN pipelines are explicitly distinguished.

`build_auxiliary_appendix.py` regenerates four auxiliary temporal-data tables and `supplementary_auxiliary_scores.csv`. It reconciles the direct-feature scores with saved monthly errors, recomputes national-mean forecast RMSE with equal target-month weight, reconstructs tree corrections from monthly residual moments, and checks TabPFN row IDs, dates, original labels and errors. Source hashes and the verification level for each comparison are recorded in `evidence_audit.json`. Monthly-updated and fixed-origin information scenarios, aggregate targets and individual-rental targets remain separate. Later-inspection follow-ups are labelled explicitly.
