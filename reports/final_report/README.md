# Final report draft

The manuscript follows the pasted IEEEtran conference template. The compiled draft has seven main-content pages, one references page, and two appendix pages. Appendix A holds team/workload and reproduction notes; Appendix B is one page of supplementary validation diagnostics. The main text remains self-contained.

Read the [compiled report](final_report.pdf) or edit `final_report.tex` with its bundled figures and tables. The report package is self-contained for LaTeX compilation.

The underlying experiment code and results are on `experiment/tabpfn-anniversary-history`, including commit `91599ab18029345de3f12203d0de1da9b528af68`. The figure/evidence builder expects that branch's data layout and local prediction artifacts; some original TabPFN predictions remain excluded by the existing ignore rules. Publishing the paper on `main` does not import the experimental training code into main's separate workflow.

Team name, member names/student IDs, and actual workload ownership intentionally remain placeholders, as requested. The report prioritizes chronological validation: 451.03/441.83 annual/recent RMSE for the grouped tree/Transformer pipeline and 440.95 recent RMSE for the separate equal tree/TabPFN blend. Archived Kaggle values appear only as supplementary context in the appendix.

From the repository root:

```powershell
.venv/Scripts/python.exe reports/final_report/build_report.py
Push-Location reports/final_report
../../.cache/report-tools/tectonic.exe final_report.tex --outdir ../../output/pdf --keep-logs
Pop-Location
```

Alternatively compile `final_report.tex` with `latexmk -pdf`. IEEEtran and the listed LaTeX packages are required; the figures are vector PDFs generated with matplotlib. `evidence_audit.json` records checks and hashes. The builder does not train models, alter source data, or upload submissions.

Run `.venv/Scripts/python.exe reports/final_report/verify_pdf.py` after compilation to verify the eight-page main-content limit, check text/page boundaries, and render all pages to `tmp/pdfs/final_report/` for visual inspection. Rendering uses PyMuPDF because Poppler is not installed here. The source package can be compiled independently with its bundled figures and tables; rebuilding the evidence and figures requires the repository data, pandas, NumPy, and matplotlib.

The original root `output/` training artifacts are absent from this checkout. Archived submitted-model values come from retained experiment tables and the selection record; the report makes this limit explicit. Available recent tree/TabPFN predictions are checked against original labels and independently scored.

The supplied bibliography is preserved in `reference_pool.bib`; it is a source pool, not a list of implemented methods. The revised paper adds three selected references on Singapore automated valuation, tree/neural tabular benchmarks, and chronological forecast evaluation. `reference_notes.md` records the primary-source checks and one published-title correction.

Supplementary figures show group-deletion effects, cohort support, rent-band error concentration, and monthly effects of training-only filtering. The paired interval table uses 5,000 circular three-month-block resamples of the same ten validation months (seed 42), without refitting or selecting models. `supplementary_paired_intervals.csv` and `supplementary_rent_band_scores.csv` contain the numerical outputs. Intervals are conditional descriptive sensitivity measures and do not account for adaptive model selection.
