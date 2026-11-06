# Reference selection and checks

The full user-supplied pool is preserved, with only trailing blank lines normalized, in `reference_pool.bib`. Inclusion in that pool does not establish implementation, provenance of the competition data, or predictive performance of this project. The manuscript retains its seven algorithm references and adds the three relevant items below.

| Manuscript key | Primary source checked | Use in this report |
| --- | --- | --- |
| `housing` | Original article abstract and metadata at https://pubmed.ncbi.nlm.nih.gov/38624904/ ; DOI 10.1007/s11146-021-09861-1 | Singapore housing valuation precedent for boosted trees. The paper studies sale prices; no published performance number is reused as our rental result. |
| `treebenchmark` | Published NeurIPS page: https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html | Motivation for strong tree baselines alongside neural models. The published title includes **typical tabular data**, whereas the supplied pool uses the shorter arXiv title. The manuscript uses the published title. |
| `forecasting` | Authors' book, section 5.10: https://otexts.com/fpp3/toolbox.html#time-series-cross-validation | Earlier-information evaluation and the distinction between fitted residuals and genuine forecasts. No claim that the report's reused fixed-origin windows form an untouched rolling-origin test. |

Checked on November 6, 2026. The supplied pool also contains references to methods, imagery, causal estimation, and external data that are not part of the main experiment. Those entries are not added as citations simply because they are available.

All appendix graphs are newly drawn from local project tables or saved predictions; none reproduces a published figure. Figure 4's deletion panel uses the archived fixed-model, three-annual-window table. Figure 4's support panel uses the saved TabPFN cohort coverage table. Figure 5 and Table VII use original validation labels and aligned saved predictions; all 28,995 rows are retained. The paired bootstrap preserves all cases within each sampled month and recomputes RMSE from sums of squared errors and counts.
