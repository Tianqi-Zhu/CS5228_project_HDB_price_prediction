[Chinese version](README.zh-CN.md)

# Main experiment code

| Content | Location |
|---|---|
| Active experiments | experiment/main/, indexed by experiment-card ID |
| Original architecture | Preserve original paths under baselines/, feature_engineering/, model_comparison/, optimization/, reporting/ and submissions/. Main retains selected entry points and their static dependency closure. |
| Dependency classification | review/source_partition.csv: entry points/tests, shared runtime dependencies and appendix-only modules |
| Important distinction | sixth_wave and older trainers may be runtime dependencies of F1; this does not make them independent main experiments. Appendix retains complete original code for reproduction. |
| Fixed F1 training snapshot | reproduction/f1/runtime/, isolated from root grouping modules |
