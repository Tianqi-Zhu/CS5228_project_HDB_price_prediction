[Chinese version](main-appendix-experiment-plan.zh-CN.md)

> Follow the [main requirement-card status](main-execution-status.md) and [five improvements](main-five-item-completion.md) for current execution. Below are historical designs and acceptance criteria. Offline audits and fixed-recipe F1 Y/J refits are complete; F2 dependencies remain outstanding.

# A. Main branch plan

This table preserves original requirement cards and acceptance goals. See [current status](main-execution-status.md) for completion: F2 candidate components and recent J pairing are verifiable; annual Y, the complete ordered dictionary and screenshot association remain outstanding.

| Planning boundary/shared definition | Current value | Status/constraint |
|---|---|---|
| Current stage | Execute feasible parts of new main cards and audit the repository | Follow the [status table](main-execution-status.md); do not fabricate F2 dependencies; push normally after verification; do not draft the report body |
| main / appendix | Experiment ownership is implemented in two physical branches; preserve others' integrated work and extend current main | The user authorized a normal verified push, without force-overwriting the remote |
| Project task | Predict individual HDB `MONTHLY_RENT` in SGD; 150,000 labels and 50,000 queries | Raw hashes, dates and counts are verified; use the [supplied-snapshot/fixed-origin scenario](../methods/prediction-scenario.md) |
| Six stages | S1 EDA/preprocessing; S2 classical models; S3 features; S4 capability/bottlenecks; S5 complex models; S6 two final results | Nodes anchor experiments; controlled comparisons relate stages without claiming causal housing-feature effects |
| Final F1 | Grouped blending + PLR-FT, `kaggle_structure_blend.csv` | Retain one of your three variants; Y451.0295/J441.8345; displayed Kaggle 415.37529; method/CSV evidence exists |
| Final F2 | `submission_zch_tabpfn87_equal_150k.csv` | Retain as requested; screenshot 413.19635; 87 inputs, members, formula, actual training count and Y/J results: [to be supplied] |
| Common Y window | Fit through 2024-03; validate 2024-04 through 2025-03 | 114,842/35,158 rows; retain every validation row |
| Common J window | Fit through 2024-05; validate 2024-06 through 2025-03 | 121,005/28,995 rows; overlaps Y and must not be pooled as an independent test |
| Metrics | SGD RMSE primary; MAE, mean predicted-minus-actual bias and monthly/group counts secondary | Pool RMSE through combined SSE before square root; separate targets/protocols into distinct tables |
| Randomness | Metric recomputation trains nothing; sampling/block resampling seed 42; reused training retains original seeds | Prespecify42/43/44 only for new matched refits; changing seeds cannot reuse old scores |
| Temporal constraints | Fitting histories use earlier months; query histories freeze at the origin; transforms learn from fitting data only | Separately disclose380 March2025 full-data queries; distinguish supplied future macro observations from prospective fixed-origin forecasts |
| Data boundary | F1 uses existing train and supplied aux; actual F2 sources: [to be supplied] | Exclude external target-month rents from prediction experiments; identify other external data by source/release date |
| Assessment constraints | Course rubric: methods 60%, report 30%, reproduction 10% | Do not invent sub-scores, exact deductions or separate novelty/experiment-count weights. Eventual PDF body≤8 pages including figures/tables, excluding references/appendices; self-contained body, limited appendices, team/workload and reproducible source. This stage produces experiment materials only. |
| Compute/budget | F1 used an RTX4060 Laptop8GB; 15 base models trained for each Y/J window | See [five-item acceptance](main-five-item-completion.md) for measured end-to-end times. No paid API was used. Additional GPU hours, deadline and API budget: [to be supplied] |
| Evidence version | Reuse frozen snapshots from 13 branches and [existing branch evidence](tabpfn-branch-split.md) | Use commits in table C; subsequent branch changes require new run IDs/code hashes, without mixing old results |
| Assessment tags | M=method quality60%; Q=report quality30%; R=reproducibility10% | Tags guide experiment selection without adding assessment categories |

| Experiment ID | Narrative stage | Purpose | Branch use (reuse/modify/new) | Data and features | Models and controls | Key metrics | Supported stage-level conclusion | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|---|---|---|
| M00 | Shared S1/S6 gate | Freeze comparison protocols and result identities | New audit; reuse B11/B12 inventories and complete B05 | Raw hashes; Y/J row IDs; schema/target/as-of/macro/seed | Independently recompute predictions against original labels/settings; no trained model | Counts, mismatches, hashes, cutoff violations and recomputation error | Which scores may share a comparison table; M/R | P0 | Each record has source commit, data/feature/prediction hashes, target restoration and information policy. Zero misaligned/duplicate IDs or invalid cutoffs. Explicitly mark missing F2 evidence and block unsupported pairing conclusions. |
| M01 | S1 | Decide which data to retain, transform or flag | Reuse B12 EDA, B13 spatial and B09 cleaning audits; modify only identified inconsistencies | Raw main table, six aux sources, constants/aliases/duplicates/derived missingness | Raw versus normalized definitions; no rent deletion versus existing fitting-only cleaning controls | Coverage, missingness, duplicates, constants, invalid ages and date/rent distributions | Data justification for preprocessing; M | P0 | Recheck train 150,000/test 50,000 and raw duplicates425/5,508; retain all test rows; joins cannot multiply rows. Document every rule/evidence; do not automatically delete unverified low rents. MRT claims require snapshot wording or X08. |
| M02 | S2 | Establish classical references before adding histories | Reuse B12 `baseline_classical_models` and CatBoost C baseline; include recomputation in M00 | Y; seven C inputs: type/model/area/month index/month1–12/age/town; valid fit-only preprocessing | Mean, Ridge, RandomForest, LGB, Cat; saved RMSE 624.8788/493.7719/485.0928/487.2160 for the latter four, respectively | RMSE/MAE/bias, n and fixed hyperparameters/iterations | Linear/tree references using basic property/time/location inputs; M/Q | P0 | Confirm common C7/Y from `baseline_classical_models/results.json`; hash Cat C provenance; list same-Y mean separately. Do not attribute differences from later19/27-input configurations to one factor. Fill only missing cells. |
| M03 | S3 | Test incremental value of histories and anniversary cohorts | Reuse B12 fixed ladder, not another 27-input model's hyperparameters | Y/J; nested19→27→47→62 schemas | Fixed LGB353 trees/15 leaves/lr.03/L2=10; GPU Cat535 iterations/depth6/lr.04/L2=10; four levels each | RMSE/MAE/bias per level, paired deltas and history coverage | Which information to retain and consistency across learners/windows; M | P0 | Align settings/queries for 16 existing fits; do not update validation histories. Compute feature deltas within tables only. Mixed gains are acceptable; every level need not win. |
| M04 | S4 | Locate residual errors and distinguish coverage, drift and indistinguishable inputs | New offline diagnostics; reuse B12 predictions and B09 definitions, not B09 error values | M03 representative and F1 predictions on identical Y/J; month/type/24-/36-month support; actual-rent groups are post-hoc only | Representative62-input model versus F1; full data versus prespecified groups with training-derived thresholds | Group n/RMSE/bias/SSE share, monthly errors and within-identical-input diagnostics | Which errors warrant complex models; local conflicts do not prove a global error floor; M | P0 | Groups cover all rows or disclose overlap; reconcile SSE. Group inputs must be known at deployment, with actual-rent diagnostics separately labelled. No mandatory error-floor target. |
| M05 | S5 | Test complex-model and blending increments | Reuse B12 common27-input table and round 10, completing metadata if needed; B04 gives F2 context | Separate comparison blocks: common27/Y and common62 neural models/Y/J; distinguish tree/network schemas | Block1: same27 trees versus EmbeddingMLP/FT. Block2: PLE128/3 versus PLR128/3, plus fixed round 9 blends with no addition/PLE/PLR | Standalone/full-blend RMSE/MAE/bias and monthly deltas | Architectural benefit under matched inputs versus blend complementarity; M/Q | P0 | Check block1 budgets against common-input PROTOCOL. Fix block2 grouping/regularization, changing only the added predictions. Mark capacity/epoch differences. No cross-block architecture attribution; retain PLR's annual deterioration. |
| M06 | S6/F1 selection | Retain one of your three candidates | Reuse B11/B12 without new parameter search | Same Y/J predictions; separately verify identities of three50,000-row CSVs | History/macro/structure pipelines; separate macro information setting | Y/J RMSE, displayed scores and source/CSV hashes | F1 selection rationale; M/Q/R | P0 | Verify structural candidate is lowest on all three archived metrics. Retain other variants in X03; do not claim global or independent-test optimality. |
| M07 | S5/S6/F2 identity | Establish the second final pipeline's actual generation chain | New provenance audit; reuse modified B05, with B04 background | 87-input schema, members/target/cutoff, training count and full-data CSV: [to be supplied] | Submission CSV versus reconstruction from actual members/settings; check whether F1 is a member by hashes | Rows/IDs, member count, formula, maximum reconstruction error and hashes | What F2 is and whether it contains F1; M/R | P0 | Obtain actual generator/entry point/settings, ordered87-column dictionary, checkpoint/seed/views and original CSV/hash. Reconstruction≤1e−8 SGD or explain output rounding tolerance. Filename alone cannot establish50/50 weights. |
| M08 | S6/F1–F2 comparison | Compare final pipelines under one protocol | New offline pairing; reuse B11/B12; targeted refit only if B05 predictions are missing | Identical Y/J queries/labels and fitting cutoffs; differing features must be documented | Frozen F1 versus frozen F2; no reweighting with current query labels | Overall RMSE/MAE/bias, monthly/support-group deltas; separate displayed-score table | Which periods/support conditions favor each method; M | P0 | Align35,158/28,995 rows. Query labels never enter features, stopping or new weights. Disclose reused development periods and retain negative results. Do not compute gains between413 and 441 from different evaluations. |
| M09 | S6/delivery gate | Minimal reproduction for each final method | Adapt B11 verification pattern and add independent F2 requirements | Original data/configuration/model/prediction inventories; independent F1/F2 manifests | Original CSV versus saved-model/component replay; source hashes before/after execution | CSV identity, row differences, finite predictions, unchanged inputs and complete environments/commands | Both results traceable and reproducible; R | P0 | Reuse verified F1 evidence; accept F2 independently. Each needs50,000 aligned IDs, finite positive values and explicit replay tolerance. Unreplayable work remains incomplete; distinguish CPU reconstruction from full retraining. |

# B. Appendix branch plan

| Experiment ID | Supplementary purpose | Branch use (reuse/modify/new) | Experiment type (ablation/robustness/sensitivity/additional comparison/failure/reproduction) | Data and features | Models and controls | Key metrics | Main conclusion supported | Priority | Acceptance criteria |
|---|---|---|---|---|---|---|---|---|---|
| X01 | Audit feature redundancy and group effects | Reuse B12's156 individual/group deletion refits and 31→27 deduplication | Ablation | Original three annual windows and separate L stress window; individual/group/joint deletions from 27 fields | Full27 inputs versus each deletion/refit | Delta RMSE, fold n and deletion-improvement counts | M03 information increments and M04 bottlenecks | P1 | Do not add contributions or infer causality; deleting one field does not remove every representation of its source information |
| X02 | Evaluate the cost of cleaning extreme labels | Reuse B09/B10 without forcing their fitting protocols together | Sensitivity/failure analysis | B09 interval-calibration and B10 Q1 protocols, each retaining full validation | Raw versus obvious-low/highest-only cleaning; Q1 IQR controls separately | RMSE/MAE/bias, removed counts and tail SSE | M01 preprocessing and M04 errors | P1 | Separate B09's105,837-row fitting protocol from 147,983-row full refit; do not transfer metrics or equate low rents with wrong labels |
| X03 | Retain your other two versions and information-setting controls | Reuse B11/B12 history/macro/structure archives | Additional comparison/ablation | Y/J and three full submissions; explicitly record macro_policy | History415.86912 and supplied-macro 415.78232 versus F1 | RMSE/MAE/bias, displayed scores and weights | M06 three-way selection and M08 information comparability | P1 | Preserve original variants/weights; future macro values are additional supplied information, not an architectural explanation for tiny score differences |
| X04 | Retain coverage/randomness controls leading to F1 | Reuse B12 rounds8/9/10 | Ablation/sensitivity/failure analysis | Same Y/J; remove36-month fields, average seeds, group by support and shrink weights | Fixed replacement, global versus support-conditioned weights and PLR addition | Delta RMSE, monthly deltas, group n and weight constraints | M04 support differences and M05 complexity increments | P1 | Separate locked choices from later selection. Retain round 9's better annual score; do not attribute all combined changes to PLR. |
| X05 | Record alternative spatiotemporal/macro hypotheses | Reuse B01/B02/B07 | Additional comparison/failure analysis | Separate U monthly-updated, F three-fold and L16-month tables; RNN monthly sequences and PCA rental surfaces | RNN/LSTM versus matched no-latent controls; surface model versus same-row Cat; macro versus same-protocol no-macro | Original-target RMSE, folds/fitting counts and monthly bias | M03 rationale for current histories and M05 alternatives | P2 | Do not mix U and fixed origins; surface diagnostics are not individual-rental scores. Unrun configurations have no failure scores. |
| X06 | Record alternative neural/foundation models and budgets | Reuse B04/B06/B09/B10 and B12 rounds6/7 | Additional comparison/hyperparameter sensitivity | Record schema/context/queries/seed/views/budget for each result | Distinguish official TabM from custom style models, frozen/adapted TabPFN and matched ReLU/MLP baselines | RMSE/MAE, existing measured time/memory only, and budget | M02/M05 model selection | P2 | No architecture ranking across different inputs. Disclose licensing/external data; missing measured cost is [to be supplied]. |
| X07 | Check training-randomness sensitivity of both final methods | New matched repeats after checking B11/B05 model availability | Robustness | Y/J; freeze all recipes and selection rules | F1/F2 seeds 42/43/44; distinguish F1's internal four-seed average from external repeats | Per-seed RMSE, mean/standard deviation and paired directions | Stability of M08 comparison | P1; insufficient budget limits stability claims without blocking point estimates | Retain original seed 42 controls. New seeds must not retune hyperparameters/weight recipes. Downgrade stability claims if inconsistent; do not discard poor seeds. |
| X08 | Correct future-MRT and temporal information availability | New audit/controlled sensitivity; adapt B04 audit and B12/B13 | Sensitivity/data audit | Same aux snapshot; original STATUS versus verified opening dates/future-only lines; identical Y/J | Fixed models with original/new features; refit F1/F2 only when prerequisites hold | Affected rows, feature deltas, RMSE/bias and source version | M01 interpretation and M08 common information boundaries | Audit P0; retraining P0 for historical-availability claims, otherwise snapshot-scenario retraining P1 | Trace sources/rules; change training and inference together. Fixing some stations does not validate all facilities. Inference-only changes to old models cannot pass acceptance. |
| X09 | Check long horizons and overlapping-month boundaries | New/adapted reuse; original B02/B04 L results are background only | Robustness/sensitivity | Fixed2023-12 origin over 16 months and identical queries; separately audit full-data March2025 overlap | F1/F2 same origin; ≤10 versus 11–16 months; optional strict full-data fit through 2025-02 as a separate control | Horizon-band RMSE/bias/n and strict-versus-current deltas | M08 validated horizons and deployment boundaries | P1 | Removing early months from a Y model is not a new L experiment. Do not pool L with Y/J. Budget L first; strict fitting needs a separate budget. |
| X10 | Audit public-target overlap to avoid invalid rankings | Reuse B03 | Reproduction/data-source audit | Independent public snapshot, train reconstruction and test-match statistics | Public-candidate mean reconstruction versus known training targets; not a future forecasting model | Coverage, ambiguity, reconstruction error and assumptions | M00/M08 target and information scope | P2 | Label181.25 as reconstruction and 173.41 as an estimate; exclude from F1/F2 comparisons, fitting and target filling |
| X11 | Test complementarity in F2's actual blend | New paired-error analysis after M07 establishes members | Ablation/statistical diagnostic | Same-origin predictions from actual F2 members; identify relation to F1 | Actual zch component, standalone TabPFN87 and fixed actual blend; list F1 separately if needed | Paired RMSE, E[e·d], error correlation, monthly/support deltas and weights | Mechanism of M05/M08 blending gains | P0 for complementarity claims; P1 for complete-pipeline comparisons only | Do not reoptimize Y/J weights. Low correlation alone does not prove improvement. If F1 is not a member, do not describe F2 as adding TabPFN to F1. |
| X12 | Add uncertainty for paired differences | New analysis using only frozen M08 predictions | Statistical diagnostics | Y and J separately; moving month blocks, with building resampling as sensitivity | F2−F1; fixed difference definitions/group thresholds | RMSE deltas, descriptive intervals and month/building cluster counts | Reliability of M08 differences | P1 | Seed42,3,000 replicates, three-month blocks; record algorithm. Only10/12 months and repeated development selection remain limitations; an interval crossing0 still passes acceptance. |
| X13 | Verify fresh environment and full retraining paths | Reuse B11 checking pattern; add independent F2 workflow | Reproduction | Fixed environment locks and data/source/weight manifests | Original CSV, CPU reconstruction and original-recipe retraining are three distinct levels | Reload errors, versions/resources, unchanged data and measured time/memory | M09 reproduction completeness | P1; relevant parts become P0 if M09 has no usable replay | No requirement for bitwise cross-machine CUDA identity; prespecify/document tolerances. A dependency dry run is not successful training. Preserve original CSVs. |

# C. Branch decisions

| Existing branch/candidate | Decision (reuse/adapt/new/discard) | Changes, including features | Main or appendix | Reason | Effect on the narrative |
|---|---|---|---|---|---|
| B01 `codex/rnn-lstm-market-features` @ cb167e9f | Reuse | Add common protocol labels only; do not rename U as fixed-origin | appendix X05 | Retain matched RNN/LSTM/tree negative results | S5 alternative representations, not required F1/F2 stages |
| B02 `experiment/dynamic-rental-surface-plan` @ 469e34e3 | Reuse | Separate surface and row errors; match the 2022+ fitting subset | Appendix X05/X09 background | Existing market-decomposition and long-horizon evidence | S3/S5 additional controls; not a replacement for L evaluation of both final methods |
| B03 `experiment/public-rent-lookup` @ b1429600 | Reuse audit; discard its predictive-ranking role | Identify target reconstruction and 173.41 estimate; preserve the source branch | appendix X10 | Different information conditions | S1/S6 exclusion of invalid comparisons |
| B04 `experiment/tabpfn` @ c0c987e2 | Adapt and reuse | Select implementations required by actual F2; complete schema/views/context/licensing. Keep broad old adaptation results in appendix. | Main M07 context; appendix X06/X08/X11 | A whole branch cannot substitute for the final 87-input implementation | S5/S6 distinguish foundation models from final pipelines |
| B05 `experiment/tabpfn-anniversary-history` @ 1e69bb79 | Adapt and reuse | Supply actual generator/87 fields/members/CSV/validation; adopted MRT corrections require a new run | main M07/M08/M09; appendix X08/X11/X13 | Only integration and user-designated provenance are currently confirmed | Determines F2 interpretability, comparability and reproducibility; missing evidence fails corresponding P0 checks |
| B06 `feature/neural-tabm-experiment` @ 21e4ec69 | Reuse | Name custom style models and label OOF weight-selection bias | appendix X06 | Differs from official TabM; small gains are not independent-test evidence | S5 additional models and selection-bias controls |
| B07 `feature/temporal-auxiliary-analysis` @ c3c339eb | Reuse | Separate monthly-updated economics from fixed-origin histories; preserve a common backend identity for main evaluation | Main M01/M03 context; appendix X03/X05 | Sufficient evidence for historical/temporal boundaries | S1/S3 valid histories and S5 macro limits |
| B08 ?Git `main`???? @ b05b17f7 | Discard as an effective-performance experiment; retain Git history | Do not overwrite/fabricate scores; list future physical main integration as pending | Exclude from main experiments; reproduction background only | Code scaffold without measured performance | Cannot serve as an S2 fitted baseline |
| B09 `mlp-eda-ensemble` @ 322fa4c2 | Reuse | Separate frozen/calibration/full-data tables and distinguish low-rent rules from intervals | Main M01/M04 definition references; appendix X02/X06 | Extensive cleaning/interval and large-error evidence | S1/S4 identical-input conflicts do not prove a global floor |
| B10 `relu-rental-prediction` @ dde07705 | Reuse | Retain Q1, fitting-only IQR and numerical-library differences; mark missing original outputs | appendix X02/X06 | Do not combine short windows with Y/J rankings | S2/S5 neural references and nested validation |
| B11 `results/hdb-rental-final-ensemble` @ 6f97e107 | Reuse; adapt reproduction inventory | Select F1 from three candidates; F2 needs a separate manifest and does not inherit the old three-CSV acceptance | main M06/M09; appendix X03/X13 | Complete F1 evidence; distinguish CPU reconstruction from full training | S6 F1 selection and acceptance |
| B12 `zch-complete-proj` @ 048eca86 | Reuse | S2 uses saved C7 classical baselines; common27 neural/tree comparisons belong to S5. Preserve actual settings; MRT corrections require X08. | main M00–M06/M08; appendix X01/X03/X04 | Distinguish initial classical trials, historical-information increments and complex-model comparisons with enhanced inputs | Common S1–S6 references and F1; do not present history-enhanced27 inputs as the initial baseline |
| B13 `ziyuan——spatialfeatures` @ 7caf2cb7 | Reuse; adapt MRT sensitivity | Fixed effects/matching estimate associations; distinguish26/27 inputs and clarify school/station deduplication | main M01; appendix X01/X08 | Supports definitions and confounding checks | S1 interpretation of spatial inputs and S4 explanatory boundaries |
| M00/M04/M07/M08/M09 | New audit/analysis; M09 reuses F1 interface | Write only to planned new run directories, preserving original results; see D cards | main | Fill protocol, F2 identity and two-method validation gaps | Both endpoints gain experimental support only after P0 acceptance |
| X07/X08/X09/X11/X12/X13 | New or adapted reuse; conditional priorities in B/D | Recompute/refit missing parts only; reuse existing matched artifacts | appendix | Stability, temporal availability, horizons, complementarity and reproduction | Support key conclusions; negative results remain valid |
| Your historical415.86912 and macro 415.78232 variants | Reuse without a final-method slot | X03 retains complete settings, scores and information scenarios | Compact main M06 selection; appendix details | Avoid treating three similar versions as three endpoints | Support F1 selection while preserving alternatives |

# D. New experiment requirement cards

| Shared execution rules | Requirement |
|---|---|
| Card coverage | Covers all new/modified analyses requiring reevaluation; unchanged reuse does not trigger redundant training |
| Artifact root | `output/planning_validation/main/<run_id>/`; derive run_id from inputs/code/data digests. Offline parts are implemented; see [status](main-execution-status.md). |
| Executability | Follow card inputs, algorithms, controls and acceptance. Entry point: `python -m analysis.main planned`; see [status](main-execution-status.md) for numerical scope and [running guide](../guides/running_experiments.md) for reused commands. Legacy report generators may train/overwrite and are not read-only checks. |
| Budget units | Prefer counts of new base-model fits, full-pipeline evaluations and bootstrap replicates. Actual GPU hours, authorized API costs and deadline: [to be supplied]. Record missing dependencies/budget before execution rather than guessing. |
| Meaning of acceptance | Valid, aligned, verifiable work is complete whether it supports or rejects the hypothesis; lower scores are not mandatory |
| Result selection | Y/J are already examined development periods; do not add searches or redefine primary metrics/groups/endpoints based on them |

### D-M00: common protocol and result-identity audit

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | M00; whether existing results can be reused under explicit conditions; shared S1/S6 |
| Variables and controls | Variable=result source/pipeline; controls=raw-data version, Y/J query IDs/labels, SGD target and metric implementation. Record schema/information policy per result. |
| Splits, seeds and repeats | Fixed Y/J; full-data CSV format/identity only. Deterministic single audit; changed inputs require a new run. |
| Controls and evaluation metrics | Independently recompute saved CSV/predictions against original labels; n, unique/aligned IDs, SHA256, metric differences and temporal violations. Existing `analysis/verify_repository_review.py` checks may be reused. |
| Method | Record commit, data_hash, schema_hash, fit_end, query_range, target_transform, history_policy, macro_policy, seed, selection_stage, prediction_hash and score_source. Join by source-row ID rather than assuming row-order alignment. |
| Expected tables/figures | `protocol_registry.csv`, `alignment_checks.json`, `source_manifest.json`, `metric_recalculation.csv`; no new figure required |
| Acceptance criteria | Zero ID/label/date mismatches and duplicate prediction IDs. Full-precision metric tolerance1e−6 SGD. For rounded CSVs, bound tolerance by half the last prediction unit plus half the reported metric unit; record precision first. Mark missing configurations incomparable and link evidence gaps instead of assuming equivalence. |
| Priority/budget | P0; zero new fits; one scan of saved predictions. Recheck affected cells only after data/config changes; actual I/O time [to be supplied]. |

### D-M04: feature capability and remaining-error diagnostics

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | M04; test whether errors concentrate in sparse histories, months/types or indistinguishable inputs; S4 |
| Variables and controls | Variables=fixed model/prespecified groups; controls=identical Y/J rows, original predictions and target/scale. Do not change models or select thresholds by validation performance. |
| Splits, seeds and repeats | Y/J separately; four24-/36-month support groups using origin-known availability. Mark low/mid/high rent groups using fitting q10/q90 as post-hoc; normalize flat types. One deterministic recomputation. |
| Controls and evaluation metrics | M03 representative62-input model and F1; overall/group n, RMSE, MAE, bias, SSE share and monthly deltas |
| Method | Read support from saved origin-specific feature frames; merge predictions/raw labels by ID and reconcile group SSE. Group identical inputs by **actual ordered model inputs**; describe local variation without inferring global irreducible error. |
| Expected tables/figures | `error_groups.csv`, `monthly_errors.csv`, `feature_support_audit.json`; one month/model error plot and one support-group plot |
| Acceptance criteria | Retain complete Y/J. Nonoverlapping n/SSE must reconcile; label overlaps and groups n<100 as sparse without omitting values. Do not copy another model's25.7% low-rent contribution. Inference cannot use actual-rent groups. |
| Priority/budget | P0; zero fits; at most three frozen prediction sets×two windows. Recover missing representative archives before separately registering refits. |

### D-M07: F2 identity, schema and reconstruction

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | M07; establish the generator behind screenshot 413.19635 and its relation to F1; S5/S6 |
| Variables and controls | Variables=member predictions/recorded fixed combination; controls=original CSV/settings/IDs/weights. Do not fit weights to the screenshot score. |
| Splits, seeds and repeats | Use all 50,000 queries for identity checks; recover Y/J/inner-selection records first. Seeds/views/checkpoint and actual fitting count [to be supplied]; one deterministic reconstruction. |
| Controls and evaluation metrics | Original CSV versus actual formula; compare member hashes with archived F1/history/macro predictions; columns, n, IDs, max difference, source hashes |
| Method | Obtain generator commit/entry point, ordered87-input dictionary with provenance/cutoffs, full-precision members, weights and target baseline. Verify SGD baseline is added once. Draw dependency edges only after member identity is confirmed. |
| Expected tables/figures | `f2_manifest.json`, `feature_dictionary_87.csv`, `component_registry.csv`, `reconstruction_checks.json`, confirmed component dependency graph |
| Acceptance criteria | Trace every input/weight; 50,000 aligned finite positive predictions; reconstruction≤1e−8 SGD or demonstrated CSV-rounding tolerance. A count of 87 is not a dictionary. The current remote merge commit alone cannot pass this card. |
| Priority/budget | P0; recover artifacts and reconstruct on CPU before training. Missing models/CSV stop numerical verification and remain [to be supplied]; do not launch new TabPFN training first. |

### D-M08: matched-protocol comparison of final pipelines

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | M08; compare F1/F2 errors and applicability on identical records/information; S6 |
| Variables and controls | Variable=complete F1/F2 pipeline; controls=origin, fitting cutoff, query IDs/labels, metrics, source-data policy. Document different inputs/pretraining; no architecture-only attribution. |
| Splits, seeds and repeats | Complete Y/J; prefer original frozen-seed predictions. Rebuild missing origins with fixed original recipe/seed 42; epochs and combination rules follow pre-origin selection. One evaluation per pipeline/window. |
| Controls and evaluation metrics | F1 versus F2; overall RMSE/MAE/bias and month/type/support-group n/deltas. Fix the comparison objective before seeing winners. |
| Method | Complete M00+M07 first. Recompute matched predictions on CPU; otherwise refit only missing pipeline/origin, rebuilding histories/vocabularies/transforms within fitting data. Never backtest150k models on their fitting Y/J rows. F2 inner weights must exclude corresponding Y/J labels. |
| Expected tables/figures | `final_pair_metrics.csv`, `paired_predictions`, `monthly_pair.csv`, `protocol_checks.json`; monthly differences and support-group plots |
| Acceptance criteria | Fully align Y35,158/J28,995 and information policies; retain model/weight selection dates and positive/negative differences. Do not pool overlapping windows as independent data or call examined development periods blind tests. |
| Priority/budget | P0; CPU first; at most two pipelines×two windows=four evaluations, reusing valid predictions. Actual fits/VRAM/GPU hours depend on M07 inventory: [to be supplied]. |

### D-M09: minimum reproduction acceptance for both endpoints

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | M09; confirm executable provenance and prediction replay for both final results; S6 |
| Variables and controls | Variables=original CSV/replay mode; controls=data, code, weights, preprocessing, members, IDs. Preserve original CSVs. |
| Splits, seeds and repeats | All50,000 queries; original seeds/learned weights; one replay per pipeline. M00 verifies Y/J provenance. |
| Controls and evaluation metrics | Original CSV versus reconstructed/reloaded predictions; maximum absolute difference, IDs/rows/columns, positive/finite values, input hashes before/after |
| Method | Reuse B11 standard-library F1 verification/CPU reconstruction and record missing packages. Build independent F2 manifest/entry point from M07. Register component reconstruction, loaded-model inference and raw-data retraining as separate acceptance levels. |
| Expected tables/figures | Per-endpoint `manifest.json`, `verification.json`, executable commands and environment inventory; no paper generation needed |
| Acceptance criteria | Two valid50,000-row outputs; M07/original-format tolerances; clear code/data/artifact identities. F1 acceptance does not substitute for F2. Register incomplete full retraining under X13 without overstating reproduction. |
| Priority/budget | P0; zero new fits initially; one F1 recheck and one independent F2 replay. Missing reload and training resources separately [to be supplied]. |

### D-X07: matched-seed robustness

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X07; whether differences depend on one training seed; S5/S6 |
| Variables and controls | Variable=external training seed; controls=schema, origin, hyperparameter/epoch rules, budget, blend algorithm. F1's internal four-seed average is not four external repeats. |
| Splits, seeds and repeats | Y/J; three external repeats42/43/44. Reuse valid42 artifacts and add43/44 only. M07 determines shared-pretraining seed policy. |
| Controls and evaluation metrics | Same-seed F2−F1; per-seed RMSE/MAE/bias and mean/SD. Do not report the best seed as the result. |
| Expected tables/figures | `seed_metrics.csv`, `fixed_recipe.json`, paired-seed point plot and resource records per fit |
| Acceptance criteria | Retain all planned repeats and failure reasons. Freeze selection rules; inner weight fitting uses only pre-origin labels. Do not change features for a new seed. |
| Priority/budget | P1; at most two methods×two windows×three seeds=12 evaluations, at most eight new if42 is valid. Each pipeline may contain many models; actual budget [to be supplied]. Preflight before scheduling. |

### D-X08: temporal amenity availability and sensitivity

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X08; whether MRT source status misrepresents historical accessibility and whether correction changes comparisons; S1/S6 |
| Variables and controls | Variable=availability policy; controls=distance calculation, deduplication/count radii, raw CSV, Y/J, hyperparameters/fixed duration. Separate planned facilities from operating status. |
| Splits, seeds and repeats | Pre-origin operation evidence [to be supplied: date source/version]. Audit full main/query tables before Y/J sensitivity; one controlled seed 42 run. |
| Controls and evaluation metrics | Original STATUS versus verifiable-date/future-only-line correction; affected-field rates, feature deltas, RMSE/MAE/bias. Do not use validation rents to choose stations. |
| Method | Review B04's34 future-only coordinates and reconcile B12/B13 deduplication. Incomplete dates permit partial-correction claims only. Rebuild all affected training/query features, refit and save a new version; reuse valid frozen controls. |
| Expected tables/figures | `facility_source_manifest.json`, `availability_issues.csv`, `feature_deltas.csv`, `sensitivity_metrics.csv`; affected-row table and compact comparison chart |
| Acceptance criteria | Disclose issues/information scenario. No unsupported claim that future stations have no effect. Historical-as-of claims require both endpoints under corrected policy; supplied-snapshot claims must explicitly mark unfinished sensitivity. |
| Priority/budget | Audit P0(zero fits); corrected refits P0 for historical-as-of claims, otherwise P1. At most two policies×two methods×two windows=eight evaluations, reusing valid old-policy results. Fits/hours [to be supplied]. |

### D-X09: matched long horizons and overlapping-month audit

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X09; whether F1/F2 extrapolation degrades differently beyond ten months; S6 |
| Variables and controls | Variables=method/horizon; controls=common2023-12 origin, fit through 2023-11, pre-origin preprocessing/history/selection. Do not reuse Y/J models. |
| Splits, seeds and repeats | All queries over 2023-12 through 2025-03, sixteen months; one seed 42 run each. Missing earlier selection evidence remains [to be supplied]; do not tune against the long window. |
| Controls and evaluation metrics | F1 versus F2; overall/1–10/11–16-month RMSE/MAE/bias/n; disclose any F2 month10 plateau |
| Expected tables/figures | `long_horizon_metrics.csv`, `horizon_predictions`, `cutoff_checks.json`; monthly/horizon plots and information audit for 380 overlapping-month rows |
| Acceptance criteria | Identical long-window rows/origin; no backward use of later weights; no pooling with Y/J. Strict full-data refits through 2025-02 require a separate execution budget. |
| Priority/budget | P1; initially at most two methods×one origin=two evaluations; fits/GPU hours [to be supplied]. No automatic strict full-data refit under this card. |

### D-X11: complementarity and ablation of actual F2 members

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X11; observable error complementarity from F2's added members; S5/S6 |
| Variables and controls | Variables=frozen actual member set/original weights; controls=same Y/J, target restoration, origin, actual original weights. M07 determines whether F1 is a member. |
| Splits, seeds and repeats | Matched M08 predictions; one deterministic recomputation. Reuse X12 for uncertainty without reselecting weights. |
| Controls and evaluation metrics | Actual zch component, standalone TabPFN87, actual combination; RMSE/MAE/bias, error correlation, E[(p−y)(a−p)]. Extend to the real formula if more than two branches. |
| Method | Identify p/a from actual members, not assumed F1/TabPFN. Reconstruct fixed weights and check the MSE identity. Prespecify renormalization for member deletion; do not optimize Y/J weights. |
| Expected tables/figures | `member_ablation.csv`, `error_moments.csv`, `blend_identity.json`; member differences and monthly-delta plots |
| Acceptance criteria | Match F2 reconstruction; low correlation alone is insufficient. Distinguish complete-pipeline and added-member effects. If F2 is not a blend, mark inapplicable with reasons and test actual-method ablations instead. |
| Priority/budget | P0 for complementarity-gain claims, otherwise P1. Zero training with available predictions; recompute actual k members×two windows. Recover missing members before expanding search. |

### D-X12: descriptive intervals for paired differences

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X12; quantify month/building sensitivity conditional on frozen predictions; S6 |
| Variables and controls | Variable=resampled month blocks/buildings; controls=identical draws applied to paired F1/F2 records/models. No refitting/reweighting. |
| Splits, seeds and repeats | Y/J separately; seed 42; 3,000 noncircular moving-block draws of three months. Sample starts uniformly with replacement from 0 throughT−3, concatenate at leastT months, then truncate toT; retain every transaction weight. Separately resample the same number of normalized Town/Street/Block clusters with replacement for 3,000 building-sensitivity replicates. |
| Controls and evaluation metrics | Recompute F2−F1 RMSE/MAE differences per draw; point estimates, 2.5%/97.5% quantiles, cluster counts/group n. Analyze Y/J separately, not as independent replications. Preserve sampling multiplicities rather than deduplicating clusters. |
| Expected tables/figures | `paired_intervals.csv`, `resampling_spec.json`; difference-interval plot |
| Acceptance criteria | Retain all paired transactions inside blocks. Independent-row bootstrap is not temporal uncertainty. Disclose omitted development reuse, selection and fitting uncertainty. Crossing0 is not failure. |
| Priority/budget | P1; zero fits; two windows×two schemes×3,000 metric recomputations. Preaggregate SSE/absolute errors to bound CPU memory. |

### D-X13: independent environments and full retraining

| Fields | Requirement |
|---|---|
| ID/purpose/hypothesis/stage | X13; installation, reload and full-training levels beyond M09; S6/R10% |
| Variables and controls | Variables=environment/run mode; controls=data, code, recipe, checkpoint source, seeds, learned weights. Original results are read-only. |
| Splits, seeds and repeats | Fix full-data recipes; one CPU reconstruction each; one full GPU refit each if scheduled. Do not rerun historical searches or report independent validation on fitting labels. |
| Controls and evaluation metrics | Original CSV versus new-environment reconstruction/model reload/original-recipe refit; hashes, versions, prediction differences, measured time/VRAM and failure reasons |
| Expected tables/figures | Environment locks, `clean_environment_checks.json`, `checkpoint_replay.json`, `training_run.json`; reproduction-level table for both endpoints |
| Acceptance criteria | Separate CPU and full-training acceptance; document licensing/download prerequisites and prespecified GPU tolerances. New refits have unknown Kaggle scores. Successful pip resolution does not prove successful training. |
| Priority/budget | P1; required M09 replay gaps become P0. Start with zero-fit CPU checks; at most two full final pipelines. F1 has about 15 predictors; F2 count and GPU hours/VRAM [to be supplied]. |

# E. Narrative-to-branch mapping

| Narrative stage | Main experiments | Appendix experiments | Primary sources | Conditions for progressing to the next stage |
|---|---|---|---|---|
| S1 EDA/preprocessing | M00, M01 | X02, X08 audit, X10 | B12/B13/B09; B04 temporal issues | Freeze data/query identities, processing rules, history cutoffs and snapshot/as-of meanings. Mark unresolved risks without unsupported historical-availability claims. |
| S2 classical models | M02 | X06 classical/simple-neural controls | B12 C7 classical baselines; B10 separate window | Complete same-C7/Y mean, linear, RandomForest and two boosted-tree references. Features may be studied even without a reliable gain; 27-input complex models belong to S5. |
| S3 feature engineering | M03 | X01, X03, X05 | B12 fixed ladder; B07/B13 rationale | Verify four nested-input increments with fixed learners on Y/J; justify retained/rejected groups without requiring every level to improve |
| S4 capability/bottlenecks | M04 | X01, X02, X04 | B12 original predictions; B09 diagnostic definitions only | Reconcile same-model month/support/tail errors and propose testable complex-model objectives. Local indistinguishability is not a global RMSE floor. |
| S5 complex models | M05 and M07 method identity | X04, X05, X06, X11 | B12 common27-input tree/neural controls and round 10; B04/B05 | Document matched-input controls/selection and confirm F2 features, members, target, cutoff. Retain negative results; do not assume complex models must win. |
| S6 two final results | M06, M07, M08, M09 | X03, X07, X08, X09, X11, X12, X13 | B11/B12 support F1; B05 actual generator supports F2 | Accept F1 selection, F2 generation, matched two-method results and independent reproduction inventories. Two screenshots alone are insufficient. |

# F. Remaining gaps

| Category/gap | Available evidence | [To be supplied] | Experiments/priority affected | Provider/resolution |
|---|---|---|---|---|
| F2 generation provenance | User screenshot 413.19635; designated B05; current frozen head1e69bb7 is a merge | [To be supplied] Actual generator commit, script, complete settings and run record | M07/M08/M09, P0 | Generator author supplies/pushes artifacts for direct audit |
| F2's87 inputs and members | B04 documents77/84/89 inputs, not a substitute for 87 | [To be supplied] Ordered87-field dictionary, provenance, as-of/category/missing policy, checkpoint/adapter/views/seed, actual fitting/context size, formula/member IDs | M07, X08/X11, P0 | Generator supplies; match repository evidence rather than guessing from filenames |
| F2 numerical files | Screenshot only; no verified CSV/member predictions | [To be supplied] Original50,000-row CSV, SHA256, full-precision members, Y/J row predictions and label IDs | M00/M07/M08/M09, P0 | Obtain existing artifacts; plan targeted Y/J refits if absent |
| F1/F2 relationship | Complete F1 package; B05 merges both code sides | [To be supplied] Whether F2 contains F1 or another zch variant's predictions | M07/M08/X11, P0 | Verify hashes/settings; Git integration is not model blending |
| Prediction scenario | Supplied aux includes future macro/static facilities; final F1 excludes the macro branch | [To be supplied] Competition-snapshot versus strict historical deployment setting; F2 external-data/macro policy | M00/M01/X08, P0 | Team confirms scenario; course materials supply any required rules |
| Historical MRT sources | B04 audits future-only lines; shared pipeline filters STATUS | [To be supplied] Verifiable opening dates/versions, initial correction scope, and historical school/mall dates | X08 audit P0; correction conditional P0/P1 | Audit existing sources; complete historical dates require an authorized source |
| Common validation and selection records | Y/J and F1 available; most periods already used in development | [To be supplied] F2 pre-origin model/epoch/weight rules and identical queries; availability of genuinely untouched data | M00/M08 P0; independent generalization later | Recover/audit recipes; do not relabel examined months blind tests |
| Environments/resources | Historical device/environment records are scattered; F1 CPU package works | [To be supplied] Current CPU/RAM/GPU/VRAM, GPU hours, concurrency, checkpoints, licenses/API costs | X07/X08/X09/X13 and M08 missing-prediction refits | Team/runtime confirmation; planning performs no installation/training |
| Execution branches | Existing13 branches/snapshots | [To be supplied] Physical main/appendix integration branch names, review/merge policy and owners | Required before table C execution; does not block this plan | Team confirmation; no branch creation/overwrite now |
| Deadline and team details | Skill contains a placeholder deadline only | [To be supplied] Actual deadline, team/member names, student IDs, workload and any official template/detailed deductions | Scheduling/eventual delivery; no manuscript in this stage | User/course materials; Git authors do not identify student IDs |
| Statistical/cost evidence | Some old routes have time/seed/interval records under different protocols | [To be supplied] Matched F1/F2 training repeats, same-hardware costs and paired intervals | X07/X12/X13, P1 | New analysis/matched repeats; limited budget narrows corresponding claims |

# G. Priorities and execution advice

| Priority/order | Experiment | Dependencies | Execution advice | Completion/minimum delivery |
|---|---|---|---|---|
| P0-1 Identity and policy | M00, M01, X08 audit; recover M07 concurrently | Existing data/13 branches; F2 generator artifacts | Start with zero-fit registration, alignment, EDA and temporal audit; confirm snapshot/as-of setting | Complete protocol/input/source records; expose F2 gaps and temporal-risk rules |
| P0-2 Maximize reuse | M02, M03, M05, M06; recompute M04 | M00/M01; existing frozen metrics/predictions | Do not rerun aligned classical models, 16 ladder fits or F1 selection; complete missing metadata/diagnostics only | Matched-input table, fixed ladder, same-model errors, F1 selection inventory; controls in every table |
| P0-3 Fix the F2 implementation | Complete M07; X11 if claiming complementarity | Generator/schema/CSV/members | Recover members and reconstruct on CPU before new models or weight searches | Verify87 fields, members, formula, target restoration, training counts, origins, file identities and F1 relationship |
| P0-4 Two-method evaluation | M08; X08 correction/evaluation for strict-as-of claims | M00+M07; same-origin F1/F2 predictions | Recompute valid predictions; targeted recipe refits only for missing origins. Without a GPU budget, list precise pending operations first. | Complete Y/J overall/month/support comparisons and information policies; screenshots cannot substitute |
| P0-5 Independent endpoint acceptance | M09; required X13 replay | M07/M08, available environments/weights | Separate manifest per endpoint; old F1 checks do not cover F2. Distinguish CPU reconstruction, model reload and training. | Two traceable, replayable50,000-row outputs with unchanged sources; honestly retain independent-generalization/full-refit gaps |
| P1-1 Low-cost confidence checks | Reuse X01/X02/X03/X04; X11 if not P0, and X12 | Relevant aligned archives; X12 needs M08 | First handle zero-fit ablations, unselected variants, selection bias and paired intervals | Positive/negative controls, intervals and protocol limitations support key conclusions; PDF length is outside this stage |
| P1-2 Budgeted stronger checks | X07, X08 correction if not P0, X09, X13 full refits | Fixed F2 implementation, confirmed environment/GPU hours | Schedule operation counts, prioritizing main-conclusion sensitivities; add only missing repeats/origins | Separately register stability, long horizons, amenity sensitivity and reproduction levels; retain failures/negative results |
| P2 Exploratory archive | Nonessential X05/X06/X10 details | Existing source reports/metrics | Reuse by topic without restarting broad searches; separately mark unrun, broken or incomparable entries | Deduplicate and cover13 branches with inspectable originals; target reconstruction stays outside prediction rankings |

| Minimum deliverable/acceptance gate | Required content | Status if incomplete |
|---|---|---|
| **main P0** | Complete M00–M09, X08 audit and scenario-required corrections; X11 is P0 if complementarity is essential. F1/F2 each require controls, metrics, provenance and reproduction inventories. | If F2 has only a screenshot or lacks matched predictions, state "F1 supported; F2 identity/evaluation/reproduction [to be supplied]" rather than claiming both final methods supported |
| **appendix P0/P1** | Conditional P0 such as X08 and key X01–X04/X11/X12 controls; budget permitting, matched X07/X09/X13 training/extrapolation | Missing repeats, long windows or full refits leave corresponding claims [to be supplied]. Deterministic comparisons remain valid without extending their scope. |
| **Final objective** | **Main P0 should support both final results; appendix P0/P1 should support the credibility of main conclusions** | Evidence and acceptance determine progression, not mandatory complex-model/F2 wins or a particular RMSE |

## TabPFN source integration update

The two user-requested TabPFN branches are now partitioned: main keeps the minimal 77-input background and controls; appendix keeps the full 81-report archive, alternative methods, original dependencies and derived datasets. [Detailed decisions](tabpfn-branch-split.md). This updates earlier statements that no external source had been imported; F2 generation, matched row-level predictions and paired comparisons remain pending.
