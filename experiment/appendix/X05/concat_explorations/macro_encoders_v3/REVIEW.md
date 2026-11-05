# Macro encoders v3 — independent read-only audit

Scope: v3 macro time boundary and the current encoder API. **Read-only**: no export,
no fitting, no artifact rewrite beyond regenerating `smoke.json` with code hashes
(§6). No training grid. No file outside
`experiments/macro_encoders_v3/` and `reports/macro_encoders_v3/` was touched, and
nothing in Hermes2's directories was read or written.

Evidence script:
`experiments/macro_encoders_v3/audit.py` (re-runnable, prints the numbers below).

Contract audited:
`fusion-contract-v3.0.0__43867d0ba111__c52fef9f33cf__e32fccd39da3`, policy
`origin_asof`.

---

## 0. One documentation error found and corrected

My `API.md` and `STATUS.md` previously stated the fit-only scaler used **27 of 39**
windows on `orig/B` and **22 of 27** on `orig/A`. Both were wrong.

Measured:

| stage | total windows | windows referenced by `fit` | windows referenced only by frozen rows |
|---|---:|---:|---:|
| `orig/A` | 27 | **26** | 1 (the inner-holdout window) |
| `orig/B` | 39 | **38** | 1 (the outer-validation window) |

So the correct figures are **26/27** and **38/39**. The earlier numbers were an
arithmetic slip, not a code difference: the fit set is exactly "all windows except
the one the frozen population points at", which follows from the export's
construction (a single frozen cutoff month shared by every held-out row). Both
documents are corrected.

This is the only substantive defect the audit found in the documentation.

---

## 1. Unique windows per population

| split/stage | rows | `fit` rows → windows | frozen rows → windows |
|---|---:|---|---|
| `orig/A` | 88,192 | 52,310 → **26** | holdout 35,882 → **1** |
| `orig/B` | 123,350 | 88,192 → **38** | validation 35,158 → **1** |

`window_count` sums to the row count for both stages (verified separately in the
package's own `verify.py`, 58 checks). The single frozen window is expected under
`origin_asof`: every held-out row shares the cutoff month, hence one window.

---

## 2. Time boundary: no window reads past its cutoff

For every window, checked against the row origin it serves:

| stage | DBS date violations | COE month violations | window-cutoff vs row-origin mismatches |
|---|---:|---:|---:|
| `orig/A` | 0 | 0 | 0 |
| `orig/B` | 0 | 0 | 0 |

Two separate checks per window, both at zero violations:

- **DBS**: the `dbs_asof_date` and *every* masked-in date in `dbs_dates` satisfy
  `YYYYMM <= cutoff YYYYMM`. Since the cutoff is the last day of the origin month,
  this is the whole-month boundary, not a partial-month one.
- **COE**: every non-zero `coe_months` entry converts back to a month index
  `<= window_cutoff_month`.

Under `origin_asof` the window cutoff equals the row's `origin_month_index` for every
row (0 mismatches both stages), so a validation row reads exactly up to the split
cutoff and a stage-A holdout row up to the inner cutoff. No row sees its own target
month.

---

## 3. Fit-only scaler: independent recomputation

Recomputed with a float64 accumulator from the raw exported arrays, using only the
fit windows:

| stage | channel | state value | hand value | match |
|---|---|---:|---:|---|
| `orig/A` | DBS `log_return` mean | 0.000706076315 | 0.000706076315 | **yes** |
| `orig/A` | DBS `log_return` n_used | 1,570 | 1,570 | **yes** |
| `orig/B` | DBS `log_return` mean | 0.000493428287 | 0.000493428287 | **yes** |
| `orig/B` | DBS `log_return` n_used | 2,326 | 2,326 | **yes** |
| `orig/A` | COE `a.premium` mean | 57,928.028455 | 57,928.028455 | **yes** (error 0.0) |
| `orig/B` | COE `a.premium` mean | 69,901.552564 | 69,901.552564 | **yes** (error 0.0) |

`n_windows_used` matches the measured fit-window count in both stages (26, 38), and
`fitted_on` names the fit population explicitly.

**One nuance worth recording.** An earlier manual check appeared to disagree by
~0.003–0.006 SGD on the COE premium mean. That was an artifact of accumulating in
float32 (the exported arrays are float32): `np.mean` over float32 gives
57,928.031250 where the float64 accumulator gives 57,928.028455. The scaler is
correct; the discrepancy is the accumulator dtype, and it is 3–6 thousandths of an
SGD on a ~58k–70k premium, i.e. immaterial. The audit now reports both, so the
difference cannot be mistaken for a bug again.

---

## 4. Category missing-mask semantics

| stage | per-category observed month-cells (a / b / c / e) | cells with some-but-not-all categories | cells with none | observed cells with premium exactly 0.0 |
|---|---|---|---:|---:|
| `orig/A` | 258 / 258 / 258 / 258 | **0** | 66 | **0** |
| `orig/B` | 402 / 402 / 402 / 402 | **0** | 66 | **0** |

Reading:

- **The four categories are observed symmetrically** in every exported window. There
  is no month in which, say, A had an auction but E did not. This matches the source
  for the months `orig` reaches (the asymmetry only appears in the partial 2026-08,
  which no `orig` window contains).
- **66 fully-empty month-cells** per stage are the leading padding of early windows
  (before 2021-01), not missing data inside the observation range.
- **No observed cell has a premium of exactly zero**, so the "genuine zero premium
  preserved as data" path is exercised only by the synthetic fixture, not by real
  `orig` data. That is worth knowing: the code path is tested, but no real `orig`
  window depends on it. (Source-level category E has 134 rows whose *prevailing*
  premium is zero; that column is not used at all, as documented in the v3 contract.)
- Because `some-but-not-all` is zero, the coarse per-month `coe_mask` and the finer
  per-category `coe_channel_mask` are equivalent **on `orig`**. They would diverge on
  a window covering 2026-08. A consumer relying on the coarse mask is therefore safe
  for `orig`/`early`/`long` but not automatically safe for a later cutoff.

---

## 5. Encoder behaviour on the real leading-padded window

Independent of the package's own tests, the audit confirms `orig/A` window 0 carries
43 leading zeros then 20 real DBS observations, and 11 zeros then 1 COE month — the
layout that made the old `pack_padded_sequence` prefix assumption wrong. The 36-test
suite pins that the token is now invariant to padding position and that changing the
real tail *does* change the token.

---

## 6. Smoke checkpoint provenance

The audit found that `smoke.json` recorded **no code hashes**, so a reader could not
prove which implementation produced the checkpoints. That has been fixed: `smoke.py`
now records SHA-256 of every package module at run time, and the smoke was
regenerated.

After regeneration:

| check | result |
|---|---|
| `smoke.json` records `code_hashes` | yes |
| recorded `encoders.py` hash equals current source | **yes** |
| checkpoint mtime >= `encoders.py` mtime | **yes** |

So the current `smoke.json` and `checkpoints/*.pt` were produced by the current,
post-padding-fix code, and this is now verifiable rather than argued from
timestamps.

Verified content (unchanged by the regeneration):

| stage | rows | windows | tokens | params | both branches non-zero grad | checkpoint reproduces |
|---|---:|---:|---|---:|---|---|
| `orig/A` | 88,192 | 27 | 48-d each | 14,048 | yes | yes |
| `orig/B` | 123,350 | 39 | 48-d each | 14,048 | yes | yes |

Still true and still important: the smoke loss is a **surrogate** `mean(token²)`.
There is no rental-prediction score here and nothing indicates whether macro inputs
help.

---

## 7. What this audit does not cover

- **No `early` / `long` audit.** Same code path, not exercised here.
- **No `target_previous_month` audit.** No such contract has been exported.
- **No assessment of predictive value.** That needs a fusion run.
- **No check of Hermes2's consumer code.** Pending its `macro_bridge`; the follow-up
  will be a read-only review of row mapping and checkpoint reconstruction.

---

## 8. Findings summary

| # | finding | severity | state |
|---|---|---|---|
| 1 | `API.md`/`STATUS.md` stated fit-only window counts as 27/39 and 22/27 | documentation | **corrected** to 38/39 and 26/27 |
| 2 | `smoke.json` recorded no code hashes, so checkpoint provenance was unprovable | provenance gap | **fixed**; smoke regenerated with hashes |
| 3 | Time boundary, fit-only means/counts, category masks, window dedup | — | **0 violations** |
| 4 | Genuine-zero premium path is not exercised by real `orig` data | test coverage note | documented, not a defect |

No code defect was found in the encoder or the macro export. The two actionable
items were in my own documentation and provenance, both now corrected.
