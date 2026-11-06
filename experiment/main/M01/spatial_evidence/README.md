[Chinese version](README.zh-CN.md)

# Spatial EDA: main selections and provenance verification

English reading copies of the source narratives: [feature associations](source/EDA_summary.md) and [facility controls](source/amenities_adjusted_summary.md). Their linked `_zh.md` originals remain the frozen provenance sources.

| Main evidence | Content | Usage boundaries |
|---|---|---|
| [Raw and adjusted associations](raw_adjusted_comparison.csv) | Six features: CBD, MRT, mall and primary-school distances, floor and building age | Shows Pearson and Pearson adjusted for month, flat type and area; correlation is not predictive gain |
| [Facility-distance bin comparisons](facility_bin_controls.csv) | Three facility types and 18 bins under `joint_cbd` | 149,841 complete records, 159 excluded; clustered over 9,395 buildings, pointwise 95% intervals |
| [Source and input verification](../evidence-audit.json) | Byte-level hashes of original sources and content identity of current raw data | Verifies provenance and statistical definitions; does not independently refit the original regressions |

Source: `ziyuan——spatialfeatures @7caf2cb77095f3b99cd242db0074189a355c0aae`. Original paths and Git blob hashes are in [imported-evidence.json](../imported-evidence.json). `source/` contains unchanged historical snapshots, retaining old relative links and `OPEN_MRT` naming. The current interpretation uses the [provided-facility snapshot scenario](../../../../docs/methods/prediction-scenario.md). Raw CSV CRLF and Git-source LF are normalized only for content-identity checks, without modifying training-input bytes.

The two adjustments differ: the association table controls month, flat type and area; bin models additionally use month × flat-type fixed effects, town fixed effects, quadratic area and age, floor and CBD distance. The distance reference is 0–0.3 km. These must not be described as one model or interpreted as causal facility effects.

Main uses only these two selected tables to explain spatial definitions and confounding; full exploration remains in appendix X01/X08. Differences between the source's 26 columns and main static19, common27 and adjusted62 are in the [feature catalog](../../M03/feature_schemas/README.md).
