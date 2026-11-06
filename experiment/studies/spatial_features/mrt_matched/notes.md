[Chinese version](notes_zh.md)

# MRT distance: descriptive comparisons within shared strata

Reference distance is 0≤d≤0.3 km to the nearest open MRT; other bins are left-open/right-closed, except the final d>2 km bin. Primary strata match approval year-month, flat type, town, 10-square-meter area bins and 5-year age bins; sensitivity analysis uses 20-square-meter and 10-year bins. Area and age match by interval, not exact value.

Each comparison retains only strata containing both groups, without extrapolation beyond common support. Weights h=n_reference×n_comparison/(n_reference+n_comparison) are normalized to average within-stratum mean-rent differences. Different distance contrasts have different overlap populations, so the curve is not one population's dose response. Neither rent-based matching nor test data are used.

contrasts.csv lists raw and standardized mean differences, supported records, distinct buildings and strata for both groups. Repeated buildings are not independent household samples. covariate_balance.csv lists raw/standardized differences and SMD for area, age, maximum floor and CBD distance, using both groups' original pooled standard deviation. Maximum floor is a building attribute, not unit floor. Balance uses strata with nonmissing covariates in both arms and separately reports missing counts.

These are not causal effects: renovation, actual unit floor and detailed location remain uncontrolled; current facility snapshots do not guarantee operation at approval time. Report coarser-bin sensitivity even if primary support is sparse, without selecting only more supported or expected results. Confidence intervals are omitted because repeated buildings and stratum weights require dedicated dependence treatment.

The distant rebound is also examined directly by comparing 1.5–2 and 2+ km with 1–1.5 km. Mean differences cannot exactly decompose the raw median-curve rebound.

- primary_10sqm_5years, (0.3,0.6]km versus [0,0.3]km: 6788 supported strata; arms 17953/33349 records, 1216/2885 buildings; standardized difference -35.6 SGD.
- primary_10sqm_5years, (0.6,1.0]km versus [0,0.3]km: 4632 supported strata; arms 12781/20855 records, 1067/2276 buildings; standardized difference -97.6 SGD.
- primary_10sqm_5years, (1.0,1.5]km versus [0,0.3]km: 1614 supported strata; arms 5516/4060 records, 587/432 buildings; standardized difference -148.9 SGD.
- primary_10sqm_5years, (1.5,2.0]km versus [0,0.3]km: 321 supported strata; arms 878/646 records, 114/93 buildings; standardized difference -214.1 SGD.
- primary_10sqm_5years, (2.0, inf)km versus [0,0.3]km: 122 supported strata; arms 495/308 records, 48/52 buildings; standardized difference -190.6 SGD.
- primary_10sqm_5years, (1.5,2.0]km versus (1.0,1.5]km: 432 supported strata; arms 1178/854 records, 236/121 buildings; standardized difference -46.1 SGD.
- primary_10sqm_5years, (2.0, inf)km versus (1.0,1.5]km: 143 supported strata; arms 435/267 records, 76/56 buildings; standardized difference -34.4 SGD.
- sensitivity_20sqm_10years, (0.3,0.6]km versus [0,0.3]km: 6461 supported strata; arms 20805/41841 records, 1299/3176 buildings; standardized difference -35.0 SGD.
- sensitivity_20sqm_10years, (0.6,1.0]km versus [0,0.3]km: 4838 supported strata; arms 16164/28253 records, 1155/2599 buildings; standardized difference -102.6 SGD.
- sensitivity_20sqm_10years, (1.0,1.5]km versus [0,0.3]km: 1938 supported strata; arms 8295/5469 records, 706/515 buildings; standardized difference -161.2 SGD.
- sensitivity_20sqm_10years, (1.5,2.0]km versus [0,0.3]km: 373 supported strata; arms 1225/871 records, 144/94 buildings; standardized difference -202.8 SGD.
- sensitivity_20sqm_10years, (2.0, inf)km versus [0,0.3]km: 129 supported strata; arms 666/485 records, 51/53 buildings; standardized difference -167.9 SGD.
- sensitivity_20sqm_10years, (1.5,2.0]km versus (1.0,1.5]km: 485 supported strata; arms 1552/1046 records, 258/126 buildings; standardized difference -27.8 SGD.
- sensitivity_20sqm_10years, (2.0, inf)km versus (1.0,1.5]km: 162 supported strata; arms 678/516 records, 88/62 buildings; standardized difference 39.7 SGD.
