"""neural_fusion_v3 — spatial tokenisation, macro encoders and query fusion.

This package **reuses** `experiments/neural_fusion_v2` for the contract reader, the
per-stage scaler, training loop, metrics and prediction export. Only the pieces this plan
changes live here:

* `tokens.py`   — the numeric house/spatial representation (block token vs per-feature tokens)
* `macro.py`    — DBS/COE window readers and their encoders (awaiting the v3 contract)
* `models.py`   — control models assembled from a token configuration

Reusing v2 rather than copying it is deliberate: the comparison must hold sample tables,
scalers, target parameterisation and training protocol byte-identical, and the cheapest
way to guarantee that is to call the same functions.

v2 is frozen (`reports/neural_fusion_v2/FROZEN.md`); this package never writes there.
"""

__version__ = "3.0.0-gate1-tokens"
