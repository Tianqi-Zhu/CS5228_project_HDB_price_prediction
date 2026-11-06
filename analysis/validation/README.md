[Chinese version](README.zh-CN.md)

# Offline experiment acceptance checks

| Module | Responsibility | Dependencies |
|---|---|---|
| `main.py` | Frozen-prediction audits and diagnostics for M00/M01/M03/M04/M05/M07/M08/M09 | Main input NPZ files, raw data and standard scientific Python environment |
| `prepare_inputs.py` | Build frozen inputs from historical predictions/settings; reconstruct permissible support and identical-input groups | Local historical model artifacts; no training |
| `repository_audit.py` | Full file hashes, syntax of all registered Python files and current navigation links | Root manifest; does not import trainers |
| `feature_catalog.py` | Ordered schemas, component inputs and corrections to old PLE/PLR metadata | Frozen metadata and new saved-model input records for Y/J |
| `evidence_audit.py` | Spatial provenance/versions, facility-statistic definitions, matched cleaning controls and row-level recomputation | M01 sources and compact NPZ files; no retraining |
| `refit_audit.py` | Code/data identities, model-reload records and fixed-blend reconstruction for new F1 Y/J fits | Records of 30 new fits and component arrays; the audit itself uses CPU only |

[Current acceptance status](../../docs/planning/main-execution-status.md) · [Report assessment criteria](../../skills/cs5228-final-report/SKILL.md)
