"""neural_fusion_v2 — Gate 1 control comparison on the shared v2 contract.

Package-qualified imports only, so the generic module names used by
``experiments/neural_fusion_v1`` (``config``, ``data``, ``models``, ``training``)
cannot collide with this package's modules.

Nothing in this package computes its own historical rent statistics: every
target-derived input comes from ``reports/fusion_contract_v2/<contract_id>/`` written
by the contract owner. If the contract is missing, the CLI fails loudly with
``ContractNotReady`` instead of substituting a local approximation.
"""

__version__ = "2.0.0-gate1"
