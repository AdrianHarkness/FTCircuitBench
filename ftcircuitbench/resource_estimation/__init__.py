"""Physical-layer resource estimation bridges for FTCircuitBench.

Currently a single backend: the Azure Quantum Resource Estimator, reached
through the optional `qdk` dependency (`uv sync --extra qre`). See
:mod:`ftcircuitbench.resource_estimation.azure_qre`.
"""

from .azure_qre import (
    DEFAULT_MODEL_NAMES,
    HARDWARE_MODELS,
    CircuitResourceEstimate,
    HardwareModel,
    LogicalResourceCounts,
    PhysicalEstimate,
    default_hardware_models,
    estimate_circuit,
    estimate_circuits,
    is_qre_available,
    logical_counts_from_stats,
    qubits_from_label,
    read_ct_stats_csv,
    read_stats_json,
    resolve_hardware_models,
)

__all__ = [
    "CircuitResourceEstimate",
    "DEFAULT_MODEL_NAMES",
    "HARDWARE_MODELS",
    "HardwareModel",
    "LogicalResourceCounts",
    "PhysicalEstimate",
    "default_hardware_models",
    "estimate_circuit",
    "estimate_circuits",
    "is_qre_available",
    "logical_counts_from_stats",
    "qubits_from_label",
    "read_ct_stats_csv",
    "read_stats_json",
    "resolve_hardware_models",
]
