"""Frontends that bring circuits from other toolchains into FTCircuitBench.

`cirq_frontend` is the general path: any Cirq circuit lowers to OpenQASM 2 and
into the FTCircuitBench pipeline. `qualtran_frontend` adds the Bloq -> Cirq step
for Qualtran. Both need optional dependencies (`uv sync --extra cirq --extra
qualtran`), so neither is re-exported from the package root.

pyLIQTR also emits Cirq circuits, but its releases pin `numpy<2` and
`qualtran==0.4.0`, which cannot coexist with FTCircuitBench's own dependency
floor. Export QASM from a pyLIQTR environment with `tools/export_cirq_qasm.py`
and analyse the file here; see `docs/examples.md`.
"""

from .cirq_frontend import (
    CIRQ_INSTALL_HINT,
    NonUnitaryCircuitError,
    cirq_to_qasm2,
    cirq_to_qiskit,
    decompose_for_qasm,
    has_qasm_representation,
    is_cirq_available,
    op_census,
)
from .qualtran_frontend import (
    QUALTRAN_INSTALL_HINT,
    bloq_t_complexity,
    bloq_to_cirq,
    bloq_to_qasm2,
    bloq_to_qiskit,
    count_measurement_uncompute,
    is_qualtran_available,
    unitary_uncompute_interceptor,
)

__all__ = [
    "CIRQ_INSTALL_HINT",
    "NonUnitaryCircuitError",
    "QUALTRAN_INSTALL_HINT",
    "bloq_t_complexity",
    "bloq_to_cirq",
    "bloq_to_qasm2",
    "bloq_to_qiskit",
    "cirq_to_qasm2",
    "cirq_to_qiskit",
    "count_measurement_uncompute",
    "decompose_for_qasm",
    "has_qasm_representation",
    "is_cirq_available",
    "is_qualtran_available",
    "op_census",
    "unitary_uncompute_interceptor",
]
