"""
Canonical basis constants and shared pipeline helpers for the GS / SK / NWQEC
Clifford+T transpilers.

All three transpilers MUST share the same staging:

    input -> prepare_input -> is_clifford_t_basis (early return)
          -> to_intermediate_rz -> <engine-specific RZ synthesis>
          -> enforce_pbc_basis -> output

Only the engine-specific RZ-synthesis step is allowed to differ between
pipelines. Everything else is shared so the pipelines remain truly parallel.
"""

from __future__ import annotations

from typing import Tuple, Union

from qiskit import QuantumCircuit, transpile

from ftcircuitbench.parser import load_qasm_circuit

INTERMEDIATE_RZ_BASIS: Tuple[str, ...] = (
    "cx",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "x",
    "y",
    "z",
    "rz",
)
PBC_COMPATIBLE_CLIFFORD_T_BASIS: Tuple[str, ...] = (
    "cx",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "x",
    "y",
    "z",
)


def is_clifford_t_basis(circuit: QuantumCircuit) -> bool:
    """Return True iff every operation in `circuit` is in the PBC Clifford+T basis."""
    allowed = set(PBC_COMPATIBLE_CLIFFORD_T_BASIS)
    return {inst.operation.name for inst in circuit.data}.issubset(allowed)


def prepare_input(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    remove_final_measurements: bool = True,
) -> QuantumCircuit:
    """Normalize pipeline input: accept QC or QASM (string/path), strip measurements."""
    if isinstance(circuit_input, str):
        qc = load_qasm_circuit(circuit_input, is_file=is_file)
    elif isinstance(circuit_input, QuantumCircuit):
        qc = circuit_input.copy()
    else:
        raise TypeError(
            "circuit_input must be a QuantumCircuit or str (QASM source or filepath)."
        )
    if remove_final_measurements:
        qc.remove_final_measurements(inplace=True)
    return qc


def to_intermediate_rz(circuit: QuantumCircuit) -> QuantumCircuit:
    """Transpile to the canonical RZ-bearing intermediate basis at optimization_level=0."""
    return transpile(
        circuit,
        basis_gates=list(INTERMEDIATE_RZ_BASIS),
        optimization_level=0,
    )


def enforce_pbc_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Re-transpile to the PBC-compatible Clifford+T basis if any unexpected gates remain.

    Pass-through if the circuit is already strictly in PBC_COMPATIBLE_CLIFFORD_T_BASIS,
    so no extra qiskit work happens on the common case.
    """
    if is_clifford_t_basis(circuit):
        return circuit
    return transpile(
        circuit,
        basis_gates=list(PBC_COMPATIBLE_CLIFFORD_T_BASIS),
        optimization_level=0,
    )
