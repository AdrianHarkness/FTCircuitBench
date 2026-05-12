"""
nwqec C++ Gridsynth-based Clifford+T transpilation entrypoint.

Pipeline shape is the canonical one defined in `_basis.py`. Only the RZ-synthesis
step is engine-specific (nwqec C++ Gridsynth here).
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple, Union

from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.qasm2 import loads as qasm2_loads

from ._basis import (
    INTERMEDIATE_RZ_BASIS,
    PBC_COMPATIBLE_CLIFFORD_T_BASIS,
    enforce_pbc_basis,
    is_clifford_t_basis,
    prepare_input,
    to_intermediate_rz,
)

# Utility ops that nwqec's QASM loader doesn't accept and that carry no semantic
# weight after the qiskit transpile stage.
_NON_SEMANTIC_OPS = {"id", "I", "delay", "barrier"}


def is_nwqec_available() -> bool:
    try:
        import nwqec as _nq  # noqa: F401

        return True
    except Exception:
        return False


def _strip_non_semantic(qc: QuantumCircuit) -> QuantumCircuit:
    """Drop barriers/delays/ids — nwqec's loader rejects them, and they don't
    affect the transpiled circuit semantics post-canonical-intermediate."""
    cleaned = QuantumCircuit(*qc.qregs, *qc.cregs)
    for item in qc.data:
        if item.operation.name in _NON_SEMANTIC_OPS:
            continue
        cleaned.append(item.operation, item.qubits, item.clbits)
    return cleaned


def transpile_to_clifford_t_cpp(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    epsilon: float | None = None,
    keep_ccx: bool = False,
    remove_final_measurements: bool = True,
    forbid_python_fallback: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    """Transpile to Clifford+T via the nwqec C++ Gridsynth backend.

    Pipeline:
      1. prepare_input (QC/QASM, optional measurement removal)
      2. early-return if already in PBC Clifford+T basis
      3. to_intermediate_rz (canonical {cx, h, s, rz} basis)
      4. nwqec C++ Gridsynth synthesis of Rz gates
      5. enforce_pbc_basis ({cx, h, s, t, tdg})
    """
    import nwqec as nq

    if forbid_python_fallback and not getattr(nq, "WITH_GRIDSYNTH_CPP", False):
        raise RuntimeError(
            "nwqec C++ gridsynth not available; refusing Python fallback"
        )

    initial_circuit = prepare_input(
        circuit_input,
        is_file=is_file,
        remove_final_measurements=remove_final_measurements,
    )

    if is_clifford_t_basis(initial_circuit):
        print(
            "      Circuit is already in Clifford+T basis. Skipping RZ transpilation."
        )
        return (
            (initial_circuit, initial_circuit)
            if return_intermediate
            else initial_circuit
        )

    intermediate_circuit = to_intermediate_rz(initial_circuit)

    # Hand the canonical intermediate off to nwqec via QASM round-trip.
    nwqec_input = _strip_non_semantic(intermediate_circuit)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".qasm", delete=False) as tmp:
            tmp.write(qasm2_dumps(nwqec_input))
            tmp_path = tmp.name
        circ = nq.load_qasm(tmp_path)

        kwargs: dict = {}
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        circ = nq.to_clifford_t(circ, keep_ccx=keep_ccx, **kwargs)

        ct_qc = qasm2_loads(circ.to_qasm())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    ct_qc = enforce_pbc_basis(ct_qc)
    return (intermediate_circuit, ct_qc) if return_intermediate else ct_qc


__all__ = [
    "INTERMEDIATE_RZ_BASIS",
    "PBC_COMPATIBLE_CLIFFORD_T_BASIS",
    "is_nwqec_available",
    "transpile_to_clifford_t_cpp",
]
