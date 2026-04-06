"""
nwqec-based Clifford+T transpilation entrypoint.
"""

from __future__ import annotations

import os
import tempfile
from typing import Tuple, Union

from qiskit import QuantumCircuit
from qiskit import transpile as qk_transpile
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.qasm2 import loads as qasm2_loads


def is_nwqec_available() -> bool:
    try:
        import nwqec as _nq  # noqa: F401

        return True
    except Exception:
        return False


def transpile_to_clifford_t_cpp(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    epsilon: float | None = None,
    keep_ccx: bool = False,
    remove_final_measurements: bool = True,
    forbid_python_fallback: bool = True,
    return_intermediate: bool = False,
) -> Union[QuantumCircuit, Tuple[QuantumCircuit, QuantumCircuit]]:
    import nwqec as nq

    if forbid_python_fallback and not getattr(nq, "WITH_GRIDSYNTH_CPP", False):
        raise RuntimeError(
            "nwqec C++ gridsynth not available; refusing Python fallback"
        )

    # Build an intermediate circuit in the same spirit as the Python path
    # (RZ-capable basis) for debugging/compatibility when requested.
    def _build_intermediate_qiskit(qc: QuantumCircuit) -> QuantumCircuit:
        INTERMEDIATE_RZ_BASIS = ["rz", "s", "h", "x", "z", "cx"]
        inter = qk_transpile(
            qc, basis_gates=INTERMEDIATE_RZ_BASIS, optimization_level=0
        )
        return inter

    # Normalize to an nwqec-supported basis and strip unsupported utility ops
    def _prepare_for_nwqec(qc: QuantumCircuit) -> QuantumCircuit:
        SUPPORTED_BASIS = [
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "sx",
            "sxdg",
            "rx",
            "ry",
            "rz",
            "cx",
            "cz",
            "swap",
            "ccx",
            "cp",
        ]
        cleaned = QuantumCircuit(*qc.qregs, *qc.cregs)
        for item in qc.data:
            inst, qargs, cargs = item.operation, item.qubits, item.clbits
            if inst.name in {"id", "I", "delay", "barrier"}:
                continue
            cleaned.append(inst, qargs, cargs)
        return qk_transpile(cleaned, basis_gates=SUPPORTED_BASIS, optimization_level=0)

    tmp_path = None
    try:
        # Prepare QASM on disk for nwqec
        if isinstance(circuit_input, QuantumCircuit):
            qiskit_input = circuit_input.copy()
            if remove_final_measurements:
                qiskit_input.remove_final_measurements(inplace=True)
            qiskit_input = _prepare_for_nwqec(qiskit_input)
            qasm = qasm2_dumps(qiskit_input)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".qasm", delete=False
            ) as tmp:
                tmp.write(qasm)
                tmp_path = tmp.name
            circ = nq.load_qasm(tmp_path)
        elif isinstance(circuit_input, str):
            if is_file:
                circ = nq.load_qasm(circuit_input)
                with open(circuit_input, "r") as f:
                    qiskit_input = qasm2_loads(f.read())
                if remove_final_measurements:
                    qiskit_input.remove_final_measurements(inplace=True)
                qiskit_input = _prepare_for_nwqec(qiskit_input)
            else:
                qiskit_input = qasm2_loads(circuit_input)
                if remove_final_measurements:
                    qiskit_input.remove_final_measurements(inplace=True)
                qiskit_input = _prepare_for_nwqec(qiskit_input)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".qasm", delete=False
                ) as tmp:
                    tmp.write(qasm2_dumps(qiskit_input))
                    tmp_path = tmp.name
                circ = nq.load_qasm(tmp_path)
        else:
            raise TypeError("circuit_input must be QuantumCircuit or str")

        # Optional intermediate for return
        intermediate_qiskit = (
            _build_intermediate_qiskit(qiskit_input) if return_intermediate else None
        )

        # Run Clifford+T conversion (RZ synthesis in C++)
        kwargs = {}
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        circ = nq.to_clifford_t(circ, keep_ccx=keep_ccx, **kwargs)

        # Convert back to Qiskit and ensure PBC-compatible Clifford+T basis
        qasm_ct = circ.to_qasm()
        ct_qc = qasm2_loads(qasm_ct)

        PBC_COMPATIBLE_CLIFFORD_T_BASIS = ["cx", "h", "s", "t", "tdg"]
        ct_qc = qk_transpile(
            ct_qc, basis_gates=PBC_COMPATIBLE_CLIFFORD_T_BASIS, optimization_level=0
        )

        if return_intermediate:
            return intermediate_qiskit, ct_qc
        return ct_qc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
