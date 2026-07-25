"""
Adapter to convert nwqec PBC QASM into FTCircuitBench PBM QuantumCircuit.
Also provides entrypoint to run nwqec PBC pipeline.

Parses lines like:
  t_pauli +XIZ;
  s_pauli -XYZI;
  z_pauli +ZII;
  m_pauli -ZII;

and emits PBM-style gates that FTCircuitBench analyzers expect:
  R<activePauli>(angle) on active qubits (non-I positions),
  Meas<sign><activePauli> on active qubits (sign preserved).

Angle mapping per user spec:
- t_pauli -> ±pi/8
- s_pauli -> ±pi/4
- z_pauli -> ±pi/2
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.qasm2 import dumps as qasm2_dumps

from ftcircuitbench.analyzer import analyze_pbc_circuit
from ftcircuitbench.pbc_converter.layers import commuting_layer_runs
from ftcircuitbench.pbc_converter.pbc_circuit_saver import (
    save_pbc_layers_txt,
    save_pbc_measurement_basis_txt,
)
from ftcircuitbench.pbc_converter.pbm import PBM, Rotation
from ftcircuitbench.pbc_converter.tab_gate import TableauForGate

_QREG_RE = re.compile(r"^qreg\s+([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\];")
_T_RE = re.compile(r"^t_pauli\s+([+-][IXYZ]+)\s*;?")
_S_RE = re.compile(r"^s_pauli\s+([+-][IXYZ]+)\s*;?")
_Z_RE = re.compile(r"^z_pauli\s+([+-][IXYZ]+)\s*;?")
_M_RE = re.compile(r"^m_pauli\s+([+-][IXYZ]+)\s*;?")


def is_nwqec_available() -> bool:
    try:
        import nwqec as _nq  # noqa: F401

        return True
    except Exception:
        return False


def transpile_to_pbc_cpp(
    circuit_input: Union[QuantumCircuit, str],
    is_file: bool = False,
    epsilon: float | None = None,
    t_opt: bool = False,
    keep_cx: bool = False,
    forbid_python_fallback: bool = True,
    output_prefix: Optional[str] = None,
) -> Tuple[QuantumCircuit, Dict]:
    import nwqec as nq

    fuse_supported = hasattr(nq, "fuse_t")
    fuse_applied = False
    pre_opt_rotation_ops = 0
    pre_opt_measurement_ops = 0
    post_opt_rotation_ops = 0
    post_opt_measurement_ops = 0
    pre_opt_stats: Dict = {}

    if forbid_python_fallback and not getattr(nq, "WITH_GRIDSYNTH_CPP", False):
        raise RuntimeError(
            "nwqec C++ gridsynth not available; refusing Python fallback"
        )

    # Load circuit to nwqec Circuit (using file-based load to avoid QASMParser dependency)
    tmp_path = None
    try:
        if isinstance(circuit_input, QuantumCircuit):
            qasm = qasm2_dumps(circuit_input)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".qasm", delete=False
            ) as tmp:
                tmp.write(qasm)
                tmp_path = tmp.name
            circ = nq.load_qasm(tmp_path)
        elif isinstance(circuit_input, str):
            if is_file:
                circ = nq.load_qasm(circuit_input)
            else:
                # treat as QASM source string
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".qasm", delete=False
                ) as tmp:
                    tmp.write(circuit_input)
                    tmp_path = tmp.name
                circ = nq.load_qasm(tmp_path)
        else:
            raise TypeError("circuit_input must be QuantumCircuit or str")

        # Use top-level to_pbc and optional opt_t
        kwargs = {}
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        circ = nq.to_pbc(circ, keep_cx=keep_cx, **kwargs)
        pre_opt_counts = circ.count_ops()
        pre_opt_rotation_ops = pre_opt_counts.get("t_pauli", 0)
        pre_opt_measurement_ops = pre_opt_counts.get("m_pauli", 0)
        # Analyze pre-optimization PBC circuit to populate pre_opt_* stats
        pre_qasm = circ.to_qasm()
        if output_prefix:
            _save_nwqec_artifacts(pre_qasm, output_prefix, "pre_opt")
        pre_pbc_qc, pre_basic_stats = pbc_qasm_to_pbm(pre_qasm)
        pre_analysis = analyze_pbc_circuit(
            pre_pbc_qc, pbc_conversion_stats=pre_basic_stats
        )
        # Normalize keys to pre_opt_* namespace
        for k, v in pre_basic_stats.items():
            pre_opt_stats[f"pre_opt_{k}"] = v
        for k, v in pre_analysis.items():
            if k.startswith("pbc_"):
                pre_opt_stats[f"pre_opt_{k[len('pbc_') :]}"] = v
            else:
                pre_opt_stats[f"pre_opt_{k}"] = v
        if t_opt and fuse_supported:
            circ = nq.fuse_t(circ)
            fuse_applied = True
        elif t_opt and not fuse_supported:
            # Gracefully skip when the installed nwqec lacks fuse_t (renamed from opt_t).
            print("[nwqec] fuse_t not available; skipping T optimization.")
        post_counts = circ.count_ops()
        post_opt_rotation_ops = post_counts.get("t_pauli", 0)
        post_opt_measurement_ops = post_counts.get("m_pauli", 0)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    # Export to QASM and adapt to PBM circuit
    qasm = circ.to_qasm()
    if output_prefix:
        _save_nwqec_artifacts(qasm, output_prefix, "post_opt")
    pbc_qc, stats = pbc_qasm_to_pbm(qasm)
    # Analyze post-optimization PBC circuit to populate pbc_* stats
    post_analysis = analyze_pbc_circuit(pbc_qc, pbc_conversion_stats=stats)
    stats.update(post_analysis)
    stats["pbc_fuse_t_supported"] = fuse_supported
    stats["pbc_fuse_t_applied"] = fuse_applied
    stats["pre_opt_rotation_operators"] = pre_opt_rotation_ops
    stats["pbc_rotation_operators"] = post_opt_rotation_ops
    stats["pre_opt_measurement_operators"] = pre_opt_measurement_ops
    stats["pbc_measurement_operators"] = post_opt_measurement_ops
    stats.update(pre_opt_stats)

    # Normalize Pauli weight keys to match print_circuit_stats expectations
    # (expects *_avg_operator_pauli_weight, *_std_operator_pauli_weight, *_max_operator_pauli_weight)
    def _alias_weight(src_prefix: str, dst_prefix: str, dest: Dict[str, Any]) -> None:
        mappings = [
            ("avg_pauli_weight", "avg_operator_pauli_weight"),
            ("std_pauli_weight", "std_operator_pauli_weight"),
            ("max_pauli_weight", "max_operator_pauli_weight"),
        ]
        for src_suffix, dst_suffix in mappings:
            src_key = f"{src_prefix}{src_suffix}"
            dst_key = f"{dst_prefix}{dst_suffix}"
            if src_key in dest and dst_key not in dest:
                dest[dst_key] = dest[src_key]

    _alias_weight("pbc_", "pbc_", stats)
    _alias_weight("pre_opt_", "pre_opt_", stats)
    return pbc_qc, stats


def _extract_num_qubits_from_qasm(qasm: str) -> int:
    for line in qasm.splitlines():
        m = _QREG_RE.match(line.strip())
        if m:
            return int(m.group(2))
    # Fallback: try to infer from longest pauli string encountered
    max_len = 0
    for line in qasm.splitlines():
        for pat in (_T_RE, _S_RE, _Z_RE, _M_RE):
            m = pat.match(line.strip())
            if m:
                max_len = max(max_len, len(m.group(1)) - 1)  # minus sign
    if max_len > 0:
        return max_len
    raise ValueError("Could not determine number of qubits from QASM")


def _active_qubits_and_pauli(pauli_with_sign: str) -> Tuple[list[int], str, str]:
    """Return (active_indices, active_pauli_str, sign) from ±[IXYZ]+ string."""
    sign = pauli_with_sign[0]
    pauli = pauli_with_sign[1:]
    active_indices = [i for i, ch in enumerate(pauli) if ch != "I"]
    active_pauli = "".join(ch for ch in pauli if ch != "I")
    return active_indices, active_pauli, sign


def _angle_for(op: str, sign: str) -> str:
    if op == "t":
        return Rotation.PI_8.value if sign == "+" else Rotation.PI_m8.value
    if op == "s":
        return "pi/4" if sign == "+" else "-pi/4"
    if op == "z":
        return "pi/2" if sign == "+" else "-pi/2"
    raise ValueError(f"Unknown op kind for angle mapping: {op}")


def _collect_pbc_paulis(qasm: str) -> Tuple[list[str], list[str]]:
    """Extract the full-width signed rotation and measurement Pauli strings.

    Returns (t_pauli_strings, m_pauli_strings) in emission order. These are the
    same full-width representation the text artifacts use, so they can be handed
    straight to TableauForGate.convert_back.
    """
    t_paulis: list[str] = []
    m_paulis: list[str] = []
    for raw in qasm.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = _T_RE.match(line)
        if m:
            t_paulis.append(m.group(1))
            continue
        m = _M_RE.match(line)
        if m:
            m_paulis.append(m.group(1))
    return t_paulis, m_paulis


def _save_nwqec_artifacts(qasm: str, output_prefix: str, stage: str) -> None:
    """Write the *_tlayers.txt / *_measure_basis.txt artifacts for nwqec output.

    Layer boundaries are recovered from the emitted order (see
    `pbc_converter.layers`), then reused through the same savers the Python path
    uses so both backends produce byte-compatible files.
    """
    t_paulis, m_paulis = _collect_pbc_paulis(qasm)

    layer_tabs = []
    for layer in commuting_layer_runs(t_paulis):
        rows = [TableauForGate.convert_back(p).tableau for p in layer]
        layer_tabs.append(TableauForGate(np.vstack(rows)))
    save_pbc_layers_txt(layer_tabs, f"{output_prefix}_{stage}_tlayers.txt")

    measure_tab = None
    if m_paulis:
        rows = [TableauForGate.convert_back(p).tableau for p in m_paulis]
        measure_tab = TableauForGate(np.vstack(rows))
    save_pbc_measurement_basis_txt(
        measure_tab, f"{output_prefix}_{stage}_measure_basis.txt"
    )


def pbc_qasm_to_pbm(qasm: str) -> Tuple[QuantumCircuit, Dict]:
    """
    Convert nwqec PBC QASM text into a PBM QuantumCircuit and basic stats.

    Rotations are grouped into layers (recovered from the emitted order) and
    separated by barriers, matching the structure the Python PBC path emits.

    Returns:
        (pbc_qc, stats)
    """
    num_qubits = _extract_num_qubits_from_qasm(qasm)
    qreg = QuantumRegister(num_qubits, "q")
    pbc_qc = QuantumCircuit(qreg)

    rotations = 0
    measurements = 0

    for raw in qasm.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue

        m = _S_RE.match(line)
        if m:
            raise RuntimeError(
                f"Encountered s_pauli in nwqec PBC output; unsupported for now: '{line}'"
            )

        m = _Z_RE.match(line)
        if m:
            raise RuntimeError(
                f"Encountered z_pauli in nwqec PBC output; unsupported for now: '{line}'"
            )

    t_paulis, m_paulis = _collect_pbc_paulis(qasm)

    # Emit rotations layer by layer, barrier-separated. Mirrors the Python path:
    # no leading barrier, one barrier between consecutive layers.
    for layer in commuting_layer_runs(t_paulis):
        if pbc_qc.data:
            pbc_qc.barrier(qreg)
        for pauli_with_sign in layer:
            idxs, active_pauli, sign = _active_qubits_and_pauli(pauli_with_sign)
            if not active_pauli:
                continue
            qargs = [qreg[i] for i in idxs]
            angle = _angle_for("t", sign)
            pbc_qc.append(PBM.generate_gate(active_pauli, angle), qargs)
            rotations += 1

    if m_paulis:
        if pbc_qc.data:
            pbc_qc.barrier(qreg)
        for pauli_with_sign in m_paulis:
            idxs, active_pauli, sign = _active_qubits_and_pauli(pauli_with_sign)
            if not active_pauli:
                continue
            qargs = [qreg[i] for i in idxs]
            # Preserve sign in measurement gate name: Meas+XYZ / Meas-XYZ
            pbc_qc.append(PBM.generate_measure(sign + active_pauli), qargs)
            measurements += 1

    stats: Dict = {
        "pbc_t_operators": rotations,  # rotation count (t/s/z combined)
        "pbc_measurement_operators": measurements,
    }
    return pbc_qc, stats
