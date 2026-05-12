from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit

import ftcircuitbench.api as api_mod
from ftcircuitbench.api import (
    PipelineConfig,
    run_analysis,
    run_analysis_for_file,
    run_pipeline,
)


def _fake_transpile_gs(circuit, **kwargs):
    ct = QuantumCircuit(circuit.num_qubits)
    ct.h(0)
    if circuit.num_qubits > 1:
        ct.cx(0, 1)
    if kwargs.get("return_intermediate"):
        inter = QuantumCircuit(circuit.num_qubits)
        inter.rz(0.2, 0)
        return inter, ct
    return ct


def _fake_transpile_sk(circuit, **kwargs):
    ct = QuantumCircuit(circuit.num_qubits)
    ct.s(0)
    if kwargs.get("return_intermediate"):
        inter = QuantumCircuit(circuit.num_qubits)
        inter.rz(0.3, 0)
        return inter, ct
    return ct


def _patch_api_dependencies(monkeypatch):
    monkeypatch.setattr(
        api_mod, "transpile_to_gridsynth_clifford_t", _fake_transpile_gs
    )
    monkeypatch.setattr(
        api_mod, "transpile_to_solovay_kitaev_clifford_t", _fake_transpile_sk
    )
    monkeypatch.setattr(
        api_mod,
        "convert_to_pbc_circuit",
        lambda _c, **_kwargs: (QuantumCircuit(2), {"pbc_t_operators": 1}),
    )
    monkeypatch.setattr(
        api_mod,
        "analyze_clifford_t_circuit",
        lambda _c, gridsynth_precision_used=None: {
            "total_gate_count": 2,
            "precision_seen": gridsynth_precision_used,
        },
    )
    monkeypatch.setattr(
        api_mod,
        "analyze_pbc_circuit",
        lambda _c, pbc_conversion_stats=None: {"pbc_measurement_operators": 1},
    )
    monkeypatch.setattr(
        api_mod,
        "calculate_circuit_fidelity",
        lambda *_args, **_kwargs: {
            "fidelity": 0.99,
            "method": "unitary_based",
            "status": "success",
        },
    )


def test_run_pipeline_gs_returns_expected_schema(monkeypatch) -> None:
    _patch_api_dependencies(monkeypatch)
    qc = QuantumCircuit(2)
    qc.h(0)
    cfg = PipelineConfig(
        pipeline="gs",
        return_intermediate=True,
        layering_method="v2",
        layering_max_checks=5,
    )

    result = run_pipeline(qc, cfg)
    as_dict = result.to_dict(include_circuits=True, include_artifacts=True)

    assert result.pipeline == "gs"
    assert "clifford_stats" in as_dict
    assert "pbc_stats" in as_dict
    assert as_dict["fidelity"]["status"] == "success"
    assert result.parameters["layering_method"] == "v2"
    assert result.parameters["layering_max_checks"] == 5


def test_run_pipeline_saves_artifacts(monkeypatch, tmp_path: Path) -> None:
    _patch_api_dependencies(monkeypatch)
    qc = QuantumCircuit(2)
    output_qasm = tmp_path / "artifacts" / "ct_output.qasm"
    pbc_prefix = tmp_path / "artifacts" / "pbc"

    cfg = PipelineConfig(
        pipeline="sk",
        return_intermediate=True,
        clifford_output_path=str(output_qasm),
        pbc_output_prefix=str(pbc_prefix),
    )
    result = run_pipeline(qc, cfg)

    assert output_qasm.exists()
    assert "clifford_t_qasm" in result.artifacts
    assert result.artifacts["pbc_pre_opt_layers"].endswith("_pre_opt_tlayers.txt")


def test_run_analysis_aggregates_multiple_pipelines(monkeypatch) -> None:
    _patch_api_dependencies(monkeypatch)
    qc = QuantumCircuit(2)
    configs = [PipelineConfig(pipeline="gs"), PipelineConfig(pipeline="sk")]

    result = run_analysis(qc, configs, source_path="in-memory")
    assert result.input_path == "in-memory"
    assert set(result.pipelines) == {"gs", "sk"}
    serialized = result.to_dict(include_circuits=False)
    assert "pipelines" in serialized


def test_run_analysis_for_file_loads_and_forwards(monkeypatch, write_qasm) -> None:
    _patch_api_dependencies(monkeypatch)
    qasm_text = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
"""
    qasm_file = write_qasm("api_input.qasm", qasm_text)

    result = run_analysis_for_file(str(qasm_file), PipelineConfig(pipeline="gs"))
    assert result.input_path == str(qasm_file)
    assert result.original_qubits == 2
