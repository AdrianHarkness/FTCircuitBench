"""
Public programmatic API for FTCircuitBench.

This module wraps the lower-level transpilers, PBC converter, fidelity helpers,
and analyzers into a small set of documented functions and data classes. The
functions here are intended to be stable entry points for both CLI tooling and
library use.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.qasm2 import dump as qasm2_dump
from qiskit.qasm2 import dumps as qasm2_dumps

from .analyzer import analyze_clifford_t_circuit, analyze_pbc_circuit
from .fidelity import calculate_circuit_fidelity
from .parser import load_qasm_circuit
from .pbc_converter import convert_to_pbc_circuit
from .transpilers import (
    transpile_to_gridsynth_clifford_t,
    transpile_to_solovay_kitaev_clifford_t,
)

LayeringMethod = Literal["bare", "v2", "singleton"]
PipelineName = Literal["gs", "sk"]


@dataclass
class PipelineConfig:
    """
    Configuration for a single pipeline run.

    Attributes:
        pipeline: Which pipeline to execute ('gs' for Gridsynth or 'sk' for Solovay-Kitaev).
        gridsynth_precision: Precision passed to Gridsynth when pipeline='gs'.
        sk_recursion: Recursion depth for Solovay-Kitaev when pipeline='sk'.
        layering_method: PBC layering strategy ('bare', 'v2', or 'singleton').
        layering_max_checks: Optional bound on v2's backward layer scan; ignored for other methods.
        optimize_t_maxiter: Number of T-merging iterations for the PBC step (0 disables optimization).
        prefer_cpp: Prefer the nwqec C++ backend for Gridsynth when available.
        use_nwqec_pbc: Use the nwqec C++ PBC adapter when available; set False to
            force the Python PBC pipeline (`RotationPauliCirc` + layering pass).
            Useful for validating the Python path or for layer-count metrics that
            the C++ path doesn't expose.
        calculate_fidelity: If True, compute fidelity between the original and Clifford+T circuits.
        return_intermediate: Request intermediate circuits from transpilers (recommended for fidelity).
        max_workers: Optional worker cap for the parallel PBC converter.
        clifford_output_path: If provided, save the Clifford+T circuit QASM to this path.
        pbc_output_prefix: If provided, save PBC layer/measurement artifacts with this prefix.
    """

    pipeline: PipelineName = "gs"
    gridsynth_precision: int = 3
    sk_recursion: int = 2
    layering_method: LayeringMethod = "v2"
    layering_max_checks: Optional[int] = None
    optimize_pbc: bool = False
    optimize_t_maxiter: int = 5
    prefer_cpp: bool = True
    use_nwqec_pbc: bool = True
    calculate_fidelity: bool = True
    return_intermediate: bool = True
    max_workers: Optional[int] = None
    clifford_output_path: Optional[str] = None
    pbc_output_prefix: Optional[str] = None


@dataclass
class PipelineResult:
    """Structured result from running a single pipeline."""

    pipeline: PipelineName
    clifford_t_circuit: QuantumCircuit
    pbc_circuit: QuantumCircuit
    clifford_stats: Dict[str, Any]
    pbc_stats: Dict[str, Any]
    fidelity: Optional[Dict[str, Any]]
    timings: Dict[str, float]
    parameters: Dict[str, Any]
    intermediate_circuit: Optional[QuantumCircuit] = None
    artifacts: Optional[Dict[str, str]] = None

    def to_dict(
        self,
        include_circuits: bool = False,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """Return a JSON-serializable view of the pipeline result."""
        data = {
            "pipeline": self.pipeline,
            "parameters": self.parameters,
            "timings": self.timings,
            "clifford_stats": self.clifford_stats,
            "pbc_stats": self.pbc_stats,
            "fidelity": self.fidelity,
        }

        if include_artifacts and self.artifacts:
            data["artifacts"] = self.artifacts

        if include_circuits:
            data["clifford_t_qasm"] = qasm2_dumps(self.clifford_t_circuit)
            data["pbc_qasm"] = qasm2_dumps(self.pbc_circuit)

        return data


@dataclass
class AnalysisResult:
    """Aggregated result for one or more pipelines run on a single input circuit."""

    input_path: Optional[str]
    original_qubits: int
    original_gates: int
    pipelines: Dict[str, PipelineResult]

    def to_dict(
        self,
        include_circuits: bool = False,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        return {
            "input_path": self.input_path,
            "original_qubits": self.original_qubits,
            "original_gates": self.original_gates,
            "pipelines": {
                name: result.to_dict(
                    include_circuits=include_circuits,
                    include_artifacts=include_artifacts,
                )
                for name, result in self.pipelines.items()
            },
        }


def _ensure_config_list(
    configs: PipelineConfig | Iterable[PipelineConfig],
) -> List[PipelineConfig]:
    if isinstance(configs, PipelineConfig):
        return [configs]
    return list(configs)


def _save_qasm(circuit: QuantumCircuit, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        qasm2_dump(circuit, handle)


def run_pipeline(circuit: QuantumCircuit, config: PipelineConfig) -> PipelineResult:
    """
    Execute a single pipeline on the provided circuit.

    Args:
        circuit: Input circuit to compile.
        config: Pipeline configuration controlling transpilation, PBC settings, and fidelity.

    Returns:
        PipelineResult describing the compiled circuit, statistics, and optional artifacts.
    """
    working_circuit = circuit.copy()
    timings: Dict[str, float] = {}
    artifacts: Dict[str, str] = {}
    parameters: Dict[str, Any] = {}

    # Step 1: Clifford+T synthesis
    ct_start = time.time()
    intermediate_circuit = None
    if config.pipeline == "gs":
        if config.return_intermediate:
            intermediate_circuit, clifford_t_circuit = (
                transpile_to_gridsynth_clifford_t(
                    working_circuit,
                    gridsynth_precision=config.gridsynth_precision,
                    return_intermediate=True,
                    prefer_cpp=config.prefer_cpp,
                )
            )
        else:
            clifford_t_circuit = transpile_to_gridsynth_clifford_t(
                working_circuit,
                gridsynth_precision=config.gridsynth_precision,
                return_intermediate=False,
                prefer_cpp=config.prefer_cpp,
            )
        parameters["gridsynth_precision"] = config.gridsynth_precision
    else:
        # Solovay-Kitaev pipeline
        if config.return_intermediate:
            intermediate_circuit, clifford_t_circuit = (
                transpile_to_solovay_kitaev_clifford_t(
                    working_circuit,
                    recursion_degree=config.sk_recursion,
                    return_intermediate=True,
                )
            )
        else:
            clifford_t_circuit = transpile_to_solovay_kitaev_clifford_t(
                working_circuit,
                recursion_degree=config.sk_recursion,
                return_intermediate=False,
            )
        parameters["sk_recursion_degree"] = config.sk_recursion

    timings["transpilation_clifford_t_time"] = time.time() - ct_start

    if config.clifford_output_path:
        _save_qasm(clifford_t_circuit, config.clifford_output_path)
        artifacts["clifford_t_qasm"] = config.clifford_output_path

    clifford_stats = analyze_clifford_t_circuit(
        clifford_t_circuit,
        gridsynth_precision_used=(
            config.gridsynth_precision if config.pipeline == "gs" else None
        ),
    )

    # Step 2: PBC conversion
    pbc_start = time.time()
    pbc_circuit, pbc_stats = convert_to_pbc_circuit(
        clifford_t_circuit.copy(),
        optimize_pbc=config.optimize_pbc,
        optimize_t_maxiter=config.optimize_t_maxiter,
        if_print_rpc=False,
        layering_method=config.layering_method,
        layering_max_checks=config.layering_max_checks,
        output_prefix=config.pbc_output_prefix,
        max_workers=config.max_workers,
        use_nwqec=config.use_nwqec_pbc,
    )
    timings["pbc_conversion_time"] = time.time() - pbc_start
    if config.pbc_output_prefix:
        artifacts.update(
            {
                "pbc_pre_opt_layers": f"{config.pbc_output_prefix}_pre_opt_tlayers.txt",
                "pbc_pre_opt_measurement_basis": f"{config.pbc_output_prefix}_pre_opt_measure_basis.txt",
                "pbc_post_opt_layers": f"{config.pbc_output_prefix}_post_opt_tlayers.txt",
                "pbc_post_opt_measurement_basis": f"{config.pbc_output_prefix}_post_opt_measure_basis.txt",
            }
        )

    pbc_analysis = analyze_pbc_circuit(pbc_circuit, pbc_conversion_stats=pbc_stats)
    combined_pbc_stats = {**pbc_stats, **pbc_analysis}

    # Step 3: Fidelity (optional)
    fidelity_result: Optional[Dict[str, Any]] = None
    if config.calculate_fidelity:
        # `calculate_circuit_fidelity` routes itself based on qubit count and
        # pipeline: small circuits use full-unitary process fidelity; large
        # circuits use per-Rz product fidelity (gridsynth-based for GS,
        # SK-based for SK via `rz_product_fidelity_sk`).
        fidelity_result = calculate_circuit_fidelity(
            working_circuit,
            clifford_t_circuit,
            gridsynth_precision=config.gridsynth_precision,
            sk_recursion_degree=(
                config.sk_recursion if config.pipeline == "sk" else None
            ),
            intermediate_qc=intermediate_circuit,
        )

    parameters.update(
        {
            "layering_method": config.layering_method,
            "layering_max_checks": config.layering_max_checks,
            "optimize_t_maxiter": config.optimize_t_maxiter,
            "max_workers": config.max_workers,
        }
    )

    result = PipelineResult(
        pipeline=config.pipeline,
        clifford_t_circuit=clifford_t_circuit,
        pbc_circuit=pbc_circuit,
        clifford_stats=clifford_stats,
        pbc_stats=combined_pbc_stats,
        fidelity=fidelity_result,
        timings=timings,
        parameters=parameters,
        intermediate_circuit=intermediate_circuit,
        artifacts=artifacts,
    )
    return result


def run_analysis(
    circuit: QuantumCircuit,
    configs: PipelineConfig | Sequence[PipelineConfig],
    source_path: Optional[str] = None,
) -> AnalysisResult:
    """
    Run one or more pipeline configurations on a circuit.

    Args:
        circuit: The input circuit to analyze.
        configs: A single PipelineConfig or a list of configurations to run.
        source_path: Optional path to the circuit, stored in the result metadata.

    Returns:
        AnalysisResult aggregating all pipeline runs.
    """
    config_list = _ensure_config_list(configs)
    pipelines: Dict[str, PipelineResult] = {}
    for cfg in config_list:
        pipelines[cfg.pipeline] = run_pipeline(circuit.copy(), cfg)

    return AnalysisResult(
        input_path=source_path,
        original_qubits=circuit.num_qubits,
        original_gates=len(circuit.data),
        pipelines=pipelines,
    )


def run_analysis_for_file(
    qasm_file: str, configs: PipelineConfig | Sequence[PipelineConfig]
) -> AnalysisResult:
    """
    Convenience helper to load a QASM file and run one or more pipelines.

    Args:
        qasm_file: Path to the QASM file to load.
        configs: Pipeline configuration(s) to execute.

    Returns:
        AnalysisResult for the loaded file.
    """
    circuit = load_qasm_circuit(qasm_file, is_file=True)
    return run_analysis(circuit, configs, source_path=qasm_file)


__all__ = [
    "AnalysisResult",
    "LayeringMethod",
    "PipelineConfig",
    "PipelineName",
    "PipelineResult",
    "run_analysis",
    "run_analysis_for_file",
    "run_pipeline",
]
