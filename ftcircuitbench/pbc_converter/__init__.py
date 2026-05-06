# ./ftcircuitbench/pbc_converter/__init__.py (Corrected)
"""
Pauli Based Computation (PBC) converter for FTCircuitBench.
Implements logic to convert Clifford+T circuits to PBC.
"""

from .pbc_circuit_reader import (
    analyze_pbc_file_content,
    parse_pauli_string,
    print_pbc_file_summary,
    read_combined_pbc_file,
    validate_pbc_file,
)
from .pbc_circuit_saver import save_pbc_layers_txt, save_pbc_measurement_basis_txt
from .pbc_generator import convert_to_pbc_circuit  # Only export the main function
from .pbm import PBM, Rotation
from .r_pauli_circ import RotationPauliCirc
from .tab_gate import TableauForGate, TableauPauliBasis

__all__ = [
    "TableauForGate",
    "TableauPauliBasis",
    "RotationPauliCirc",
    "Rotation",
    "PBM",
    "convert_to_pbc_circuit",
    "save_pbc_layers_txt",
    "save_pbc_measurement_basis_txt",
    "read_combined_pbc_file",
    "parse_pauli_string",
    "analyze_pbc_file_content",
    "validate_pbc_file",
    "print_pbc_file_summary",
]
