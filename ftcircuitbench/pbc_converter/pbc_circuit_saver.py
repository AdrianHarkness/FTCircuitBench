# ftcircuitbench/pbc_converter/pbc_circuit_saver.py
import os
from typing import List

from .tab_gate import (  # Or TableauPauliBasis if t_layers elements are that
    TableauForGate,
)


def _tableau_to_pauli_strings(tableau_obj: TableauForGate) -> List[str]:
    """Converts a TableauForGate object into a list of Pauli strings."""
    if not tableau_obj or tableau_obj.stab_counts == 0:
        return []
    return [tableau_obj.readout(i) for i in range(tableau_obj.stab_counts)]


def save_pbc_layers_txt(t_layers: List[TableauForGate], filepath: str):
    """Saves T-gate layers to a text file.
    Each layer is explicitly marked.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if not t_layers:
            f.write("NO T-GATE LAYERS\n")
            return
        for i, layer_tab in enumerate(t_layers):
            f.write(f"LAYER {i}\n")
            if not layer_tab or layer_tab.stab_counts == 0:  # Handle empty layers
                f.write("  (empty layer)\n")
            else:
                pauli_strings = _tableau_to_pauli_strings(layer_tab)
                for p_str in pauli_strings:
                    f.write(f"  {p_str}\n")
            f.write("\n")
    # Remove verbose output
    # print(f"Saved PBC T-layers to: {filepath}")


def save_pbc_measurement_basis_txt(measure_tab: TableauForGate, filepath: str):
    """Saves the measurement basis to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        if not measure_tab or measure_tab.stab_counts == 0:
            f.write("NO MEASUREMENT BASIS\n")
            return
        pauli_strings = _tableau_to_pauli_strings(measure_tab)
        for p_str in pauli_strings:
            f.write(f"{p_str}\n")
    # Remove verbose output
    # print(f"Saved PBC measurement basis to: {filepath}")
