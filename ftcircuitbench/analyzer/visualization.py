"""
Visualization utilities for FTCircuitBench.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit


def _save_plot_as_pdf(
    fig, circuit_name: Optional[str], plot_type: str, parent_folder: str = "figs"
) -> str:
    """
    Save the current plot as a PDF file in the figs folder.

    Filenames include both the circuit name and plot type to avoid collisions.

    Args:
        fig: The matplotlib figure to save
        circuit_name: Name of the circuit (e.g., 'adder_10q'); if None, falls back to 'circuit'
        plot_type: Type of plot (e.g., 'interaction_graph', 'weight_histogram')
        parent_folder: Folder to save in (default: 'figs')

    Returns:
        The filepath of the saved PDF.
    """
    base_name = (str(circuit_name).strip() or "circuit") if circuit_name else "circuit"
    safe_base = base_name.replace(" ", "_")
    safe_plot = (plot_type or "plot").replace(" ", "_")
    filename = f"{safe_base}_{safe_plot}.pdf"

    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    filepath = os.path.join(parent_folder, filename)
    fig.savefig(filepath, format="pdf", bbox_inches="tight")
    print(f"📄 Plot saved as: {filepath}")
    return filepath


def _create_stats_legend(
    stats_data: Dict[str, Any],
    fontsize: int = 25,
    frameon: bool = True,
    fancybox: bool = True,
    shadow: bool = False,
    framealpha: float = 0.3,
    position: Any = None,
) -> None:
    """
    Create and display a modular statistics legend on the current figure.

    Args:
        stats_data: Dictionary containing statistics to display
        position: Position of the legend (x, y) as fraction of figure
        fontsize: Font size for the text
        frameon: Whether to draw a frame around the legend
        fancybox: Whether to use a fancy box for the legend
        shadow: Whether to draw a shadow behind the legend
        framealpha: Transparency of the legend frame
    """
    # Create legend handles and labels
    handles = []
    labels = []

    for key, value in stats_data.items():
        if isinstance(value, float):
            label = f"{key}: {value:.3f}"
        elif isinstance(value, (int, str)):
            label = f"{key}: {value}"
        else:
            label = f"{key}: {value}"

        # Create a simple line handle for the legend
        handle = plt.Line2D([0], [0], color="none", linewidth=0)
        handles.append(handle)
        labels.append(label)

    # Create the legend
    legend = plt.legend(
        handles,
        labels,
        loc=position,
        frameon=frameon,
        fancybox=fancybox,
        shadow=shadow,
        framealpha=framealpha,
        handlelength=0,  # Hide the handle lines
        handletextpad=0,  # Remove padding between handle and text
        prop={"weight": "bold", "size": fontsize},
    )

    # Add the legend to the current axes
    plt.gca().add_artist(legend)


def show_clifford_t_interaction_graph(
    circuit: QuantumCircuit,
    title: str = "Clifford+T Circuit Interaction Graph",
    figsize: tuple = (14, 10),  # Increased from (12, 8)
    node_size: int = 1250,  # Increased from 500
    font_size: int = 25,  # Increased from 10
    font_weight: str = "bold",
    font_color: str = "white",
    edge_color: str = "black",
    alpha: float = 0.6,
    show_stats: bool = True,
    show_colorbar: bool = True,
    cmap: str = "viridis",  # Modern colormap without white
    name: Optional[str] = None,
    stats_fontsize: Optional[int] = None,
) -> None:
    """
    Create and display an interaction graph visualization for a Clifford+T quantum circuit.

    This function analyzes the two-qubit gate interactions in a Clifford+T circuit and creates
    a network graph where:
    - Nodes represent qubits (colored by interaction degree)
    - Edges represent two-qubit gate interactions (thickness proportional to count)

    Args:
        circuit: The Clifford+T quantum circuit to analyze
        title: Title for the plot
        figsize: Figure size as (width, height)
        node_size: Size of the nodes
        font_size: Font size for labels
        font_weight: Font weight for labels
        font_color: Color of the node labels
        edge_color: Color of the edges
        alpha: Transparency of edges
        show_stats: Whether to show statistics text box
        show_colorbar: Whether to show the colorbar
        cmap: Colormap for node colors
        name: Name for the circuit (used in PDF filename). If None, will be auto-extracted.
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes for each qubit
    num_qubits = circuit.num_qubits
    for i in range(num_qubits):
        G.add_node(i, label=f"q{i}")

    # Count interactions between qubits and collect operator stats
    interaction_counts: Dict[Any, int] = defaultdict(int)
    qubit_degree: Dict[int, int] = defaultdict(int)

    for instruction in circuit.data:
        qargs = instruction.qubits

        if len(qargs) == 2:
            q1_idx = circuit.find_bit(qargs[0]).index
            q2_idx = circuit.find_bit(qargs[1]).index

            # Add edge with weight (interaction count)
            edge = tuple(sorted((q1_idx, q2_idx)))
            interaction_counts[edge] += 1

            # Update qubit degrees
            qubit_degree[q1_idx] += 1
            qubit_degree[q2_idx] += 1

    # Add edges to the graph
    for (q1, q2), count in interaction_counts.items():
        G.add_edge(q1, q2, weight=count)

    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout with improved parameters for better spread
    pos = nx.spring_layout(G, k=3.0, seed=42)

    # Scale the positions to spread nodes out more
    if pos:
        # Get the current range of positions
        x_coords = [coord[0] for coord in pos.values()]
        y_coords = [coord[1] for coord in pos.values()]

        if x_coords and y_coords:
            # Scale factor to spread nodes out more
            scale_factor = 1.5

            # Apply scaling to spread nodes out
            for node in pos:
                pos[node] = (pos[node][0] * scale_factor, pos[node][1] * scale_factor)

    # Draw nodes
    node_colors = [qubit_degree.get(i, 0) for i in range(num_qubits)]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.get_cmap(cmap),
        node_size=node_size,
        alpha=1,
    )

    # Draw edges with thickness proportional to interaction count (normalized)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    # Normalize edge widths to a reasonable range (1-10)
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            # Normalize to range [1, 10] with logarithmic scaling for better visualization
            normalized_weights = []
            for weight in edge_weights:
                # Use log scaling to handle large variations in weights
                log_weight = np.log(weight + 1)  # +1 to avoid log(0)
                log_min = np.log(min_weight + 1)
                log_max = np.log(max_weight + 1)
                if log_max > log_min:
                    normalized = 1 + 9 * (log_weight - log_min) / (log_max - log_min)
                else:
                    normalized = 1
                normalized_weights.append(normalized)
        else:
            # All weights are the same, use uniform width
            normalized_weights = [3.0] * len(edge_weights)
    else:
        normalized_weights = []

    nx.draw_networkx_edges(
        G, pos, width=normalized_weights, edge_color=edge_color, alpha=alpha
    )

    # Add labels
    node_labels = {i: f"q{i}" for i in range(num_qubits)}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=font_size,
        font_weight=font_weight,
        font_color=font_color,
    )

    # plt.title(title, fontsize=16, fontweight="bold")

    # Create proper colorbar with normalization
    if show_colorbar and node_colors:
        norm = plt.Normalize(min(node_colors), max(node_colors))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Clifford+T Interaction Degree")
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.set_ylabel("Clifford+T Interaction Degree", fontsize=font_size)

    # Add statistics as text
    if show_stats:
        total_interactions = sum(interaction_counts.values())

        # Calculate mean and std of qubit interaction degrees
        degrees = list(qubit_degree.values())
        mean_degree = np.mean(degrees) if degrees else 0
        std_degree = np.std(degrees) if degrees else 0

        # Calculate community detection statistics
        community_stats = {}
        if len(G.edges()) > 0:
            try:
                communities = nx.community.louvain_communities(G)
                modularity = nx.community.modularity(G, communities)
                num_communities = len(communities)
                community_stats = {
                    "Louvain modularity": modularity,
                    "Number of communities": num_communities,
                }
            except Exception:
                community_stats = {
                    "Louvain modularity": "Not computable",
                    "Number of communities": "Not computable",
                }
        else:
            community_stats = {
                "Louvain modularity": "Not computable",
                "Number of communities": "Not computable",
            }

        stats_data = {
            "Total 2-qubit gates": total_interactions,
            "Max qubit degree": max(qubit_degree.values()) if qubit_degree else 0,
            "Mean ± std degree": f"{mean_degree:.1f} ± {std_degree:.1f}",
            **community_stats,
        }

        _create_stats_legend(stats_data, position="upper right", framealpha=0.5)

    plt.tight_layout()

    # Save the plot as PDF
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "interaction_graph")

    plt.show()

    # Print summary statistics
    print("✅ Clifford+T Interaction graph created!")
    print(f"   - Total 2-qubit gates: {sum(interaction_counts.values())}")
    print(f"   - Qubit interaction degrees: {dict(qubit_degree)}")


def show_pbc_interaction_graph(
    circuit: QuantumCircuit,
    title: str = "PBC Circuit Interaction Graph",
    figsize: tuple = (14, 10),  # Increased from (12, 8)
    node_size: int = 1250,  # Increased from 500
    font_size: int = 25,  # Increased from 10
    font_weight: str = "bold",
    font_color: str = "white",
    edge_color: str = "black",
    alpha: float = 0.6,
    show_stats: bool = True,
    show_colorbar: bool = True,
    cmap: str = "plasma",  # Modern colormap without white
    name: Optional[str] = None,
) -> None:
    """
    Create and display an interaction graph visualization for a PBC circuit.

    This function analyzes the multi-qubit operator interactions in a PBC circuit and creates
    a network graph where:
    - Nodes represent qubits (colored by interaction degree)
    - Edges represent multi-qubit operator interactions (thickness proportional to count)

    Args:
        circuit: The PBC quantum circuit to analyze
        title: Title for the plot
        figsize: Figure size as (width, height)
        node_size: Size of the nodes
        font_size: Font size for labels
        font_weight: Font weight for labels
        font_color: Color of the node labels
        edge_color: Color of the edges
        alpha: Transparency of edges
        show_stats: Whether to show statistics text box
        show_colorbar: Whether to show the colorbar
        cmap: Colormap for node colors
        name: Name for the circuit (used in PDF filename). If None, will be auto-extracted.
    """
    # Import the parse_pbc_gate_name function
    try:
        from .pbc_analyzer import parse_pbc_gate_name
    except ImportError:
        # Fallback if the module is not available
        def parse_pbc_gate_name(gate_name: str):
            """Fallback parser for PBC gate names."""
            import re

            rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
            meas_match = re.match(r"Meas([IXYZ]+)", gate_name)

            if rot_match:
                return "rotation", rot_match.group(1), [rot_match.group(2)]
            elif meas_match:
                return "measurement", meas_match.group(1), None
            elif gate_name in ["barrier", "snapshot", "delay", "id"]:
                return "utility", gate_name, None
            return "unknown", gate_name, None

    # Create a graph
    G = nx.Graph()

    # Add nodes for each qubit
    num_qubits = circuit.num_qubits
    for i in range(num_qubits):
        G.add_node(i, label=f"q{i}")

    # Count interactions between qubits and collect operator stats
    interaction_counts: Dict[Any, int] = defaultdict(int)
    qubit_degree: Dict[int, int] = defaultdict(int)
    total_pbc_ops: int = 0
    rotation_ops: int = 0
    measurement_ops: int = 0
    multi_qubit_ops: int = 0
    op_weights: List[int] = []

    for instruction in circuit.data:
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits

        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        # Only consider rotation and measurement operators for interaction graph
        if op_type in ["rotation", "measurement"]:
            total_pbc_ops += 1
            rotation_ops += 1 if op_type == "rotation" else 0
            measurement_ops += 1 if op_type == "measurement" else 0

            num_op_qubits = len(qargs)
            op_weights.append(num_op_qubits)
            if num_op_qubits > 1:
                multi_qubit_ops += 1

            # Map Qiskit Qubit objects to their integer indices
            qubit_indices = sorted([circuit.find_bit(q).index for q in qargs])

            # Count interactions between all pairs of qubits (for edge weights)
            for i in range(len(qubit_indices)):
                qubit_degree[qubit_indices[i]] += 1
                for j in range(i + 1, len(qubit_indices)):
                    pair = (qubit_indices[i], qubit_indices[j])
                    interaction_counts[pair] += 1

    # Add edges to the graph
    for (q1, q2), count in interaction_counts.items():
        G.add_edge(q1, q2, weight=count)

    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Use spring layout with improved parameters for better spread
    pos = nx.spring_layout(G, k=50.0, iterations=500, seed=42)
    # pos = nx.kamada_kawai_layout(G, scale=3.0)

    # Scale the positions to spread nodes out more
    if pos:
        # Get the current range of positions
        x_coords = [coord[0] for coord in pos.values()]
        y_coords = [coord[1] for coord in pos.values()]

        if x_coords and y_coords:
            # Scale factor to spread nodes out more
            scale_factor = 1.5

            # Apply scaling to spread nodes out
            for node in pos:
                pos[node] = (pos[node][0] * scale_factor, pos[node][1] * scale_factor)

    # Draw nodes
    node_colors = [qubit_degree.get(i, 0) for i in range(num_qubits)]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=plt.cm.get_cmap(cmap),
        node_size=node_size,
        alpha=1,
    )

    # Draw edges with thickness proportional to interaction count (normalized)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    # Normalize edge widths to a reasonable range (1-10)
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            # Normalize to range [1, 10] with logarithmic scaling for better visualization
            normalized_weights = []
            for weight in edge_weights:
                # Use log scaling to handle large variations in weights
                log_weight = np.log(weight + 1)  # +1 to avoid log(0)
                log_min = np.log(min_weight + 1)
                log_max = np.log(max_weight + 1)
                if log_max > log_min:
                    normalized = 1 + 9 * (log_weight - log_min) / (log_max - log_min)
                else:
                    normalized = 1
                normalized_weights.append(normalized)
        else:
            # All weights are the same, use uniform width
            normalized_weights = [3.0] * len(edge_weights)
    else:
        normalized_weights = []

    nx.draw_networkx_edges(
        G, pos, width=normalized_weights, edge_color=edge_color, alpha=alpha
    )

    # Add labels
    node_labels = {i: f"q{i}" for i in range(num_qubits)}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=font_size,
        font_weight=font_weight,
        font_color=font_color,
    )

    # plt.title(title, fontsize=16, fontweight="bold")

    # Create proper colorbar with normalization
    if show_colorbar and node_colors:
        norm = plt.Normalize(min(node_colors), max(node_colors))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="PBC Interaction Degree")
        cbar.ax.tick_params(labelsize=font_size)
        cbar.ax.set_ylabel("PBC Interaction Degree", fontsize=font_size)

    # Add statistics as text
    if show_stats:
        # Calculate mean and std of qubit interaction degrees
        degrees = list(qubit_degree.values())
        mean_degree = np.mean(degrees) if degrees else 0
        std_degree = np.std(degrees) if degrees else 0

        # Mean/std operator weight (active qubits per op)
        mean_weight = np.mean(op_weights) if op_weights else 0
        std_weight = np.std(op_weights) if op_weights else 0

        # Calculate community detection statistics
        community_stats = {}
        if len(G.edges()) > 0:
            try:
                communities = nx.community.louvain_communities(G)
                modularity = nx.community.modularity(G, communities)
                num_communities = len(communities)
                community_stats = {
                    "Louvain modularity": modularity,
                    "Number of communities": num_communities,
                }
            except Exception:
                community_stats = {
                    "Louvain modularity": "Not computable",
                    "Number of communities": "Not computable",
                }
        else:
            community_stats = {
                "Louvain modularity": "Not computable",
                "Number of communities": "Not computable",
            }

        stats_data = {
            "Total PBC operators": total_pbc_ops,
            "Rotation ops": rotation_ops,
            "Measurement ops": measurement_ops,
            "Multi-qubit ops": multi_qubit_ops,
            "Mean ± std weight": f"{mean_weight:.2f} ± {std_weight:.2f}",
            "Max qubit degree": max(qubit_degree.values()) if qubit_degree else 0,
            "Mean ± std degree": f"{mean_degree:.1f} ± {std_degree:.1f}",
            **community_stats,
        }

        _create_stats_legend(stats_data, position="upper right", framealpha=0.5)

    plt.tight_layout()

    # Save the plot as PDF
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "pbc_interaction_graph")

    plt.show()

    # Print summary statistics
    print("✅ PBC Interaction graph created!")
    print(
        f"   - Total PBC operators: {total_pbc_ops} (rot: {rotation_ops}, meas: {measurement_ops})"
    )
    print(f"   - Multi-qubit ops: {multi_qubit_ops}")
    print(f"   - Mean weight: {mean_weight:.2f} ± {std_weight:.2f}")


def show_operator_weight_histogram(
    circuit: QuantumCircuit,
    title: str = "PBC Operator Weight Distribution",
    figsize: tuple = (10, 6),
    bins: Optional[int] = None,
    color: str = "lightcoral",
    alpha: float = 0.8,
    edgecolor: str = "black",
    show_stats: bool = True,
    name: Optional[str] = None,
    font_size: int = 20,
) -> None:
    """
    Create and display a bar plot of rotation and measurement operator weights in PBC circuits.

    This function analyzes PBC circuits to find rotation and measurement operators
    and creates a bar plot showing the distribution of their weights, where weight
    is the number of non-identity Paulis in the Pauli string (e.g., "XYZ" has weight 3,
    "XIZ" has weight 2, "III" has weight 0).

    Each bar represents the exact integer weight value, making it clear that
    the weight corresponds to the center of each bar.

    Args:
        circuit: The PBC quantum circuit to analyze
        title: Title for the plot
        figsize: Figure size as (width, height)
        bins: Not used (kept for backward compatibility)
        color: Color of the histogram bars
        alpha: Transparency of the bars
        edgecolor: Color of the bar edges
        show_stats: Whether to show statistics text box
        name: Name for the circuit (used in PDF filename). If None, will be auto-extracted.
    """
    # Import the parse_pbc_gate_name function
    try:
        from .pbc_analyzer import parse_pbc_gate_name
    except ImportError:
        # Fallback if the module is not available
        def parse_pbc_gate_name(gate_name: str):
            """Fallback parser for PBC gate names."""
            import re

            rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
            meas_match = re.match(r"Meas([IXYZ]+)", gate_name)

            if rot_match:
                return "rotation", rot_match.group(1), [rot_match.group(2)]
            elif meas_match:
                return "measurement", meas_match.group(1), None
            elif gate_name in ["barrier", "snapshot", "delay", "id"]:
                return "utility", gate_name, None
            print(f"Unknown operator: {gate_name}")
            return "unknown", gate_name, None

    # Collect weights of rotation and measurement operators
    rotation_weights = []
    measurement_weights = []

    for instruction in circuit.data:
        op = instruction.operation
        op_name = op.name

        # Parse as PBC operator
        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        if op_type == "rotation":
            # For PBC rotations, weight is the number of non-Identity Paulis
            # Count non-identity Paulis in the Pauli string
            weight = sum(1 for p_char in pauli_str_in_name if p_char != "I")
            rotation_weights.append(weight)
        elif op_type == "measurement":
            # For PBC measurements, weight is the number of non-Identity Paulis
            # Count non-identity Paulis in the Pauli string
            weight = sum(1 for p_char in pauli_str_in_name if p_char != "I")
            measurement_weights.append(weight)
        # Skip utility and unknown operators for PBC analysis

    # Combine rotation and measurement weights
    all_weights = rotation_weights + measurement_weights

    if not all_weights:
        print("⚠️  No PBC rotation or measurement operators found in the circuit.")
        print(
            "   This function is designed for PBC circuits with operators like RXYZ(pi/8) and MeasXYZ."
        )
        return

    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Count occurrences of each weight
    weight_counts: Dict[Any, int] = {}
    for weight in all_weights:
        weight_counts[weight] = weight_counts.get(weight, 0) + 1

    # Get sorted weights and their counts
    weights = sorted(weight_counts.keys())
    counts = [weight_counts[weight] for weight in weights]

    # Create bar plot (clearer than histogram for discrete integer data)
    ax.bar(weights, counts, color=color, alpha=alpha, edgecolor=edgecolor, linewidth=1)

    # Customize the plot
    ax.set_xlabel("Operator Weight", fontsize=font_size)
    ax.set_ylabel("Count", fontsize=font_size)
    ax.grid(True, alpha=0.3)

    # Set y-tick label font size
    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)

    # Smart x-axis tick spacing for better readability
    num_weights = len(weights)
    if num_weights <= 20:
        # For small numbers of weights, show all
        ax.set_xticks(weights)
        ax.set_xticklabels([str(w) for w in weights], fontsize=font_size)
    else:
        # For larger numbers, show every nth weight
        step = max(1, num_weights // 20)  # Show max 20 ticks
        tick_weights = weights[::step]

        # Always include the first and last weights
        if weights[0] not in tick_weights:
            tick_weights.insert(0, weights[0])
        if weights[-1] not in tick_weights:
            tick_weights.append(weights[-1])

        # Sort to ensure proper order
        tick_weights.sort()

        ax.set_xticks(tick_weights)
        ax.set_xticklabels([str(w) for w in tick_weights], fontsize=font_size)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)

    # Add statistics as legend
    if show_stats:
        mean_weight = np.mean(all_weights)
        std_weight = np.std(all_weights)
        stats_data = {
            "Rotation ops": len(rotation_weights),
            "Measurement ops": len(measurement_weights),
            "Mean ± std weight": f"{mean_weight:.2f} ± {std_weight:.2f}",
        }

        _create_stats_legend(stats_data, fontsize=20)

    plt.tight_layout()

    # Save the plot as PDF
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "pbc_operator_weight_histogram")

    plt.show()

    # Print summary statistics
    print("✅ PBC operator weight bar plot created!")
    print(f"   - Total PBC operators: {len(all_weights)}")
    print(f"   - Rotation operators: {len(rotation_weights)}")
    print(f"   - Measurement operators: {len(measurement_weights)}")
    print(f"   - Weight range: {min(all_weights)} to {max(all_weights)}")
    print(f"   - Mean weight: {np.mean(all_weights):.2f} ± {np.std(all_weights):.2f}")


def show_qubit_pbc_operations_plot(
    circuit: QuantumCircuit,
    title: str = "PBC Operations per Qubit",
    figsize: tuple = (12, 6),
    color: str = "steelblue",
    alpha: float = 0.7,
    edgecolor: str = "black",
    show_stats: bool = True,
    show_grid: bool = True,
    name: Optional[str] = None,
    font_size: int = 20,
) -> None:
    """
    Create and display a bar plot showing the number of PBC operations per qubit.

    This function analyzes PBC circuits to count rotation and measurement operations
    that each qubit participates in, and creates a bar plot where:
    - X-axis: Qubit number (0, 1, 2, ...)
    - Y-axis: Total number of PBC operations (rotations + measurements) per qubit

    Args:
        circuit: The PBC quantum circuit to analyze
        title: Title for the plot
        figsize: Figure size as (width, height)
        color: Color of the bars
        alpha: Transparency of the bars
        edgecolor: Color of the bar edges
        show_stats: Whether to show statistics text box
        show_grid: Whether to show grid lines
        name: Name for the circuit (used in PDF filename). If None, will be auto-extracted.
    """
    # Import the parse_pbc_gate_name function
    try:
        from .pbc_analyzer import parse_pbc_gate_name
    except ImportError:
        # Fallback if the module is not available
        def parse_pbc_gate_name(gate_name: str):
            """Fallback parser for PBC gate names."""
            import re

            rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
            meas_match = re.match(r"Meas([IXYZ]+)", gate_name)

            if rot_match:
                return "rotation", rot_match.group(1), [rot_match.group(2)]
            elif meas_match:
                return "measurement", meas_match.group(1), None
            elif gate_name in ["barrier", "snapshot", "delay", "id"]:
                return "utility", gate_name, None
            return "unknown", gate_name, None

    # Count PBC operations per qubit
    qubit_operation_counts: Dict[int, int] = defaultdict(int)
    qubit_rotation_counts: Dict[int, int] = defaultdict(int)
    qubit_measurement_counts: Dict[int, int] = defaultdict(int)
    # Whole-circuit totals (count each operator once)
    total_ops: int = 0
    total_rot: int = 0
    total_meas: int = 0

    for instruction in circuit.data:
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits

        # Parse as PBC operator
        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        if op_type in ["rotation", "measurement"]:
            # Whole-circuit totals (each instruction counts once)
            total_ops += 1
            if op_type == "rotation":
                total_rot += 1
            else:
                total_meas += 1
            # Count operations for each qubit this gate acts on
            for qubit in qargs:
                qubit_idx = circuit.find_bit(qubit).index
                qubit_operation_counts[qubit_idx] += 1

                if op_type == "rotation":
                    qubit_rotation_counts[qubit_idx] += 1
                elif op_type == "measurement":
                    qubit_measurement_counts[qubit_idx] += 1

    if not qubit_operation_counts:
        print("⚠️  No PBC rotation or measurement operators found in the circuit.")
        print(
            "   This function is designed for PBC circuits with operators like RXYZ(pi/8) and MeasXYZ."
        )
        return

    # Create the visualization
    fig, ax = plt.subplots(figsize=figsize)

    # Get all qubit indices and their operation counts
    num_qubits = circuit.num_qubits
    qubit_indices = list(range(num_qubits))
    operation_counts = [qubit_operation_counts[i] for i in qubit_indices]

    # Create bar plot
    ax.bar(
        qubit_indices,
        operation_counts,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=1,
        label="Total PBC Operations",
    )

    # Customize the plot
    ax.set_xlabel("Qubit Number", fontsize=font_size)
    ax.set_ylabel("Number of PBC Operations", fontsize=font_size)

    if show_grid:
        ax.grid(True, alpha=0.3, axis="y")

    # Set y-tick label font size
    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)

    # Smart x-axis tick spacing for better readability
    if num_qubits <= 20:
        # For small numbers of qubits, show all
        ax.set_xticks(qubit_indices)
        ax.set_xticklabels([f"q{i}" for i in qubit_indices], fontsize=font_size)
    else:
        # For larger numbers, show every nth qubit
        step = max(1, num_qubits // 20)  # Show max 20 ticks
        tick_indices = list(range(0, num_qubits, step))

        # Always include the first and last qubits
        if 0 not in tick_indices:
            tick_indices.insert(0, 0)
        if num_qubits - 1 not in tick_indices:
            tick_indices.append(num_qubits - 1)

        # Sort to ensure proper order
        tick_indices.sort()

        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"q{i}" for i in tick_indices], fontsize=font_size)

        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=font_size)

    # Add statistics as legend
    if show_stats:
        # Note: totals below are whole-circuit counts, not per-qubit sums
        max_operations = max(operation_counts) if operation_counts else 0
        min_operations = min(operation_counts) if operation_counts else 0
        mean_operations = np.mean(operation_counts) if operation_counts else 0
        std_operations = np.std(operation_counts) if operation_counts else 0

        stats_data = {
            "Max operations per qubit": max_operations,
            "Min operations per qubit": min_operations,
            "Mean ± std operations": f"{mean_operations:.1f} ± {std_operations:.1f}",
        }

        _create_stats_legend(stats_data, fontsize=20)

    plt.tight_layout()

    # Save the plot as PDF
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "pbc_operations_per_qubit")

    plt.show()

    # Print summary statistics
    print("✅ PBC operations per qubit plot created!")
    print(f"   - Total PBC operations: {total_ops}")
    print(f"   - Total rotations: {total_rot}")
    print(f"   - Total measurements: {total_meas}")
    print(
        f"   - Active qubits: {sum(1 for count in operation_counts if count > 0)}/{num_qubits}"
    )
    print(f"   - Operations range: {min(operation_counts)} to {max(operation_counts)}")
    print(
        f"   - Mean operations per qubit: {np.mean(operation_counts):.1f} ± {np.std(operation_counts):.1f}"
    )


def get_interaction_statistics(circuit: QuantumCircuit) -> Dict[str, Any]:
    """
    Get interaction statistics for a quantum circuit.

    Args:
        circuit: The quantum circuit to analyze

    Returns:
        Dictionary containing interaction statistics
    """
    interaction_counts: Dict[Any, int] = defaultdict(int)
    qubit_degree: Dict[int, int] = defaultdict(int)

    for instruction in circuit.data:
        qargs = instruction.qubits

        if len(qargs) == 2:
            q1_idx = circuit.find_bit(qargs[0]).index
            q2_idx = circuit.find_bit(qargs[1]).index

            # Add edge with weight (interaction count)
            edge = tuple(sorted((q1_idx, q2_idx)))
            interaction_counts[edge] += 1

            # Update qubit degrees
            qubit_degree[q1_idx] += 1
            qubit_degree[q2_idx] += 1

    total_interactions = sum(interaction_counts.values())
    num_qubits = circuit.num_qubits

    return {
        "total_two_qubit_gates": total_interactions,
        "qubit_interaction_degree": dict(qubit_degree),
        "interaction_counts": dict(interaction_counts),
        "max_qubit_degree": max(qubit_degree.values()) if qubit_degree else 0,
        "num_qubits": num_qubits,
    }


def plot_pbc_pauli_weight_binned_bands(
    circuit: QuantumCircuit,
    bins: int = 256,
    figsize: tuple = (12, 3),
    line_color: str = "#2ca02c",
    band_color: str = "#2ca02c",
    band_alpha: float = 0.2,
    name: Optional[str] = None,
) -> None:
    """
    Visualize Pauli operator weight over operator index using x-binning and
    percentile bands to handle very long x-ranges.

    - X is divided into `bins` equally-sized index bins across the full operator index range
    - For each bin, we compute median weight and the interquartile range (25th–75th percentiles)
    - We plot the median as a line and shade the IQR band
    """

    # Helper to parse PBC gate names (only to filter operator types)
    try:
        from .pbc_analyzer import parse_pbc_gate_name
    except ImportError:
        import re

        def parse_pbc_gate_name(gate_name: str):
            rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
            meas_match = re.match(r"Meas([IXYZ]+)", gate_name)
            if rot_match:
                return "rotation", rot_match.group(1), [rot_match.group(2)]
            elif meas_match:
                return "measurement", meas_match.group(1), None
            elif gate_name in ["barrier", "snapshot", "delay", "id"]:
                return "utility", gate_name, None
            return "unknown", gate_name, None

    # Collect weights in order
    weights: List[int] = []
    for instruction in circuit.data:
        op_name = instruction.operation.name
        op_type, _, _ = parse_pbc_gate_name(op_name)
        if op_type not in ["rotation", "measurement"]:
            continue
        weights.append(len(instruction.qubits))

    if not weights:
        print(
            "⚠️  No PBC rotation or measurement operators found to plot weight bands."
        )
        return

    n = len(weights)

    # Build bins along the index axis
    bins = max(8, int(bins))
    edges = np.linspace(1, n + 1, bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    medians = []
    q25 = []
    q75 = []
    x_centers = []

    w = np.asarray(weights)
    for i in range(bins):
        left = int(np.floor(edges[i]))
        right = int(np.floor(edges[i + 1]))
        if right <= left:
            continue
        seg = w[left - 1 : right - 1]  # indices are 1-based in edges
        if seg.size == 0:
            continue
        medians.append(float(np.median(seg)))
        q25.append(float(np.percentile(seg, 25)))
        q75.append(float(np.percentile(seg, 75)))
        x_centers.append(bin_centers[i])

    if not x_centers:
        print("⚠️  Binning produced no data; reduce number of bins.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        x_centers, q25, q75, color=band_color, alpha=band_alpha, linewidth=0
    )
    ax.plot(x_centers, medians, color=line_color, linewidth=1.8)
    ax.set_xlabel("Operator Index", fontsize=14)
    ax.set_ylabel("Operator Weight", fontsize=14)
    ax.grid(True, alpha=0.25, axis="y")

    # X ticks: at most ~10 to keep clean
    ax.locator_params(axis="x", nbins=10)

    plt.tight_layout()
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "pbc_weight_bands")
    plt.show()

    print(
        f"✅ Saved Pauli weight binned-bands plot for '{circuit_name}'. Operators binned: {len(x_centers)}"
    )


def plot_clifford_t_tgate_heatmap(
    circuit: QuantumCircuit,
    title: str = "T-gate density over time and qubits (Clifford+T)",
    figsize: tuple = (12, 3),
    cmap: str = "magma",
    name: Optional[str] = None,
    time_bins: Optional[int] = None,
    use_layers: bool = True,
) -> None:
    """
    Visualize where T/Tdg gates occur across qubits (y) and time/op index (x) for a
    Clifford+T circuit by producing a 2D colormap. Non-T operations are ignored.

    Steps:
      - Scan circuit.data, collect (op_index, qubit_index) for every T/Tdg gate
      - Bin op_index into a fixed number of bins across full length (default 512)
      - Aggregate counts per (qubit, time_bin)
      - Plot as an image with color intensity representing T density
    """
    # Collect T events
    events = []  # (time_idx, qubit_idx)
    if not use_layers:
        op_idx = 0
        for instr in circuit.data:
            instruction = instr.operation.name
            if instruction in ("barrier", "snapshot", "delay", "id"):
                continue
            op_idx += 1
            if instruction not in ("t", "tdg"):
                continue
            for q in instr.qubits:
                qidx = circuit.find_bit(q).index
                events.append((op_idx, qidx))
        max_idx = op_idx
    else:  # use layers
        from qiskit.converters import circuit_to_dag, dag_to_circuit

        dag = circuit_to_dag(circuit)
        layer_idx = 0
        for layer in dag.layers():
            layer_circ = (
                dag_to_circuit(layer["graph"])
                if isinstance(layer, dict)
                else dag_to_circuit(layer.graph)
            )
            for gate in layer_circ.data:
                instruction = gate.operation.name
                if instruction not in ("t", "tdg"):
                    continue
                for q in gate.qubits:
                    qidx = layer_circ.find_bit(q).index
                    events.append((layer_idx + 1, qidx))
            layer_idx += 1
        max_idx = layer_idx

    if not events:
        print("⚠️  No T/Tdg gates found in the provided Clifford+T circuit.")
        return

    num_qubits = circuit.num_qubits

    # Choose number of time bins based on size (or use user-provided)
    time_bins = int(time_bins) if time_bins is not None else max_idx // 100

    edges = np.linspace(1, max_idx + 1, time_bins + 1)

    # Accumulate counts per (qubit, time_bin)
    heat = np.zeros((num_qubits, time_bins), dtype=int)
    for idx, q in events:
        b = np.searchsorted(edges, idx, side="right") - 1
        b = max(0, min(time_bins - 1, b))
        heat[q, b] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
    )
    ax.set_xlabel("Operator layer (binned)", fontsize=14)
    ax.set_ylabel("Qubit index", fontsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("T/Tdg count", fontsize=14)

    plt.tight_layout()
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "tgate_heatmap")
    plt.show()

    print(
        f"✅ Saved T-gate heatmap for '{circuit_name}'. Bins: {time_bins}, Layers: {max_idx}"
    )


def plot_pbc_operator_heatmap(
    circuit: QuantumCircuit,
    figsize: tuple = (12, 3),
    cmap: str = "plasma",
    name: Optional[str] = None,
    time_bins: Optional[int] = None,
) -> None:
    """
    Visualize where PBC rotation/measurement operators occur across qubits (y)
    and operator index (x) by producing a 2D colormap. Rotation and measurement
    operators are treated the same and simply counted.

    Steps:
      - Scan circuit.data, consider only rotation/measurement operators
      - Assign each such operator an increasing index (1..N)
      - Bin the index axis into a fixed number of bins
      - For each operator, increment all participating qubits in that time bin
      - Plot the resulting (qubit, time_bin) counts as an image
    """
    # Helper to parse PBC gate names
    try:
        from .pbc_analyzer import parse_pbc_gate_name
    except ImportError:
        import re

        def parse_pbc_gate_name(gate_name: str):
            rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
            meas_match = re.match(r"Meas([IXYZ]+)", gate_name)
            if rot_match:
                return "rotation", rot_match.group(1), [rot_match.group(2)]
            elif meas_match:
                return "measurement", meas_match.group(1), None
            elif gate_name in ["barrier", "snapshot", "delay", "id"]:
                return "utility", gate_name, None
            return "unknown", gate_name, None

    # First pass to count PBC operators and collect nothing else to keep memory small
    pbc_op_count = 0
    for instruction in circuit.data:
        op_name = instruction.operation.name
        op_type, _, _ = parse_pbc_gate_name(op_name)
        if op_type in ["rotation", "measurement"]:
            pbc_op_count += 1

    if pbc_op_count == 0:
        print("⚠️  No PBC rotation or measurement operators found in the circuit.")
        return

    # Determine binning
    max_idx = pbc_op_count
    time_bins = int(time_bins) if time_bins is not None else max_idx // 100
    time_bins = max(1, time_bins)

    edges = np.linspace(1, max_idx + 1, time_bins + 1)

    # Accumulate counts per (qubit, time_bin) in a streaming second pass
    num_qubits = circuit.num_qubits
    heat = np.zeros((num_qubits, time_bins), dtype=int)

    current_idx = 0
    for instruction in circuit.data:
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits
        op_type, _, _ = parse_pbc_gate_name(op_name)
        if op_type not in ["rotation", "measurement"]:
            continue
        current_idx += 1
        # Determine bin for this operator index
        b = np.searchsorted(edges, current_idx, side="right") - 1
        b = max(0, min(time_bins - 1, b))
        # Increment all participating qubits
        for qubit in qargs:
            qidx = circuit.find_bit(qubit).index
            heat[qidx, b] += 1

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        heat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
    )
    ax.set_xlabel("Operator index (binned)", fontsize=14)
    ax.set_ylabel("Qubit index", fontsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PBC operator count", fontsize=14)

    plt.tight_layout()
    circuit_name = name or getattr(circuit, "name", None)
    _save_plot_as_pdf(fig, circuit_name, "pbc_operator_heatmap")
    plt.show()

    print(
        f"✅ Saved PBC operator heatmap for '{circuit_name}'. Bins: {time_bins}, Operators: {max_idx}"
    )
