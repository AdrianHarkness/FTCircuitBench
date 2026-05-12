# ./ftcircuitbench/analyzer/pbc_analyzer.py
import collections
import re
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit


def parse_pbc_gate_name(gate_name: str):
    """
    Parses PBC gate names like 'RXYZ(pi/8)' or 'MeasXYZ'.
    Returns (type: 'rotation'|'measurement', pauli_string_in_name, params).
    pauli_string_in_name is the XYZ part from the gate name.
    Params are angle strings for rotation, None for measurement.
    """
    # Regex to capture the Pauli part (e.g., XYZ) and parameters (e.g., pi/8)
    # Assumes Pauli string in name consists of I, X, Y, Z
    rot_match = re.match(r"R([IXYZ]+)\(([^)]+)\)", gate_name)
    # Allow optional leading sign for measurement label: Meas+XYZ / Meas-XYZ
    meas_match = re.match(r"Meas([+-]?)([IXYZ]+)", gate_name)

    if rot_match:
        pauli_str_in_name = rot_match.group(1)
        angle_str = rot_match.group(2)
        return "rotation", pauli_str_in_name, [angle_str]
    elif meas_match:
        sign = meas_match.group(1) or "+"
        pauli_str_in_name = meas_match.group(2)
        return "measurement", f"{sign}{pauli_str_in_name}", None
    elif gate_name in [
        "barrier",
        "snapshot",
        "delay",
        "id",
    ]:  # Common non-operational Qiskit gates
        return "utility", gate_name, None
    return "unknown", gate_name, None


def count_total_pbc_operators(circuit: QuantumCircuit) -> int:
    """Count PBC operators (rotation + measurement), excluding barriers and utilities."""
    total = 0
    for instruction in circuit.data:
        op_name = instruction.operation.name
        op_type, _, _ = parse_pbc_gate_name(op_name)
        if op_type in ["rotation", "measurement"]:
            total += 1
    return total


def count_non_utility_ops(circuit: QuantumCircuit) -> int:
    """Count all gates excluding utility ops (barrier, snapshot, delay, id)."""
    utility = {"barrier", "snapshot", "delay", "id"}
    return sum(1 for inst in circuit.data if inst.operation.name not in utility)


def analyze_pbc_circuit(
    pbc_circuit: QuantumCircuit, pbc_conversion_stats: Optional[dict] = None
) -> dict:
    """
    Analyzes a quantum circuit assumed to be in Pauli Based Computation form.

    Args:
        pbc_circuit (QuantumCircuit): The input PBC circuit.
        pbc_conversion_stats (dict, optional): Statistics dictionary obtained from
            the `convert_to_pbc_circuit` function.

    Returns:
        dict: A dictionary containing analysis metrics.
    """
    stats = {}
    if pbc_conversion_stats:
        stats.update(pbc_conversion_stats)

    num_qubits_circuit = pbc_circuit.num_qubits  # Total qubits in the circuit
    stats["num_qubits"] = num_qubits_circuit

    rotation_pauli_weights = []  # Store weights of effective Paulis in rotation ops
    measurement_pauli_weights = (
        []
    )  # Store weights of effective Paulis in measurement ops

    num_rotation_ops = 0
    num_measurement_ops = 0
    num_utility_ops = 0
    unknown_ops: Dict[str, int] = collections.defaultdict(int)

    # For interaction graph
    interaction_counts_pbc: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    qubit_degree_pbc: Dict[int, int] = collections.defaultdict(
        int
    )  # For individual qubit involvement in multi-Q ops

    for instruction in pbc_circuit.data:
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits  # List of Qiskit Qubit objects gate acts on

        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        # The number of qubits this gate instance acts on
        num_op_qubits = len(qargs)

        if op_type == "rotation":
            num_rotation_ops += 1

            weight = num_op_qubits

            if len(pauli_str_in_name) != num_op_qubits:
                print(
                    f"Warning: Mismatch for rotation gate {op_name}. "
                    f"Pauli string in name '{pauli_str_in_name}' (len {len(pauli_str_in_name)}) "
                    f"vs. number of qubits gate acts on ({num_op_qubits})."
                )

            if weight > 0:  # Only consider actual Pauli operations
                rotation_pauli_weights.append(weight)

            if num_op_qubits > 1 and weight > 0:  # Multi-qubit Pauli rotation
                # Map Qiskit Qubit objects to their integer indices
                qubit_indices = sorted([pbc_circuit.find_bit(q).index for q in qargs])
                for i in range(len(qubit_indices)):
                    qubit_degree_pbc[qubit_indices[i]] += 1
                    for j in range(i + 1, len(qubit_indices)):
                        pair = (qubit_indices[i], qubit_indices[j])
                        interaction_counts_pbc[pair] += 1

        elif op_type == "measurement":
            num_measurement_ops += 1
            # pauli_str_in_name may include a leading sign; strip it for weight
            core = (
                pauli_str_in_name[1:]
                if pauli_str_in_name and pauli_str_in_name[0] in "+-"
                else pauli_str_in_name
            )
            weight = len(core)

            if len(core) != num_op_qubits:
                print(
                    f"Warning: Mismatch for measurement gate {op_name}. "
                    f"Pauli string in name '{core}' (len {len(core)}) "
                    f"vs. number of qubits gate acts on ({num_op_qubits})."
                )

            if weight > 0:
                measurement_pauli_weights.append(weight)

            if num_op_qubits > 1 and weight > 0:  # Multi-qubit Pauli measurement
                qubit_indices = sorted([pbc_circuit.find_bit(q).index for q in qargs])
                for i in range(len(qubit_indices)):
                    qubit_degree_pbc[
                        qubit_indices[i]
                    ] += 1  # Also count for measurement interactions
                    for j in range(i + 1, len(qubit_indices)):
                        pair = (qubit_indices[i], qubit_indices[j])
                        interaction_counts_pbc[
                            pair
                        ] += 1  # Measurement implies interaction

        elif op_type == "utility":
            num_utility_ops += 1

        else:  # unknown
            unknown_ops[op_name] += 1

    stats["pbc_t_operators"] = num_rotation_ops
    stats["pbc_measurement_operators"] = num_measurement_ops
    stats["pbc_total_operators"] = num_rotation_ops + num_measurement_ops
    if unknown_ops:
        stats["pbc_unknown_operators"] = dict(unknown_ops)

    # Merge rotation and measurement Pauli weights for unified statistics
    all_pauli_weights = rotation_pauli_weights + measurement_pauli_weights
    if all_pauli_weights:
        stats["pbc_avg_pauli_weight"] = np.mean(all_pauli_weights)
        stats["pbc_std_pauli_weight"] = np.std(all_pauli_weights)
        stats["pbc_max_pauli_weight"] = np.max(all_pauli_weights)
        stats["pbc_min_pauli_weight"] = np.min(all_pauli_weights)
        stats["pbc_pauli_weight_distribution"] = dict(
            collections.Counter(all_pauli_weights)
        )
    else:
        for key in [
            "pbc_avg_pauli_weight",
            "pbc_std_pauli_weight",
            "pbc_max_pauli_weight",
            "pbc_min_pauli_weight",
        ]:
            stats[key] = "Not computable: missing data"
        stats["pbc_pauli_weight_distribution"] = {}

    # Remove separate rotation/measurement pauli weight stats from output
    for key in [
        "pbc_avg_rotation_pauli_weight",
        "pbc_std_rotation_pauli_weight",
        "pbc_max_rotation_pauli_weight",
        "pbc_min_rotation_pauli_weight",
        "pbc_rotation_pauli_weight_distribution",
        "pbc_avg_measurement_pauli_weight",
        "pbc_std_measurement_pauli_weight",
        "pbc_max_measurement_pauli_weight",
        "pbc_min_measurement_pauli_weight",
        "pbc_measurement_pauli_weight_distribution",
    ]:
        if key in stats:
            del stats[key]

    stats["pbc_multi_qubit_operator_interaction_pairs"] = dict(interaction_counts_pbc)
    stats["pbc_qubit_interaction_degree"] = dict(qubit_degree_pbc)

    # Add standard deviation of the qubit interaction degree
    qubit_degrees = list(qubit_degree_pbc.values())
    if qubit_degrees:
        stats["pbc_avg_qubit_interaction_degree"] = np.mean(qubit_degrees)
        stats["pbc_std_qubit_interaction_degree"] = np.std(qubit_degrees)
        stats["pbc_max_qubit_interaction_degree"] = max(qubit_degrees)
    else:
        stats["pbc_avg_qubit_interaction_degree"] = (
            "Not computable: no multi-qubit operators"
        )
        stats["pbc_std_qubit_interaction_degree"] = (
            "Not computable: no multi-qubit operators"
        )
        stats["pbc_max_qubit_interaction_degree"] = (
            "Not computable: no multi-qubit operators"
        )

    # Add comprehensive interaction graph statistics
    try:
        interaction_graph_stats = get_interaction_graph_statistics(pbc_circuit)
        # Prefix the interaction graph statistics to distinguish them from basic stats
        for key, value in interaction_graph_stats.items():
            stats[f"pbc_interaction_graph_{key}"] = value
    except Exception as e:
        print(
            f"Warning: Could not compute comprehensive PBC interaction graph statistics: {e}"
        )
        # Add placeholder values for interaction graph statistics
        stats["pbc_interaction_graph_num_nodes"] = num_qubits_circuit
        stats["pbc_interaction_graph_num_edges"] = (
            len(interaction_counts_pbc) if interaction_counts_pbc else "Not computable"
        )
        stats["pbc_interaction_graph_graph_density"] = "Not computable"
        stats["pbc_interaction_graph_is_connected"] = "Not computable"
        stats["pbc_interaction_graph_num_connected_components"] = "Not computable"
        stats["pbc_interaction_graph_avg_degree"] = "Not computable"
        stats["pbc_interaction_graph_std_degree"] = "Not computable"
        stats["pbc_interaction_graph_min_degree"] = "Not computable"
        stats["pbc_interaction_graph_max_degree"] = "Not computable"
        stats["pbc_interaction_graph_clustering_coefficient"] = "Not computable"
        stats["pbc_interaction_graph_avg_shortest_path_length"] = "Not computable"
        stats["pbc_interaction_graph_diameter"] = "Not computable"
        stats["pbc_interaction_graph_degree_coefficient_of_variation"] = (
            "Not computable"
        )
        stats["pbc_interaction_graph_degree_gini_coefficient"] = "Not computable"

    total_multi_qubit_interactions = sum(interaction_counts_pbc.values())
    stats["pbc_total_multi_qubit_operator_applications_on_pairs"] = (
        total_multi_qubit_interactions
    )

    # If pre-optimization stats are present in pbc_conversion_stats, copy them into stats
    if pbc_conversion_stats:
        for key in pbc_conversion_stats:
            if key.startswith("pre_opt_"):
                stats[key] = pbc_conversion_stats[key]

    # Compute missing pre-opt stats if possible
    # 1. Avg Rotations per Layer (pre-opt)
    if (
        "pre_opt_t_gates" in stats
        and "pre_opt_rotation_layers" in stats
        and isinstance(stats["pre_opt_rotation_layers"], (int, float))
        and stats["pre_opt_rotation_layers"] > 0
    ):
        stats["pre_opt_avg_rotations_per_layer"] = (
            stats["pre_opt_t_gates"] / stats["pre_opt_rotation_layers"]
        )
    else:
        stats["pre_opt_avg_rotations_per_layer"] = "Not computable: missing data"

    # 2. Measurement Operators (pre-opt)
    if (
        "pre_opt_num_measurement_ops" not in stats
        and "pre_opt_measurement_operators" in stats
    ):
        stats["pre_opt_num_measurement_ops"] = stats["pre_opt_measurement_operators"]
    # If not present, set to 'Not computable: missing data'
    if "pre_opt_num_measurement_ops" not in stats:
        stats["pre_opt_num_measurement_ops"] = "Not computable: missing data"

    # 3. Pauli weight stats (pre-opt) - use existing operator pauli weight stats
    if "pre_opt_std_operator_pauli_weight" in stats and isinstance(
        stats["pre_opt_std_operator_pauli_weight"], (int, float)
    ):
        stats["pre_opt_std_pauli_weight"] = stats["pre_opt_std_operator_pauli_weight"]
    else:
        stats["pre_opt_std_pauli_weight"] = "Not computable: missing data"

    if "pre_opt_max_operator_pauli_weight" in stats and isinstance(
        stats["pre_opt_max_operator_pauli_weight"], (int, float)
    ):
        stats["pre_opt_max_pauli_weight"] = stats["pre_opt_max_operator_pauli_weight"]
    else:
        stats["pre_opt_max_pauli_weight"] = "Not computable: missing data"

    # 4. Qubit interaction degree (pre-opt)
    # Use the precomputed values if available and valid, otherwise fall back to old calculation
    pre_opt_avg_deg = stats.get("pre_opt_avg_qubit_interaction_degree", None)
    pre_opt_std_deg = stats.get("pre_opt_std_qubit_interaction_degree", None)
    if (
        pre_opt_avg_deg is not None
        and pre_opt_avg_deg != "Not computable: missing data"
        and pre_opt_std_deg is not None
        and pre_opt_std_deg != "Not computable: missing data"
    ):
        stats["pre_opt_avg_qubit_interaction_degree"] = pre_opt_avg_deg
        stats["pre_opt_std_qubit_interaction_degree"] = pre_opt_std_deg
    elif (
        "pre_opt_layer_occupation" in stats
        and isinstance(stats["pre_opt_layer_occupation"], list)
        and stats["pre_opt_layer_occupation"]
    ):
        # Flatten and compute mean/std
        flat_occupation = (
            np.concatenate(stats["pre_opt_layer_occupation"])
            if stats["pre_opt_layer_occupation"]
            else []
        )
        if len(flat_occupation) > 0:
            stats["pre_opt_avg_qubit_interaction_degree"] = np.mean(flat_occupation)
            stats["pre_opt_std_qubit_interaction_degree"] = np.std(flat_occupation)
        else:
            stats["pre_opt_avg_qubit_interaction_degree"] = (
                "Not computable: missing data"
            )
            stats["pre_opt_std_qubit_interaction_degree"] = (
                "Not computable: missing data"
            )
    else:
        stats["pre_opt_avg_qubit_interaction_degree"] = "Not computable: missing data"
        stats["pre_opt_std_qubit_interaction_degree"] = "Not computable: missing data"

    return stats


def generate_interaction_graph(
    circuit: QuantumCircuit,
    return_adjacency_matrix: bool = False,
    return_networkx_graph: bool = True,
) -> Union[nx.Graph, np.ndarray, Tuple[nx.Graph, np.ndarray]]:
    """
    Generate an interaction graph for a PBC circuit.

    This function analyzes the multi-qubit operator interactions in a PBC circuit and creates
    a network graph where:
    - Nodes represent qubits
    - Edges represent multi-qubit operator interactions (weighted by count)
    - Edge weights indicate the number of interactions between qubit pairs

    Args:
        circuit: The PBC quantum circuit to analyze
        return_adjacency_matrix: Whether to return adjacency matrix
        return_networkx_graph: Whether to return networkx graph object

    Returns:
        If both flags are True: (networkx_graph, adjacency_matrix)
        If only return_networkx_graph is True: networkx_graph
        If only return_adjacency_matrix is True: adjacency_matrix
        If both are False: networkx_graph (default)
    """
    # Create a graph
    G = nx.Graph()

    # Add nodes for each qubit
    num_qubits = circuit.num_qubits
    for i in range(num_qubits):
        G.add_node(i, label=f"q{i}")

    # Count interactions between qubits
    interaction_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    qubit_degree: Dict[int, int] = collections.defaultdict(int)

    for instruction in circuit.data:
        op = instruction.operation
        op_name = op.name
        qargs = instruction.qubits

        op_type, pauli_str_in_name, params = parse_pbc_gate_name(op_name)

        # Only consider rotation and measurement operators for interaction graph
        if op_type in ["rotation", "measurement"]:
            num_op_qubits = len(qargs)

            if num_op_qubits > 1:  # Multi-qubit operator
                # Map Qiskit Qubit objects to their integer indices
                qubit_indices = sorted([circuit.find_bit(q).index for q in qargs])

                # Count interactions between all pairs of qubits
                for i in range(len(qubit_indices)):
                    qubit_degree[qubit_indices[i]] += 1
                    for j in range(i + 1, len(qubit_indices)):
                        pair = (qubit_indices[i], qubit_indices[j])
                        interaction_counts[pair] += 1

    # Add edges to the graph
    for (q1, q2), count in interaction_counts.items():
        G.add_edge(q1, q2, weight=count, label=f"{count}")

    # Set node attributes for qubit degrees
    for i in range(num_qubits):
        G.nodes[i]["degree"] = qubit_degree.get(i, 0)

    # Generate adjacency matrix if requested
    adjacency_matrix = None
    if return_adjacency_matrix:
        adjacency_matrix = nx.adjacency_matrix(G, weight="weight").toarray()

    # Return based on flags
    if return_networkx_graph and return_adjacency_matrix:
        return G, adjacency_matrix
    elif return_adjacency_matrix:
        return adjacency_matrix
    else:
        return G


def get_interaction_graph_statistics(
    circuit: QuantumCircuit, graph: nx.Graph = None
) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the interaction graph of a PBC circuit.

    Args:
        circuit: The PBC quantum circuit to analyze
        graph: Optional pre-computed interaction graph

    Returns:
        Dictionary containing interaction graph statistics
    """
    if graph is None:
        graph = generate_interaction_graph(
            circuit, return_networkx_graph=True, return_adjacency_matrix=False
        )

    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    # Basic graph properties
    # Note: Graph density is not calculated for PBC circuits as it's not a relevant statistic
    # due to the potential for many interactions on the same edge
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_connected": nx.is_connected(graph) if num_edges > 0 else True,
        "num_connected_components": nx.number_connected_components(graph),
    }

    # Node degree statistics
    degrees = [d for n, d in graph.degree()]
    if degrees and any(
        d > 0 for d in degrees
    ):  # Check if there are any non-zero degrees
        stats.update(
            {
                "avg_degree": np.mean(degrees),
                "std_degree": np.std(degrees),
                "min_degree": min(degrees),
                "max_degree": max(degrees),
                "degree_distribution": dict(collections.Counter(degrees)),
            }
        )
    else:
        stats.update(
            {
                "avg_degree": "Not computable: no edges",
                "std_degree": "Not computable: no edges",
                "min_degree": "Not computable: no edges",
                "max_degree": "Not computable: no edges",
                "degree_distribution": {},
            }
        )

    # Edge weight statistics
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]
    if edge_weights:
        stats.update(
            {
                "avg_edge_weight": np.mean(edge_weights),
                "std_edge_weight": np.std(edge_weights),
                "min_edge_weight": min(edge_weights),
                "max_edge_weight": max(edge_weights),
                "total_interactions": sum(edge_weights),
                "edge_weight_distribution": dict(collections.Counter(edge_weights)),
            }
        )
    else:
        stats.update(
            {
                "avg_edge_weight": "Not computable: no edges",
                "std_edge_weight": "Not computable: no edges",
                "min_edge_weight": "Not computable: no edges",
                "max_edge_weight": "Not computable: no edges",
                "total_interactions": "Not computable: no edges",
                "edge_weight_distribution": {},
            }
        )

    # Graph centrality measures
    if num_nodes > 1 and num_edges > 0:
        try:
            # Clustering coefficient
            stats["clustering_coefficient"] = nx.average_clustering(graph)

            # Average shortest path length (only for connected graphs)
            if nx.is_connected(graph):
                stats["avg_shortest_path_length"] = nx.average_shortest_path_length(
                    graph
                )
            else:
                stats["avg_shortest_path_length"] = (
                    "Not computable: graph not connected"
                )

            # Diameter (only for connected graphs)
            if nx.is_connected(graph):
                stats["diameter"] = nx.diameter(graph)
            else:
                stats["diameter"] = "Not computable: graph not connected"

        except nx.NetworkXError:
            stats["clustering_coefficient"] = "Not computable"
            stats["avg_shortest_path_length"] = "Not computable"
            stats["diameter"] = "Not computable"
    else:
        stats["clustering_coefficient"] = "Not computable: insufficient edges"
        stats["avg_shortest_path_length"] = "Not computable: insufficient edges"
        stats["diameter"] = "Not computable: insufficient edges"

    # Modularity measures (Louvain method)
    if num_nodes > 1 and num_edges > 0:
        try:
            # Use networkx's Louvain implementation
            communities = nx.community.louvain_communities(graph)
            modularity = nx.community.modularity(graph, communities)

            stats["modularity"] = modularity
            stats["num_communities"] = len(communities)
            stats["community_sizes"] = [len(c) for c in communities]
            stats["communities"] = [list(c) for c in communities]

            # Calculate average community size and size variation
            if communities:
                community_sizes = [len(c) for c in communities]
                stats["avg_community_size"] = np.mean(community_sizes)
                stats["std_community_size"] = np.std(community_sizes)
                stats["min_community_size"] = min(community_sizes)
                stats["max_community_size"] = max(community_sizes)
            else:
                stats["avg_community_size"] = "Not computable"
                stats["std_community_size"] = "Not computable"
                stats["min_community_size"] = "Not computable"
                stats["max_community_size"] = "Not computable"

        except Exception as e:
            print(f"Warning: Could not compute Louvain modularity: {e}")
            stats["modularity"] = "Not computable"
            stats["num_communities"] = "Not computable"
            stats["community_sizes"] = "Not computable"
            stats["communities"] = "Not computable"
            stats["avg_community_size"] = "Not computable"
            stats["std_community_size"] = "Not computable"
            stats["min_community_size"] = "Not computable"
            stats["max_community_size"] = "Not computable"
    else:
        stats["modularity"] = "Not computable: insufficient edges"
        stats["num_communities"] = "Not computable: insufficient edges"
        stats["community_sizes"] = "Not computable: insufficient edges"
        stats["communities"] = "Not computable: insufficient edges"
        stats["avg_community_size"] = "Not computable: insufficient edges"
        stats["std_community_size"] = "Not computable: insufficient edges"
        stats["min_community_size"] = "Not computable: insufficient edges"
        stats["max_community_size"] = "Not computable: insufficient edges"

    return stats
