import collections
from typing import Any, Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit

# For modularity, we might need a graph library like networkx
# For simplicity, let's start with basic interaction counting
# import networkx as nx # Optional, for more advanced graph metrics

CLIFFORD_GATE_NAMES = {
    "h",
    "s",
    "sdg",
    "x",
    "y",
    "z",
    "cx",
    "cz",
    "swap",
    "id",
    "barrier",
    "snapshot",
    "delay",
    "reset",
}
# 'id', 'barrier', 'snapshot', 'delay', 'reset' are non-operational or utility.

T_GATE_NAMES = {"t", "tdg"}


def analyze_clifford_t_circuit(
    circuit: QuantumCircuit,
    gridsynth_precision_used: Optional[
        int
    ] = None,  # Optional: if known from compilation
) -> dict:
    """
    Analyzes a quantum circuit assumed to be in Clifford+T form.

    Args:
        circuit (QuantumCircuit): The input circuit.
        gridsynth_precision_used (int, optional): The precision (number of digits)
            used by Gridsynth if it was part of the compilation. Defaults to None.

    Returns:
        dict: A dictionary containing analysis metrics.
    """
    stats: Dict[str, Any] = {}
    ops_counts = collections.Counter(
        instruction.operation.name for instruction in circuit.data
    )

    # 1. Number of T gates (and Tdg gates)
    num_t_gates = ops_counts.get("t", 0)
    num_tdg_gates = ops_counts.get("tdg", 0)
    stats["t_count"] = num_t_gates
    stats["tdg_count"] = num_tdg_gates
    if isinstance(num_t_gates, int) and isinstance(num_tdg_gates, int):
        stats["total_t_family_count"] = num_t_gates + num_tdg_gates
    else:
        stats["total_t_family_count"] = "Not computable: missing data"

    # 2. Precision compiled to (if provided)
    stats["compilation_precision_digits"] = gridsynth_precision_used

    # 3. Modularity / Interaction Graph
    # Generate the interaction graph and get adjacency matrix if needed
    interaction_graph = generate_interaction_graph(
        circuit, return_adjacency_matrix=False, return_networkx_graph=True
    )
    num_qubits = circuit.num_qubits

    # Extract interaction counts from the graph edge weights
    interaction_counts: Dict[Tuple[Any, Any], int] = {}
    total_interactions = 0
    qubit_degree: Dict[Any, int] = collections.defaultdict(int)

    # Reconstruct interaction_counts and qubit_degree from graph
    for u, v, data in interaction_graph.edges(data=True):  # type: ignore[union-attr]
        weight = data.get("weight", 0)
        interaction_counts[(u, v)] = weight
        total_interactions += weight
        qubit_degree[
            u
        ] += weight  # Note: logic in original code added 1 for each interaction?
        # In original code:
        # interaction_counts[pair] += 1
        # qubit_degree[q1] += 1
        # qubit_degree[q2] += 1
        # So qubit degree is number of interaction events (gates), not sum of weights if weight means something else.
        # But generate_interaction_graph sets weight = count.
        # So if we iterate edges, we are iterating unique pairs.
        # To match original logic exactly:
        # Original: count interactions. Graph edges have weight = count.
        # So total_two_qubit_gates = sum(weights).

    # Wait, the original qubit_degree logic:
    # for each gate: degree[q1] += 1, degree[q2] += 1.
    # So degree[q] is total number of 2Q gates touching q.
    # Graph node degree (in networkx) is number of neighbors (unweighted) or weighted sum?
    # nx.degree is unweighted unless weight argument provided.
    # The original stats["qubit_interaction_degree"] seems to want the total number of gates.
    # Which corresponds to weighted degree in the graph.

    # Let's recalculate qubit_degree from the graph weights to be safe and consistent.
    qubit_degree = dict(interaction_graph.degree(weight="weight"))  # type: ignore[union-attr]

    # Fill in zeros for isolated qubits
    for i in range(num_qubits):
        if i not in qubit_degree:
            qubit_degree[i] = 0

    stats["two_qubit_gate_interaction_pairs"] = interaction_counts
    stats["total_two_qubit_gates"] = total_interactions
    stats["qubit_interaction_degree"] = qubit_degree

    if num_qubits > 1 and total_interactions > 0:
        # Density: E / E_max.
        # Note: Original code used sum(interaction_counts.values()) which is total interactions (gates),
        # divided by max_possible_edges.
        # This is actually "weighted density" or "interaction density".
        # Standard graph density is Num_Edges / Max_Edges.
        # The code explicitly calculates: sum(interaction_counts.values()) / max_possible_edges.
        max_possible_edges = (num_qubits * (num_qubits - 1)) / 2
        stats["interaction_graph_density"] = (
            total_interactions / max_possible_edges
            if max_possible_edges > 0
            else "Not computable: only one qubit"
        )

        qubit_degrees = list(qubit_degree.values())
        stats["avg_qubit_interaction_degree"] = np.mean(qubit_degrees)
        stats["std_qubit_interaction_degree"] = np.std(qubit_degrees)
        stats["max_qubit_interaction_degree"] = max(qubit_degrees)
    else:
        stats["interaction_graph_density"] = (
            "Not computable: only one qubit or no two-qubit gates"
        )
        stats["avg_qubit_interaction_degree"] = "Not computable: no two-qubit gates"
        stats["std_qubit_interaction_degree"] = "Not computable: no two-qubit gates"
        stats["max_qubit_interaction_degree"] = "Not computable: no two-qubit gates"

    # Add comprehensive interaction graph statistics
    try:
        interaction_graph_stats = get_interaction_graph_statistics(
            circuit, graph=interaction_graph
        )
        # Prefix the interaction graph statistics to distinguish them from basic stats
        for key, value in interaction_graph_stats.items():
            if key == "graph_density":
                # We already calculated a "density" above, but get_interaction_graph_statistics
                # also returns a "graph_density" (weighted).
                # The logic above (lines 89-95 in original) matches the logic in get_interaction_graph_statistics (lines 313-314).
                # So we can just take it from the stats.
                stats["interaction_graph_density"] = value
            else:
                stats[f"interaction_graph_{key}"] = value
    except Exception as e:
        print(
            f"Warning: Could not compute comprehensive interaction graph statistics: {e}"
        )
        # Add placeholder values for interaction graph statistics
        stats["interaction_graph_num_nodes"] = num_qubits
        stats["interaction_graph_num_edges"] = stats.get(
            "total_two_qubit_gates", "Not computable"
        )
        stats["interaction_graph_is_connected"] = "Not computable"
        stats["interaction_graph_num_connected_components"] = "Not computable"
        stats["interaction_graph_avg_degree"] = "Not computable"
        stats["interaction_graph_std_degree"] = "Not computable"
        stats["interaction_graph_min_degree"] = "Not computable"
        stats["interaction_graph_max_degree"] = "Not computable"
        stats["interaction_graph_clustering_coefficient"] = "Not computable"
        stats["interaction_graph_avg_shortest_path_length"] = "Not computable"
        stats["interaction_graph_diameter"] = "Not computable"
        stats["interaction_graph_degree_coefficient_of_variation"] = "Not computable"
        stats["interaction_graph_degree_gini_coefficient"] = "Not computable"

    # Simple modularity thought: Average degree vs max degree?
    # Or perhaps analyze connected components if we build a graph.
    # For now, density is a good start without external libraries.

    # 4. Clifford Count (and other gate counts)
    clifford_count = 0
    other_gate_count = 0
    detailed_clifford_counts: Dict[str, int] = collections.defaultdict(int)

    for op_name, count in ops_counts.items():
        if op_name in T_GATE_NAMES:
            continue  # Already counted
        elif op_name in CLIFFORD_GATE_NAMES:
            clifford_count += count
            detailed_clifford_counts[op_name] += count
        else:
            # This case should ideally not happen if the circuit is truly C+T
            print(f"Warning: Unexpected gate '{op_name}' found in Clifford+T analysis.")
            other_gate_count += count

    stats["clifford_gate_count"] = clifford_count
    stats["detailed_clifford_counts"] = dict(detailed_clifford_counts)
    if other_gate_count > 0:
        stats["unexpected_gate_count"] = other_gate_count

    # 5. Circuit Depth (overall, and T-depth)
    stats["depth"] = circuit.depth()

    # T-depth: Depth of circuit considering only T/Tdg gates and 2-qubit Clifford gates
    # This requires a more careful DAG traversal or custom depth calculation.
    # As a simpler proxy: number of layers containing T gates if we could schedule perfectly.
    # For a more accurate T-depth, one would typically use a scheduler.
    # Qiskit's depth() is based on longest path of any gate.

    # Placeholder for T-depth - this is non-trivial
    # For now, we can count layers that *contain* T-gates if we abstract layers.
    # A simple proxy: count T gates on each qubit
    t_gates_on_qubit: Dict[int, int] = collections.defaultdict(int)
    for instruction in circuit.data:
        op_name = instruction.operation.name
        if op_name in T_GATE_NAMES:
            for qbit in instruction.qubits:
                t_gates_on_qubit[circuit.find_bit(qbit).index] += 1
    stats["t_gates_per_qubit"] = dict(t_gates_on_qubit)
    t_gates_counts = list(t_gates_on_qubit.values())
    if t_gates_counts:
        stats["avg_t_gates_per_qubit"] = np.mean(t_gates_counts)
        stats["std_t_gates_per_qubit"] = np.std(t_gates_counts)
    else:
        stats["avg_t_gates_per_qubit"] = "Not computable: no T-gates"
        stats["std_t_gates_per_qubit"] = "Not computable: no T-gates"
    stats["max_t_gates_on_any_qubit"] = (
        max(t_gates_on_qubit.values())
        if t_gates_on_qubit
        else "Not computable: no T-gates"
    )

    # 6. Total Gate Count
    stats["total_gate_count"] = sum(ops_counts.values())
    stats["num_qubits"] = num_qubits

    return stats


def generate_interaction_graph(
    circuit: QuantumCircuit,
    return_adjacency_matrix: bool = False,
    return_networkx_graph: bool = True,
) -> Union[nx.Graph, np.ndarray, Tuple[nx.Graph, np.ndarray]]:
    """
    Generate an interaction graph for a Clifford+T circuit.

    This function analyzes the two-qubit gate interactions in a circuit and creates
    a network graph where:
    - Nodes represent qubits
    - Edges represent two-qubit gate interactions (weighted by count)
    - Edge weights indicate the number of interactions between qubit pairs

    Args:
        circuit: The Clifford+T quantum circuit to analyze
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
    interaction_counts: Dict[Tuple[int, ...], int] = collections.defaultdict(int)
    qubit_degree: Dict[int, int] = collections.defaultdict(int)

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
    Get comprehensive statistics about the interaction graph of a Clifford+T circuit.

    Args:
        circuit: The Clifford+T quantum circuit to analyze
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
    # Calculate weighted graph density: sum of edge weights / max possible edges
    if num_nodes > 1:
        max_possible_edges = (num_nodes * (num_nodes - 1)) / 2
        if num_edges > 0:
            total_edge_weight = sum(graph[u][v]["weight"] for u, v in graph.edges())
            weighted_density = total_edge_weight / max_possible_edges
        else:
            weighted_density = 0.0
    else:
        weighted_density = 0.0  # Single node has no edges

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "graph_density": weighted_density,
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
