"""
dag.py
Builds and visualises the causal DAG for ZeraMatumizi.
Defines causal relationships between risk factors and substance use disorder
progression in Kenya.
"""

import os
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Output folder for DAG visualisations
REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(REPORTS_PATH, exist_ok=True)


def build_causal_dag() -> nx.DiGraph:
    """
    Build the ZeraMatumizi causal DAG.

    Node types:
    - Upstream determinants: county-level socioeconomic factors
    - Proximal determinants: individual-level risk factors
    - Mediators: substance use initiation
    - Outcome: disorder progression
    - Instruments: variables used for IV analysis
    - Confounders: variables that affect multiple nodes

    Returns a directed graph representing causal relationships.
    """
    G = nx.DiGraph()

    # --- Define all nodes with metadata ---

    # Upstream determinants (county-level)
    upstream = [
        "Poverty",
        "Unemployment",
        "Illicit_Brew_Proximity",
        "Facility_Distance",
        "HIV_Prevalence",
    ]

    # Proximal determinants (individual-level)
    proximal = [
        "Peer_Substance_Use",
        "Age_of_Initiation",
        "School_Dropout",
        "Mental_Health_Comorbidity",
        "Family_History",
    ]

    # Confounders
    confounders = [
        "Age",
        "Gender",
        "SES_Composite",
    ]

    # Mediator
    mediators = [
        "Substance_Use_Initiation",
    ]

    # Outcome
    outcomes = [
        "Disorder_Progression",
    ]

    # Final outcomes
    final_outcomes = [
        "Mortality",
        "Crime",
        "School_Failure",
    ]

    # Add all nodes
    for node in upstream:
        G.add_node(node, type="upstream")
    for node in proximal:
        G.add_node(node, type="proximal")
    for node in confounders:
        G.add_node(node, type="confounder")
    for node in mediators:
        G.add_node(node, type="mediator")
    for node in outcomes:
        G.add_node(node, type="outcome")
    for node in final_outcomes:
        G.add_node(node, type="final_outcome")

    # --- Define causal edges (arrows) ---

    edges = [
        # Upstream → Proximal
        ("Poverty", "Unemployment"),
        ("Poverty", "School_Dropout"),
        ("Poverty", "Mental_Health_Comorbidity"),
        ("Unemployment", "Peer_Substance_Use"),
        ("Unemployment", "Mental_Health_Comorbidity"),
        ("HIV_Prevalence", "Mental_Health_Comorbidity"),

        # Upstream → Mediator (direct effects)
        ("Illicit_Brew_Proximity", "Substance_Use_Initiation"),
        ("Facility_Distance", "Disorder_Progression"),

        # Proximal → Mediator
        ("Peer_Substance_Use", "Substance_Use_Initiation"),
        ("Age_of_Initiation", "Substance_Use_Initiation"),
        ("School_Dropout", "Substance_Use_Initiation"),
        ("Mental_Health_Comorbidity", "Substance_Use_Initiation"),
        ("Family_History", "Substance_Use_Initiation"),

        # Confounders → Mediator
        ("Age", "Substance_Use_Initiation"),
        ("Gender", "Substance_Use_Initiation"),
        ("SES_Composite", "Substance_Use_Initiation"),

        # Confounders → Outcome
        ("Age", "Disorder_Progression"),
        ("Gender", "Disorder_Progression"),
        ("SES_Composite", "Disorder_Progression"),

        # Mediator → Outcome
        ("Substance_Use_Initiation", "Disorder_Progression"),

        # Outcome → Final outcomes
        ("Disorder_Progression", "Mortality"),
        ("Disorder_Progression", "Crime"),
        ("Disorder_Progression", "School_Failure"),
    ]

    G.add_edges_from(edges)

    return G


def validate_dag(G: nx.DiGraph) -> bool:
    """
    Validates that the DAG has no cycles (must be acyclic).
    Returns True if valid, raises error if cycles detected.
    """
    if nx.is_directed_acyclic_graph(G):
        print("✓ DAG validation passed — no cycles detected")
        return True
    else:
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"✗ DAG contains cycles: {cycles}")


def print_dag_summary(G: nx.DiGraph):
    """Prints a summary of the DAG structure."""
    print("\n--- Causal DAG Summary ---")
    print(f"Total nodes:  {G.number_of_nodes()}")
    print(f"Total edges:  {G.number_of_edges()}")
    print(f"\nNodes by type:")

    node_types = {}
    for node, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types.setdefault(t, []).append(node)

    for t, nodes in node_types.items():
        print(f"  {t}: {', '.join(nodes)}")

    print(f"\nKey causal paths into Disorder_Progression:")
    for predecessor in G.predecessors("Disorder_Progression"):
        print(f"  {predecessor} → Disorder_Progression")
    print("---------------------------\n")


def save_interactive_dag(G: nx.DiGraph):
    """
    Saves an interactive HTML DAG visualisation using PyVis.
    Open the HTML file in a browser to explore the causal graph.
    """
    ensure_directories()

    # Colour scheme by node type
    colours = {
        "upstream":      "#E74C3C",   # Red
        "proximal":      "#E67E22",   # Orange
        "confounder":    "#9B59B6",   # Purple
        "mediator":      "#3498DB",   # Blue
        "outcome":       "#E91E63",   # Pink
        "final_outcome": "#2C3E50",   # Dark
    }

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True
    )

    # Add nodes with colours
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        colour = colours.get(node_type, "#95A5A6")
        net.add_node(
            node,
            label=node.replace("_", " "),
            color=colour,
            size=25,
            title=f"Type: {node_type}"
        )

    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target, arrows="to", color="#AAAAAA")

    # Save
    output_path = os.path.join(REPORTS_PATH, "causal_dag.html")
    net.save_graph(output_path)
    print(f"✓ Interactive DAG saved: {output_path}")
    print(f"  Open this file in your browser to explore the causal graph!")


def save_static_dag(G: nx.DiGraph):
    """Saves a static PNG image of the DAG."""
    ensure_directories()

    colours = {
        "upstream":      "#E74C3C",
        "proximal":      "#E67E22",
        "confounder":    "#9B59B6",
        "mediator":      "#3498DB",
        "outcome":       "#E91E63",
        "final_outcome": "#2C3E50",
    }

    node_colours = [
        colours.get(G.nodes[n].get("type", "unknown"), "#95A5A6")
        for n in G.nodes()
    ]

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colours,
        node_size=2000,
        font_size=7,
        font_color="white",
        font_weight="bold",
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
    )

    # Legend
    legend = [
        mpatches.Patch(color="#E74C3C", label="Upstream determinant"),
        mpatches.Patch(color="#E67E22", label="Proximal determinant"),
        mpatches.Patch(color="#9B59B6", label="Confounder"),
        mpatches.Patch(color="#3498DB", label="Mediator"),
        mpatches.Patch(color="#E91E63", label="Outcome"),
        mpatches.Patch(color="#2C3E50", label="Final outcome"),
    ]
    plt.legend(handles=legend, loc="upper left", fontsize=8)
    plt.title("ZeraMatumizi Causal DAG\nSubstance Use Disorder Progression in Kenya",
              fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    output_path = os.path.join(REPORTS_PATH, "causal_dag.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Static DAG image saved: {output_path}")


if __name__ == "__main__":
    print("Building ZeraMatumizi Causal DAG...")
    G = build_causal_dag()
    validate_dag(G)
    print_dag_summary(G)
    save_static_dag(G)
    save_interactive_dag(G)
    print("\n✓ Causal DAG module complete!")