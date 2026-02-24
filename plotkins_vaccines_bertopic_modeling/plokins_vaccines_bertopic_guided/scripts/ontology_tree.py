#!/usr/bin/env python3
"""
Ontology Tree Generator from multiple BERTopic topic_tree_ids_labels_plotkins_vaccines.txt files
Left-to-right hierarchical layout using Graphviz (dot).

- Dummy ROOT node auto-added to attach any orphans (nodes with no parent).
- Colors: ROOT = orange, parents = blue, leaves = green.
- Edges have clear arrowheads (parent → child).
"""

import re
import csv
from dotenv import load_dotenv
import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------
# Parsing
# ---------------------------

TOPIC_RE = re.compile(r"Topic\s+(\d+)\s*\((.*?)\)")

def parse_line(line: str):
    """Extract (depth, topic_id, label) if line contains 'Topic <id> (<label>)'."""
    m = TOPIC_RE.search(line)
    if not m:
        return None
    topic_id = int(m.group(1))
    label = m.group(2).strip()
    depth = line.index("Topic")  # indent = depth proxy
    return depth, topic_id, label

def parse_tree_file(path: Path) -> Tuple[Set[Tuple[int,int]], Dict[int,str], Set[int]]:
    """Parse one topic tree file -> edges, labels, nodes."""
    edges: Set[Tuple[int,int]] = set()
    labels: Dict[int,str] = {}
    nodes: Set[int] = set()
    stack: List[Tuple[int,int]] = []  # (depth, node_id)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            parsed = parse_line(line)
            if not parsed:
                continue
            depth, node_id, label = parsed
            labels[node_id] = label
            nodes.add(node_id)

            while stack and stack[-1][0] >= depth:
                stack.pop()
            if stack:
                parent_id = stack[-1][1]
                if parent_id != node_id:
                    edges.add((parent_id, node_id))
            stack.append((depth, node_id))
    return edges, labels, nodes

# ---------------------------
# Merging
# ---------------------------

def merge_forests(paths: List[Path]) -> Tuple[Set[Tuple[int,int]], Dict[int,str], Set[int]]:
    merged_edges: Set[Tuple[int,int]] = set()
    merged_labels: Dict[int,str] = {}
    merged_nodes: Set[int] = set()
    for p in paths:
        e, lbl, nds = parse_tree_file(p)
        merged_edges |= e
        merged_labels.update(lbl)
        merged_nodes |= nds
    return merged_edges, merged_labels, merged_nodes

# ---------------------------
# Graph Drawing
# ---------------------------

def draw_tree(edges, labels, output_png="ontology_tree.png", output_svg="ontology_tree.svg"):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Ensure isolated labeled nodes are added as nodes
    for n in labels.keys():
        if n not in G:
            G.add_node(n)

    # -------- Dummy ROOT attachment for orphans --------
    # Orphans = nodes with in-degree 0 (no parent)
    orphans = [n for n in G.nodes() if G.in_degree(n) == 0]
    if orphans:
        DUMMY_ROOT = -1
        labels[DUMMY_ROOT] = "Vaccine Ontology"
        if DUMMY_ROOT not in G:
            G.add_node(DUMMY_ROOT)
        # Attach every orphan to the ROOT (excluding the root itself if present)
        for n in orphans:
            if n != DUMMY_ROOT:
                G.add_edge(DUMMY_ROOT, n)
    # ---------------------------------------------------

    # Optional pruning if too big
    MAX_NODES = 200
    if G.number_of_nodes() > MAX_NODES:
        deg = {n: (G.in_degree(n)+G.out_degree(n)) for n in G.nodes()}
        keep = set(sorted(deg, key=deg.get, reverse=True)[:MAX_NODES])
        G = G.subgraph(keep).copy()

    # Left-to-right hierarchical layout
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except Exception:
        pos = nx.spring_layout(G, k=1.5, seed=42)

    def wrap_label(s, width=25):
        return "\n".join(s[i:i+width] for i in range(0, len(s), width))

    safe_labels = {n: wrap_label(labels.get(n, str(n))) for n in G.nodes()}

    # Role-based colors: ROOT = orange, parents = blue, leaves = green
    parent_nodes = {u for u, _ in G.edges()}
    node_colors = []
    for n in G.nodes():
        if labels.get(n, "") == "ROOT" or (G.in_degree(n) == 0):  # dummy root (and any remaining absolute roots)
            node_colors.append("#FFA500")  # orange
        elif n in parent_nodes:
            node_colors.append("#4C9BE8")  # blue (has children)
        else:
            node_colors.append("#7ED957")  # green (leaf)

    plt.figure(figsize=(22, 14))
    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowstyle="-|>", arrowsize=20,
        width=1.2, edge_color="black"
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2200, node_color=node_colors,
        edgecolors="black", linewidths=0.8
    )
    nx.draw_networkx_labels(G, pos, labels=safe_labels, font_size=8, font_weight="bold")

    plt.title(
        "Ontology Tree (Parent → Child) with Dummy ROOT\n"
        "Orange = ROOT, Blue = Parent, Green = Leaf",
        fontsize=15, weight="bold"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_svg)
    print(f"Saved:\n  - {output_png}\n  - {output_svg}")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    load_dotenv()

    input_paths = os.getenv("ONTOLOGY_TREE_INPUTS")
    output_dir = os.getenv("ONTOLOGY_TREE_OUTPUT")

    if not input_paths:
        raise ValueError("Missing ONTOLOGY_TREE_INPUTS in .env")
    if not output_dir:
        raise ValueError("Missing ONTOLOGY_TREE_OUTPUT in .env")

    tree_files = [Path(p.strip()) for p in input_paths.split(",") if p.strip()]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Merging tree files:")
    for f in tree_files:
        print(" •", f)

    edges, labels, nodes = merge_forests(tree_files)

    # Save merged edges CSV
    csv_out = out_dir / "merged_edges.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "parent_label", "child_id", "child_label"])
        for p, c in sorted(edges):
            w.writerow([p, labels.get(p, ""), c, labels.get(c, "")])
    print("Saved CSV:", csv_out)

    # Draw ontology tree (PNG + SVG)
    out_png = out_dir / "ontology_tree.png"
    out_svg = out_dir / "ontology_tree.svg"
    draw_tree(edges, labels, output_png=str(out_png), output_svg=str(out_svg))
