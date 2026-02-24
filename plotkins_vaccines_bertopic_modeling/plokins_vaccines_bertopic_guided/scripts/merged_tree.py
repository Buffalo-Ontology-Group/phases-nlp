import os
import re
import csv
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # optional

# ---------------------------
# Parsing
# ---------------------------

TOPIC_RE = re.compile(r"Topic\s+(\d+)\s*\((.*?)\)")

def parse_line(line: str):
    """Return (depth, topic_id, label) for '... Topic <id> (<label>)' lines."""
    if 'Topic' not in line:
        return None
    m = TOPIC_RE.search(line)
    if not m:
        return None
    topic_id = int(m.group(1))
    label = m.group(2).strip()
    depth = line.index('Topic')  # count of chars before 'Topic' as a depth proxy
    return depth, topic_id, label

def parse_tree_file(path: Path) -> Tuple[Set[Tuple[int, int]], Dict[int, str], Set[int]]:
    """Parse one ASCII tree into (edges, labels, nodes)."""
    edges: Set[Tuple[int, int]] = set()
    labels: Dict[int, str] = {}
    nodes: Set[int] = set()
    stack: List[Tuple[int, int]] = []  # (depth, node_id)

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip('\n')
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
# Merging / utilities
# ---------------------------

def merge_forests(paths: List[str]) -> Tuple[Set[Tuple[int,int]], Dict[int,str], Set[int]]:
    merged_edges: Set[Tuple[int,int]] = set()
    merged_labels: Dict[int,str] = {}
    merged_nodes: Set[int] = set()
    for p in paths:
        e, lbl, nds = parse_tree_file(Path(p))
        merged_edges |= e
        merged_labels.update(lbl)
        merged_nodes |= nds
    return merged_edges, merged_labels, merged_nodes

def build_adj(edges: Set[Tuple[int,int]]) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = defaultdict(list)
    for p, c in edges:
        adj[p].append(c)
    for k in adj:
        adj[k].sort()
    return adj

def find_roots(edges: Set[Tuple[int,int]], all_nodes: Set[int]) -> List[int]:
    """Robust root finder with fallbacks to avoid empty output."""
    if not all_nodes:
        return []

    indeg = Counter({n: 0 for n in all_nodes})
    for _, c in edges:
        indeg[c] += 1

    # Primary: parents that never appear as children
    children = {c for _, c in edges}
    parents  = {p for p, _ in edges}
    roots = sorted(list(parents - children))
    if roots:
        return roots

    # Fallback 1: nodes with indegree 0
    zero_in = sorted([n for n, d in indeg.items() if d == 0])
    if zero_in:
        return zero_in

    # Fallback 2: nodes with minimal indegree
    min_in = min(indeg.values()) if indeg else 0
    return sorted([n for n, d in indeg.items() if d == min_in])

def write_edges_csv(edges: Set[Tuple[int,int]], labels: Dict[int,str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_id", "parent_label", "child_id", "child_label"])
        for p, c in sorted(edges):
            w.writerow([p, labels.get(p, ""), c, labels.get(c, "")])

# ---------------------------
# ASCII tree writers
# ---------------------------

def _label_of(n: int, labels: Dict[int,str], max_len=90) -> str:
    s = labels.get(n, f"Topic {n}")
    return s if len(s) <= max_len else s[:max_len-1] + "…"

def print_ascii_with_pipes(adj: Dict[int, List[int]], labels: Dict[int, str], roots: List[int]):
    """Console print using '│' and '■──' style; cycle-safe."""
    for r in roots:
        stack = [(r, 0, set())]  # node, depth, path-set
        while stack:
            node, depth, path = stack.pop()
            prefix = ("│    " * depth) if depth > 0 else ""
            if node in path:
                print(f"{prefix}↻ Topic {node} (cycle)")
                continue
            print(f"{prefix}■── Topic {node} ({_label_of(node, labels)})")
            new_path = set(path); new_path.add(node)
            # push children in reverse to keep left-to-right order
            for ch in reversed(adj.get(node, [])):
                stack.append((ch, depth + 1, new_path))

def write_ascii_with_pipes_to_file(adj: Dict[int, List[int]], labels: Dict[int, str], roots: List[int], out_path: Path):
    """File write using '│' and '■──' style; cycle-safe; never empty."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        if not roots or len(roots) < len(labels):
            f.write("(Using all nodes as roots since some may be disconnected.)\n")
            roots = sorted(labels.keys())
            
        for r in roots:
            stack = [(r, 0, set())]  # node, depth, path-set
            while stack:
                node, depth, path = stack.pop()
                prefix = ("│    " * depth) if depth > 0 else ""
                if node in path:
                    f.write(f"{prefix}↻ Topic {node} (cycle)\n")
                    continue
                f.write(f"{prefix}■── Topic {node} ({_label_of(node, labels)})\n")
                new_path = set(path); new_path.add(node)
                for ch in reversed(adj.get(node, [])):
                    stack.append((ch, depth + 1, new_path))

# ---------------------------
# .env helpers
# ---------------------------

def str_to_bool(s: str, default=False) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"1","true","yes","y","on"}

def load_paths_from_env(prefix="TREE_FILE_PATH_") -> List[str]:
    """Collect TREE_FILE_PATH_1..N (numeric order) or fallback to TREE_FILES list."""
    numbered = []
    pat = re.compile(rf'^{re.escape(prefix)}(\d+)$')
    for key, val in os.environ.items():
        m = pat.match(key)
        if m and val.strip():
            idx = int(m.group(1))
            numbered.append((idx, val.strip()))
    if numbered:
        paths = [p for _, p in sorted(numbered, key=lambda x: x[0])]
    else:
        csv_env = os.getenv("TREE_FILES", "")
        paths = [p.strip() for p in csv_env.split(",") if p.strip()]

    if not paths:
        raise ValueError(
            "No tree files specified. Define TREE_FILE_PATH_1..N in .env "
            "or set TREE_FILES as a comma-separated list."
        )
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        msg = "Missing file(s):\n" + "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(msg)
    return paths

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Load .env if available
    if load_dotenv is not None:
        load_dotenv()

    # Inputs
    tree_paths = load_paths_from_env(prefix="TREE_FILE_PATH_")
    print(f"Loaded {len(tree_paths)} tree files:")
    for p in tree_paths:
        print("  •", p)

    # Merge
    merged_edges, merged_labels, merged_nodes = merge_forests(tree_paths)
    adj = build_adj(merged_edges)

    # Find roots (robust), else add synthetic root if requested/needed
    roots = find_roots(merged_edges, merged_nodes)
    add_root_env = str_to_bool(os.getenv("SYNTHETIC_ROOT"), default=False)

    if (not roots and merged_nodes) or (add_root_env and len(roots) > 1):
        synthetic = -1
        merged_labels[synthetic] = "SYNTHETIC ROOT"
        # attach anchors: nodes with minimal indegree
        indeg = Counter({n: 0 for n in merged_nodes})
        for _, c in merged_edges:
            indeg[c] += 1
        min_in = min(indeg.values()) if indeg else 0
        anchors = sorted([n for n, d in indeg.items() if d == min_in and n != synthetic])
        for a in anchors:
            merged_edges.add((synthetic, a))
        adj = build_adj(merged_edges)
        roots = [synthetic]
        print(f"Added synthetic root (-1). Anchors attached: {len(anchors)}")

    print(f"Root count used for ASCII: {len(roots)}")

    # Console preview
    print("\n=== MERGED TREE(S) ===")
    print_ascii_with_pipes(adj, merged_labels, roots)

    # Save outputs
    results_folder = os.getenv("RESULTS_MERGED_TREE", "./results_merged_tree")
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    edges_csv = Path(results_folder) / "merged_edges.csv"
    txt_file  = Path(results_folder) / "merged_tree.txt"

    write_edges_csv(merged_edges, merged_labels, edges_csv)
    write_ascii_with_pipes_to_file(adj, merged_labels, roots, txt_file)

    print(f"\nSaved:\n  - {edges_csv}\n  - {txt_file}")
