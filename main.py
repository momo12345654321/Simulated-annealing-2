"""
Experiment runner: the repo's simulated annealing, then kcoloring removing one
color at a time (see pipeline.py).

Edit the settings below and run:  python main.py

Every combination of the settings is run once per seed. Each run is saved as
soon as it finishes:
  experiments/results.csv          one summary row per run
  experiments/runs/<name>.json     full details of the run (every step, the coloring)
"""

import contextlib
import csv
import io
import itertools
import json
import os
import random
import subprocess
import time
from datetime import datetime

from graph_utils import generate_graph_min_k, generate_geometric_graph
from pipeline import peel_then_kcoloring
from solving_twosets import is_valid_coloring

# ------------------------- settings: edit these -------------------------
EXPERIMENT = "geometric_grid"   # name that groups runs
GRAPH = "geometric"             # "geometric" or "planted"
NS = [100, 200, 500]            # geometric: every n ...
RADII = [0.1, 0.2, 0.3]         # ... with every radius
KS = [10]                       # planted: every k ...
GROUP_SIZES = [20]              # ... with every group size (or (min, max) tuple)
EDGE_PROBS = [0.30]             # ... and every edge probability
SEEDS = [0, 1, 2, 3, 4]         # every combination is run once per seed
STEPS, RESTARTS = 20000, 4      # kcoloring budget for each k it tries
NOTES = ""                      # free text saved with every run
QUIET = True                    # hide the SA's per-attempt printing
RESULTS_DIR = "experiments"     # relative to this file
# -------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, RESULTS_DIR)

CSV_COLUMNS = [
    "experiment", "time", "git_commit", "uncommitted_changes", "graph",
    "n", "radius", "k", "group_size", "edge_prob", "seed", "edges",
    "reference", "reference_type", "sa_colors", "final_colors", "gap", "valid",
    "kcoloring_tries", "sa_seconds", "total_seconds", "steps", "restarts",
    "notes", "json_file",
]


def quiet():
    """Hide printing when QUIET is on."""
    return contextlib.redirect_stdout(io.StringIO()) if QUIET else contextlib.nullcontext()


def git_version():
    """Current commit, and whether tracked code differs from it (results folder ignored)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no",
                         "--", ".", f":(exclude){RESULTS_DIR}"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, dirty


def largest_clique(graph):
    """Size of the largest clique: a lower bound on the number of colors."""
    import networkx as nx  # only needed for geometric graphs
    G = nx.Graph()
    G.add_nodes_from(graph)
    G.add_edges_from((u, v) for u in graph for v in graph[u] if u < v)
    return max(len(c) for c in nx.find_cliques(G))


def combinations():
    """Every combination of the settings lists for the chosen graph type."""
    if GRAPH == "geometric":
        return list(itertools.product(NS, RADII))
    if GRAPH == "planted":
        return list(itertools.product(KS, GROUP_SIZES, EDGE_PROBS))
    raise ValueError('GRAPH must be "geometric" or "planted"')


def label(params):
    if GRAPH == "geometric":
        return "n={} r={}".format(*params)
    return "k={} group={} p={}".format(*params)


def make_graph(params, seed):
    """Build one graph. Returns (graph, settings, reference value, reference type)."""
    random.seed(seed)
    if GRAPH == "geometric":
        n, radius = params
        graph, _ = generate_geometric_graph(n, radius)
        return graph, {"n": n, "radius": radius}, largest_clique(graph), "largest clique"
    k, group_size, edge_prob = params
    with quiet():
        graph, _ = generate_graph_min_k(k, group_size, edge_prob)
    return graph, {"k": k, "group_size": group_size, "edge_prob": edge_prob}, k, "planted k"


def run_one(params, seed, version):
    """Run the pipeline on one graph and return everything worth recording."""
    graph, settings, reference, reference_type = make_graph(params, seed)
    start = time.time()
    with quiet():
        colors, n_colors, history = peel_then_kcoloring(
            graph, seed=seed, steps=STEPS, restarts=RESTARTS)
    total = time.time() - start
    sa = history[0]
    return {
        "experiment": EXPERIMENT,
        "time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": version[0],
        "uncommitted_changes": version[1],
        "graph": GRAPH,
        **settings,
        "n": len(graph),
        "seed": seed,
        "edges": sum(len(nb) for nb in graph.values()) // 2,
        "reference": reference,
        "reference_type": reference_type,
        "sa_colors": sa["k"],
        "sa_seconds": sa["seconds"],
        "final_colors": n_colors,
        "gap": n_colors - reference,
        "valid": is_valid_coloring(graph, colors, n_colors),
        "kcoloring_tries": len(history) - 1,
        "total_seconds": round(total, 3),
        "steps": STEPS,
        "restarts": RESTARTS,
        "notes": NOTES,
        "sa_class_sizes": sa["class_sizes"],
        "history": history,
        "coloring": {str(v): c for v, c in sorted(colors.items())},
    }


def save(record, params):
    """Write the run's JSON file and append its summary row to results.csv."""
    runs_dir = os.path.join(RESULTS_PATH, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tag = label(params).replace(" ", "_").replace("=", "")
    stamp = record["time"].replace("-", "").replace(":", "")
    base = f"{record['experiment']}_{tag}_seed{record['seed']}_{stamp}"
    json_path = os.path.join(runs_dir, base + ".json")
    copy = 2
    while os.path.exists(json_path):
        json_path = os.path.join(runs_dir, f"{base}_{copy}.json")
        copy += 1
    with open(json_path, "w") as f:
        json.dump(record, f, indent=1)

    csv_path = os.path.join(RESULTS_PATH, "results.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, restval="", extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({**record, "json_file": os.path.relpath(json_path, RESULTS_PATH)})


def main():
    combos = combinations()
    runs = [(params, seed) for params in combos for seed in SEEDS]
    version = git_version()
    note = ", uncommitted changes" if version[1] else ""
    print(f"{EXPERIMENT}: {len(combos)} combinations x {len(SEEDS)} seeds"
          f" = {len(runs)} runs (code {version[0]}{note})")

    for i, (params, seed) in enumerate(runs, 1):
        record = run_one(params, seed, version)
        save(record, params)
        status = "valid" if record["valid"] else "INVALID"
        print(f"run {i}/{len(runs)}  {label(params)}  seed {seed}: "
              f"SA {record['sa_colors']} -> final {record['final_colors']} "
              f"(reference {record['reference']}), {status}, {record['total_seconds']:.1f}s",
              flush=True)

    print(f"results: {os.path.join(RESULTS_PATH, 'results.csv')}")


if __name__ == "__main__":
    main()
