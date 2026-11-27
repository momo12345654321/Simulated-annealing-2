"""
Two-stage coloring.

Stage 1: the repo's simulated annealing (find one independent set, find
another, remove both, repeat) gives a valid coloring with K colors.

Stage 2: kcoloring tries K-1 colors, starting from that coloring with its
smallest color class removed. If it removes every conflict, repeat with one
color fewer. At the first k it cannot fix, stop and return the last valid
coloring.
"""

import random
import time

from solving_twosets import color_graph_by_independent_sets_two, is_valid_coloring
from kcoloring import color_graph_k


def class_sizes(colors):
    """Number of vertices in each color class, in color-id order."""
    sizes = {}
    for c in colors.values():
        sizes[c] = sizes.get(c, 0) + 1
    return [sizes[c] for c in sorted(sizes)]


def drop_smallest_class(graph, colors, rng):
    """
    Turn a valid K-coloring into a (K-1)-coloring that may have conflicts.

    The smallest color class is removed. Its vertices, highest degree first,
    each take the remaining color with the fewest same-colored neighbours
    (ties broken at random). Returns a list indexed by vertex, colors 0..K-2.
    """
    classes = {}
    for v, c in colors.items():
        classes.setdefault(c, []).append(v)
    drop = min(classes, key=lambda c: len(classes[c]))
    keep = sorted(c for c in classes if c != drop)
    new_id = {c: i for i, c in enumerate(keep)}

    color = [None] * len(graph)
    for v, c in colors.items():
        if c != drop:
            color[v] = new_id[c]

    for v in sorted(classes[drop], key=lambda u: len(graph[u]), reverse=True):
        conflicts = [0] * len(keep)
        for u in graph[v]:
            if color[u] is not None:
                conflicts[color[u]] += 1
        fewest = min(conflicts)
        color[v] = rng.choice([c for c, x in enumerate(conflicts) if x == fewest])

    return color


def peel_then_kcoloring(graph, seed=None, steps=20000, restarts=4):
    """
    Run stage 1, then remove one color at a time with kcoloring.

    Returns (colors, n_colors, history):
      colors   : {vertex: color} — the last valid coloring found
      n_colors : number of colors it uses
      history  : one dict per step: stage, k, ok, seconds (+ class_sizes if ok)
    """
    rng = random.Random(seed)
    random.seed(seed)  # the repo's SA uses the global random module

    t = time.time()
    colors, k = color_graph_by_independent_sets_two(graph)
    if not is_valid_coloring(graph, colors, k):
        raise RuntimeError("stage 1 returned an invalid coloring")
    history = [{"stage": "SA", "k": k, "ok": True,
                "seconds": round(time.time() - t, 3),
                "class_sizes": class_sizes(colors)}]

    while k > 1:
        t = time.time()
        init = drop_smallest_class(graph, colors, rng)
        new, used, info = color_graph_k(graph, k=k - 1, init_color=init,
                                        steps=steps, restarts=restarts,
                                        seed=rng.randrange(2 ** 31))
        ok = info["valid_k_coloring"] and is_valid_coloring(graph, new, used)
        step = {"stage": "kcoloring", "k": k - 1, "ok": ok,
                "seconds": round(time.time() - t, 3)}
        if ok:
            step["class_sizes"] = class_sizes(new)
        history.append(step)
        if not ok:
            break  # cannot fix k-1 colors: keep the k-color coloring
        colors, k = new, used

    return colors, k, history
