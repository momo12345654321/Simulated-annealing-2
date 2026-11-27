# -*- coding: utf-8 -*-
"""
kcoloring.py — a coloring solver that targets "use k colors" directly.

Why this module exists (see experiments/FINDINGS.md, experiment 17):

  The existing pipeline is GREEDY PEELING: each round finds one large
  independent set as a color class, strips it out, and continues.
  It optimizes the score of a SINGLE color class
  (w_boundary*b + w_size*size), while there is
  NO GLOBAL OBJECTIVE MINIMIZING THE TOTAL NUMBER OF COLORS.

  Observed symptoms (experiment 17.1):
      n=30 typical class sequence [11, 9, 5, 4, 1]  -> last class has 1 node
      n=40 typical class sequence [12, 10, 9, 6, 3] -> tail class has 3 nodes
      n=50 typical class sequence [10, 11, 10, 11, 7, 1] -> 6 colors
  In other words: the classes keep fragmenting, and the last one or two
  classes hold only a few nodes, wasting 1~2 extra colors.

  Meanwhile (experiment 17.3) the graph has plenty of slack:
  alpha(G) >= 16 while the true solution needs classes of only 10 nodes.
  => THIS IS NOT "A NEEDLE IN A HAYSTACK"; THE SEARCH PROCEDURE IS
     NOT ALIGNED WITH THE OBJECTIVE.

  Experiment 17.4 confirms this further: merely retuning the weights
  (including the w_boundary=0 "pure size" objective) did not improve
  n=40/50 (0~2 out of 6 hits). So the objective itself must change —
  parameter tuning alone cannot fix it.

What this module does — write the problem directly as
"assign each vertex one of k colors":
  Objective (minimized): J = same-color edges + w_balance * class imbalance
  Move: recolor a vertex (occasional swaps of two vertices' colors)
  Initialization: DSATUR (pick by saturation; the standard coloring heuristic)
  Termination: same-color edges reach 0 — at that point we have
               a VALID k-COLORING

How this relates to the peel pipeline:
  - peel is "incremental, greedy": find an independent set in the
    remaining graph at each step
  - kcoloring is "global, constraint satisfaction": directly search
    for whether a k-coloring exists
  The two are complementary; experiment 18 does the A/B.
"""

import math
import random


def _dsatur_init(graph, k, rng):
    """
    DSATUR initialization: each round picks the uncolored vertex with
    the highest saturation (most distinct colors already used by its
    neighbors), breaking ties by degree, and assigns it the color that
    conflicts with the fewest already-colored neighbors.
    Returns colors[v] = color index.
    """
    n = len(graph)
    color = [-1] * n
    sat_sets = [set() for _ in range(n)]

    for _ in range(n):
        best = -1
        best_key = (-1, -1)
        for v in range(n):
            if color[v] != -1:
                continue
            key = (len(sat_sets[v]), len(graph[v]))
            if key > best_key:
                best_key = key
                best = v

        best_c = 0
        best_conf = None
        for c in range(k):
            conf = 0
            for u in graph[best]:
                if color[u] == c:
                    conf += 1
            if best_conf is None or conf < best_conf:
                best_conf = conf
                best_c = c

        color[best] = best_c
        for u in graph[best]:
            if color[u] == -1:
                sat_sets[u].add(best_c)

    return color


def _solve_once(graph, k, steps, start_temp, cooling, w_balance,
                t_floor, rng, noise=0.30, verbose=False, report_every=0,
                stats=None, init_color=None):
    """
    One SA run. All internal state is maintained incrementally:
      cnt[v][c] = number of neighbors of v whose color is c
      vsame[v]  = number of same-colored neighbors of v
                  (= cnt[v][color[v]])
      viol      = number of same-color edges (= sum(vsame)/2)
    Returns (color, best_viol)

    When verbose=True, the annealing progress is printed periodically:
      step, temperature, current conflict count, best conflict count,
      and how many vertices are still in conflict.

    report_every = number of steps between reports; 0 selects an
    automatic interval (about 20 reports per run).

    stats, if given, is a dict that receives run statistics
    (steps_used, T_final, viol_final, best_viol).

    init_color, if given, is the starting coloring (list indexed by vertex,
    colors 0..k-1) used instead of the DSATUR guess.
    """
    n = len(graph)
    if init_color is not None:
        color = list(init_color)
    else:
        color = _dsatur_init(graph, k, rng)

    cnt = [[0] * k for _ in range(n)]
    for u in graph:
        for v in graph[u]:
            cnt[u][color[v]] += 1

    sizes = [0] * k
    for c in color:
        sizes[c] += 1

    vsame = [cnt[v][color[v]] for v in range(n)]
    bad = set(v for v in range(n) if vsame[v] > 0)

    best_viol = sum(vsame) // 2
    best_color = list(color)
    viol = best_viol

    T = start_temp
    target = n / k

    if verbose:
        if report_every <= 0:
            report_every = max(1, steps // 20)
        print(f"    [anneal] start: steps={steps} T0={start_temp:.4g} "
              f"cooling={cooling:.6f} viol={best_viol} "
              f"conflicted_vertices={len(bad)}")

    steps_used = 0

    for step in range(steps):
        if not bad:
            break
        steps_used = step + 1

        # Pick a conflicted vertex
        v = rng.choice(tuple(bad))
        c1 = color[v]

        # Compute the objective change for every candidate color
        best_c = None
        best_dj = None
        cand_dj = []

        nbr_c1 = 0
        nbr_c2 = [0] * k
        for u in graph[v]:
            cu = color[u]
            if cu == c1:
                nbr_c1 += 1
            nbr_c2[cu] += 1

        for c2 in range(k):
            if c2 == c1:
                continue
            # Change in the same-color incidence count (viol is half of
            # it; using the incidence count keeps everything integral)
            d_inc = (cnt[v][c2] - cnt[v][c1]) + (nbr_c2[c2] - nbr_c1)
            # Change in the color-class balance penalty
            d_bal = ((sizes[c2] + 1 - target) ** 2 + (sizes[c1] - 1 - target) ** 2
                     - (sizes[c2] - target) ** 2 - (sizes[c1] - target) ** 2)
            dj = d_inc + w_balance * d_bal
            cand_dj.append((c2, dj))
            if best_dj is None or dj < best_dj:
                best_dj = dj
                best_c = c2

        if not cand_dj:
            break

        # Pick a random candidate color with some probability
        # (exploration), otherwise the best one (exploitation)
        if rng.random() < noise:
            c2, dj = rng.choice(cand_dj)
        else:
            c2, dj = best_c, best_dj

        # Metropolis acceptance
        accept = dj <= 0
        if not accept:
            # viol is measured in incidence-count units; scale it so
            # the temperature has a sensible magnitude
            accept = rng.random() < math.exp(-dj / max(T, 1e-9))

        if accept:
            # Apply the recoloring
            color[v] = c2
            sizes[c1] -= 1
            sizes[c2] += 1

            affected = []
            for u in graph[v]:
                cnt[u][c1] -= 1
                cnt[u][c2] += 1
                affected.append(u)

            vsame[v] = cnt[v][c2]
            for u in affected:
                cu = color[u]
                if cu == c1:
                    vsame[u] -= 1
                elif cu == c2:
                    vsame[u] += 1

            for x in affected:
                if vsame[x] > 0:
                    bad.add(x)
                else:
                    bad.discard(x)
            if vsame[v] > 0:
                bad.add(v)
            else:
                bad.discard(v)

            viol = sum(vsame) // 2
            if viol < best_viol:
                best_viol = viol
                best_color = list(color)

        T *= cooling
        if T < t_floor:
            T = t_floor

        if verbose and (step + 1) % report_every == 0:
            print(f"    [anneal] step {step+1:>6}/{steps}  T={T:.4g}  "
                  f"viol={viol}  best={best_viol}  "
                  f"conflicted_v={len(bad)}")

    if verbose:
        print(f"    [anneal] done: steps_used={steps_used} T={T:.4g} "
              f"viol={viol} best={best_viol} conflicted_v={len(bad)}")

    if stats is not None:
        stats["steps_used"] = steps_used
        stats["T_final"] = T
        stats["viol_final"] = viol
        stats["best_viol"] = best_viol

    return best_color, best_viol


def _local_repair(graph, color, k, rounds=4000, rng=None):
    """
    Greedy repair: repeatedly find a conflicting edge and move one of
    its endpoints to the color with the fewest conflicts.
    Returns the repaired color. Used as a finishing pass after the SA.
    """
    if rng is None:
        rng = random
    n = len(graph)
    color = list(color)

    cnt = [[0] * k for _ in range(n)]
    for u in graph:
        for v in graph[u]:
            cnt[u][color[v]] += 1
    vsame = [cnt[v][color[v]] for v in range(n)]

    for _ in range(rounds):
        bad = [v for v in range(n) if vsame[v] > 0]
        if not bad:
            break
        v = rng.choice(bad)
        c1 = color[v]

        # Pick the color with the fewest conflicts; on a tie, pick one
        # that drives vsame to zero
        best_c = None
        best_key = None
        for c2 in range(k):
            if c2 == c1:
                continue
            conf = cnt[v][c2]
            key = (conf, rng.random())
            if best_key is None or key < best_key:
                best_key = key
                best_c = c2

        color[v] = best_c
        affected = []
        for u in graph[v]:
            cnt[u][c1] -= 1
            cnt[u][best_c] += 1
            affected.append(u)
        vsame[v] = cnt[v][best_c]
        for u in affected:
            cu = color[u]
            if cu == c1:
                vsame[u] -= 1
            elif cu == best_c:
                vsame[u] += 1

    return color


def is_valid_coloring(graph, colors, ccolors):
    """
    Validate a coloring: every vertex colored, no edge with same-colored
    ends, and the number of distinct colors matches ccolors.

    This mirrors the helper of the same name in the baseline peel solver
    (solving_twosets.py) so that main.py can be read side by side with
    the original.
    """
    if set(colors) != set(graph):
        return False

    for u in graph:
        for v in graph[u]:
            if colors[u] == colors[v]:
                return False

    if len(set(colors.values())) != ccolors:
        return False
    return True


def color_graph_k(graph, k=5, steps=20000, start_temp=2.0,
                  cooling=None, match_schedule=False, w_balance=0.02,
                  restarts=4, seed=None, t_floor=1e-3, verbose=False,
                  cool_scale=0.1, report_every=0, init_color=None):
    """
    Solve "color graph with k colors".

    init_color: optional starting coloring (list indexed by vertex, colors
    0..k-1). Every restart starts from it instead of the DSATUR guess.

    Returns (colors_dict, n_colors_used, info)
      colors_dict   : {vertex: color_id}
      n_colors_used : number of colors actually used
                      (<= k means success)
      info          : dict(viol=remaining conflicting edges,
                          valid=whether it is a valid k-coloring,
                          restarts=number of restarts)

    -- Defaults come from experiment 21 (53 hard cases, 2x2 schedule design) --
      cool_scale=0.1  means cooling is reverse-derived for
                      steps/0.1 = 200000 steps, i.e. SLOW ANNEALING.
                      Measured:
                        slow(0.1x) x 1     50/53 = 94.3%   0.071s  <== best
                        matched(1x) x 5   50/53 = 94.3%   0.158s
                        fast(10x) x 1     37/53 = 69.8%   0.141s
                      Conclusion: SLOW ANNEALING + FEW RESTARTS beats
                      FAST ANNEALING + MANY RESTARTS, because with a fixed
                      total step budget one long anneal walks a longer
                      temperature path, whereas multiple short anneals
                      each restart in the high-temperature phase and
                      waste their exploration.
                      Slow annealing is also FASTER (each step touches
                      one vertex only, and it converges earlier).
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if cooling is None:
        if steps <= 0 or start_temp <= t_floor:
            cooling = 0.999
        else:
            # cool_scale=1.0 => cooling walks the temperature schedule
            #                  exactly over `steps` iterations
            # cool_scale=0.1 => cooling is reverse-derived for
            #                  10*steps iterations (10x slower)
            span = max(1, int(steps / max(cool_scale, 1e-9)))
            cooling = (t_floor / start_temp) ** (1.0 / span)

    if init_color is not None:
        if len(init_color) != len(graph) or not all(
                0 <= c < k for c in init_color):
            raise ValueError("init_color needs one color in 0..k-1 "
                             "for every vertex")

    best_color = None
    best_viol = None

    for r in range(restarts):
        run_stats = {}
        col, viol = _solve_once(graph, k, steps, start_temp, cooling,
                                 w_balance, t_floor, rng,
                                 verbose=verbose, report_every=report_every,
                                 stats=run_stats, init_color=init_color)
        if best_viol is None or viol < best_viol:
            best_viol = viol
            best_color = col
        if verbose:
            if viol == 0:
                print(f"Attempt {r+1}: conflicts=0; no remaining conflicts")
            else:
                print(f"Attempt {r+1}: conflicts={viol}; "
                      f"no substantial improvement")
        if viol == 0:
            break

    # Finishing repair
    if best_viol and best_viol > 0:
        if verbose:
            print(f"  [repair] starting greedy repair on {best_viol} "
                  f"conflicting edges")
        fixed = _local_repair(graph, best_color, k, rng=rng)
        cnt = [[0] * k for _ in range(len(graph))]
        for u in graph:
            for v in graph[u]:
                cnt[u][fixed[v]] += 1
        v = sum(cnt[i][fixed[i]] for i in range(len(graph))) // 2
        if verbose:
            print(f"  [repair] done: conflicts {best_viol} -> {v}")
        if v < best_viol:
            best_viol = v
            best_color = fixed

    colors = {}
    used = set()
    for node in sorted(graph):
        c = best_color[node]
        colors[node] = c
        used.add(c)

    # Renumber so the color ids are contiguous
    remap = {c: i for i, c in enumerate(sorted(used))}
    colors = {node: remap[c] for node, c in colors.items()}
    n_colors = len(remap)

    valid = True
    for u in graph:
        for v in graph[u]:
            if colors[u] == colors[v]:
                valid = False
                break
        if not valid:
            break

    return colors, n_colors, {
        "viol": best_viol,
        "valid_k_coloring": bool(valid and best_viol == 0),
        "restarts": restarts,
    }
