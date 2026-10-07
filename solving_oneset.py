import math
import random


#count how many constraints removed by choosing S
def boundary_edges(graph, S, remaining):
    count = 0
    outside = remaining - S

    for u in S:
        for v in graph[u]:
            if v in outside:
                count += 1

    return count

#how good the current is in terms of nodes colored and contraints going to be taken out          
def score_set(graph, current, remaining, w_boundary=1.0, w_size=2.0):
    b = boundary_edges(graph, current, remaining)
    size = len(current)

    return w_boundary * b + w_size * size

#give larger probability of being selected to nodes with more edges to rest of graph
def candidate_weight(graph, v, S, remaining):
    rest = remaining - S - {v}

    edges_to_rest = 0
    for neighbor in graph[v]:
        if neighbor in rest:
            edges_to_rest += 1

    return 1 + edges_to_rest
    
#get candiate
def weighted_random_add_candidate(graph, candidates, S, remaining):
    weights = []

    for v in candidates:
        weights.append(candidate_weight(graph, v, S, remaining))

    return random.choices(candidates, weights=weights, k=1)[0]

#if the next step is valid, random add/remove
def random_valid_neighbor(graph, S, remaining):
    S_new = set(S)

    move_type = random.choice(["add", "remove", "swap"])

    if move_type == "add":
        candidates = []

        for v in remaining - S_new:
            if graph[v].isdisjoint(S_new):
                candidates.append(v)

        if candidates:
            node = weighted_random_add_candidate(graph, candidates, S, remaining)
            S_new.add(node)

    #elif move_type == "remove":
        #if S_new:
            #node = random.choice(list(S_new))
            #S_new.remove(node)

    elif move_type == "swap":
        inside = list(S_new)
        outside = list(remaining - S_new)

        if inside and outside:
            remove_node = random.choice(inside)

            temp_set = set(S_new)
            temp_set.remove(remove_node)

            candidates = []

            for add_node in outside:
                if graph[add_node].isdisjoint(temp_set):
                    candidates.append(add_node)

            if candidates:
                add_node = random.choice(candidates)
                S_new.remove(remove_node)
                S_new.add(add_node)

    return S_new




def simulated_annealing_independent_set(graph, remaining, steps=5000, start_temp=10.0, cooling=0.995, w_boundary=1.0, w_size=2.0):

    remaining = set(remaining)
    remaining1 = set(remaining)

    # Start with one random node
    current = {random.choice(list(remaining))}
    current = {random.choice(list(remaining))}

    current_score = score_set(
        graph, current, remaining,
        w_boundary, w_size
    )

    best = set(current)
    best_score = current_score
    beststep=0

    T = start_temp

    for step in range(steps):
        candidate = random_valid_neighbor(graph, current, remaining)

        candidate_score = score_set(
            graph, candidate, remaining,
            w_boundary, w_size
        )

        #print(current_score)

        delta = candidate_score - current_score

        # Accept if better, or sometimes accept if worse
        if delta > 0 or random.random() < math.exp(delta / T):
            current = candidate
            current_score = candidate_score 
        
        if current_score > best_score:
            best = set(current)
            best_score = current_score
            beststep=step

        T *= cooling

        # Prevent temperature from becoming exactly 0
        if T < 1e-8:
            T = 1e-8

    return best, beststep


def color_graph_by_independent_sets(
    graph,
    steps=5000,
    start_temp=10.0,
    cooling=0.995,
    w_boundary=1.0,
    w_size=2.0,
):
    remaining = set(graph.keys())
    colors = {}
    color_id = 0

    while remaining:
        S, _= simulated_annealing_independent_set(
            graph,
            remaining,
            steps=steps,
            start_temp=start_temp,
            cooling=cooling,
            w_boundary=w_boundary,
            w_size=w_size
        )
        '''
        print("remaining before:", len(remaining))
        print("colored this round:", len(S))
        print("remaining after:", len(remaining - S))
        print()
        
        # Safety fallback: if something goes wrong, color one node
        if not S:
            S = {random.choice(list(remaining))}
        '''

        print(f"\nIndependent set {color_id+1}:")
        print(f"  Nodes: {sorted(S)}")
        print(f"  Number of nodes: {len(S)}")
        print(f"  Number of outgoing edges: {sum(1 for u in S for v in graph[u] if v in remaining - S)}")

        # Assign the same color to every node in the independent set
        for node in S:
            colors[node] = color_id

        # Remove that independent set from the remaining graph
        remaining -= S

        color_id += 1

        

    return colors