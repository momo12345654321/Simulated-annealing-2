import math
import random
from solving_oneset import simulated_annealing_independent_set

#is independent set
def is_independent_set(graph, nodes):
    nodes = set(nodes)

    for v in nodes:
        for neighbor in graph[v]:
            if neighbor in nodes:
                return False

    return True

#count how many edges removed totally from independent sets A,B from rest of graph, not including edges between the two
def boundary_edges_two_sets(graph, S1, S2, uncolored):
    U = S1 | S2
    outside = uncolored - U

    count = 0
    for u in U:
        for v in graph[u]:
            if v in outside:
                count += 1

    return count



#counts how many edges removed between A and B, not the edges with rest of the graph
def cross_edges(graph, S1, S2):
    count = 0

    for u in S1:
        for v in graph[u]:
            if v in S2:
                count += 1

    return count



#gives a score considering the total edges from both independent set A and B to the rest of the graph being taken out, the edges in between A and B being taken out, and the size of A and B being colored
def score_two_sets(
    graph,
    S1,
    S2,
    uncolored,
    w_boundary,
    w_cross,
    w_size
):
    b = boundary_edges_two_sets(graph, S1, S2, uncolored)
    c = cross_edges(graph, S1, S2)
    size = len(S1) + len(S2)

    return w_boundary * b + w_cross * c + w_size * size



#higher weights given to items with larger gains
def weighted_choice_by_gain(candidates, gains, alpha=1.5):
    weights = []

    for gain in gains:
        weights.append((1 + max(0, gain))**alpha)

    improving_candidates = [
        (node, gain)
        for node, gain in zip(candidates, gains)
        if gain > 0
    ]

    # Prefer gains that improve the score
    if improving_candidates:
        improving_nodes = [
            node for node, _ in improving_candidates
        ]

        improving_weights = [
            (1+gain) for _, gain in improving_candidates
        ]

        return random.choices(
            improving_nodes,
            weights=improving_weights,
            k=1
        )[0]
    
    # Otherwise, prefer removals that cause the least damage
    weights = []

    for gain in gains:
        weights.append(1 /( 1 + max(0, -gain))**alpha)

    return random.choices(candidates, weights=weights, k=1)[0]




#returns how much score improves from adding S1/S2
def add_gain(graph, v, target, S1, S2, uncolored, w_boundary, w_cross, w_size):
    old_score = score_two_sets(
        graph, S1, S2, uncolored,
        w_boundary, w_cross, w_size
    )

    new_S1 = set(S1)
    new_S2 = set(S2)

    if target == 1:
        new_S1.add(v)
    else:
        new_S2.add(v)

    new_score = score_two_sets(
        graph, new_S1, new_S2, uncolored,
        w_boundary, w_cross, w_size
    )

    return new_score - old_score



#gets canidate
def weighted_add_candidate(
    graph,
    candidates,
    target,
    S1,
    S2,
    uncolored,
    alpha,
    w_boundary,
    w_cross,
    w_size
):
    gains = []

    for v in candidates:
        gain = add_gain(
            graph, v, target, S1, S2, uncolored,
            w_boundary, w_cross, w_size
        )
        gains.append(gain)

    return weighted_choice_by_gain(candidates, gains, alpha)



#return how much score decrease from removing node v, less is better, take out nodes contributing less
def removal_loss(graph, v, target, S1, S2, uncolored, w_boundary, w_cross, w_size):
    old_score = score_two_sets(
        graph, S1, S2, uncolored,
        w_boundary, w_cross, w_size
    )

    new_S1 = set(S1)
    new_S2 = set(S2)

    if target == 1:
        new_S1.remove(v)
    else:
        new_S2.remove(v)

    new_score = score_two_sets(
        graph, new_S1, new_S2, uncolored,
        w_boundary, w_cross, w_size
    )

    return old_score - new_score



#get canidate to remove
def weighted_remove_candidate(
    graph,
    nodes,
    target,
    S1,
    S2,
    uncolored,
    w_boundary,
    w_cross,
    w_size
):
    losses = []

    for v in nodes:
        loss = removal_loss(
            graph, v, target, S1, S2, uncolored,
            w_boundary, w_cross, w_size
        )
        losses.append(loss)

    improving_candidates = [
        (node, loss)
        for node, loss in zip(nodes, losses)
        if loss < 0
    ]

    # Prefer removals that improve the score
    if improving_candidates:
        improving_nodes = [
            node for node, _ in improving_candidates
        ]

        improving_weights = [
            -loss for _, loss in improving_candidates
        ]

        return random.choices(
            improving_nodes,
            weights=improving_weights,
            k=1
        )[0]

    # Otherwise, prefer removals that cause the least damage
    weights = []

    for loss in losses:
        weights.append(1 / (1 + loss))

    return random.choices(nodes, weights=weights, k=1)[0]



#gives valid neighbor to S1 or S2, alternates between S1/S2
def random_valid_neighbor_two_sets(
    graph,
    S1,
    S2,
    uncolored,
    target,
    remove_prob,
    alpha,
    w_boundary,
    w_cross,
    w_size
):
    '''
    Alternates between modifying S1 and S2.

    target = 1 means this move tries to modify S1.
    target = 2 means this move tries to modify S2.

    The move is usually an add, but sometimes a remove.
    If no add is possible, it forces a remove.
    '''

    S1_new = set(S1)
    S2_new = set(S2)

    if target == 1:
        target_set = S1_new
    else:
        target_set = S2_new

    if target_set == S1_new:
        outside = uncolored - S1_new
    else: 
        outside = uncolored - S2_new

    candidates = []

    for v in outside:
        if graph[v].isdisjoint(target_set):
            candidates.append(v)

    # Remove if stuck, or sometimes remove randomly to escape bad choices
    should_remove = False

    if not candidates:
        should_remove = True
    elif target_set and random.random() < remove_prob:
        should_remove = True

    if should_remove:
        if target_set:
            nodes = list(target_set)

            node = weighted_remove_candidate(
                graph,
                nodes,
                target,
                S1_new,
                S2_new,
                uncolored,
                w_boundary,
                w_cross,
                w_size
            )

            target_set.remove(node)

    else:
        node = weighted_add_candidate(
            graph,
            candidates,
            target,
            S1_new,
            S2_new,
            uncolored,
            alpha,
            w_boundary,
            w_cross,
            w_size
        )
        if target_set == S1_new:
            if node in S2_new:
                S2_new.remove(node)
            else:
                target_set.add(node)
        else:
            if node in S1_new:
                S1_new.remove(node)
            else:
                target_set.add(node)

    return S1_new, S2_new

def get_two_single_sets(   
    graph,
    uncolored,
    steps,
    start_temp,
    cooling,
    w_boundary,
    w_cross,
    w_size
):
    uncolored = set(uncolored)

    if not uncolored:
        return set(), set(), 0, 0.0

    best_step = 0

    step1=0
    step2=0


    S1,step1 = simulated_annealing_independent_set(
        graph,
        uncolored,
        steps,
        start_temp,
        cooling,
        w_boundary,
        w_size,

    )

    if not S1:
        S1 = {random.choice(list(uncolored))}

    remaining_after_S1 = uncolored - S1

    if not remaining_after_S1:
        pair_score = score_two_sets(
        graph,
        S1,
        set(),
        uncolored,
        w_boundary,
        w_cross,
        w_size
        )   
        return S1, set(), step1, pair_score

    S2,step2 = simulated_annealing_independent_set(
        graph,
        remaining_after_S1,
        steps,
        start_temp,
        cooling,
        w_boundary,
        w_size
    )

    if not S2:
        S2 = {random.choice(list(remaining_after_S1))}


    best_step = max(step1,step2)

    current_score = score_two_sets(
        graph,
        S1,
        S2,
        uncolored,
        w_boundary,
        w_cross,
        w_size
    )


    return S1, S2, best_step, current_score

def find_best_independent_set_pair(
    graph,
    uncolored,
    steps,
    start_temp,
    cooling,
    remove_prob,
    alpha,
    w_boundary,
    w_cross,
    w_size,
    patience=20,
    min_improvement=0.05,
):
    saved_pairs = []

    best_S1 = set()
    best_S2 = set()
    best_score = float("-inf")

    attempts_without_improvement = 0
    attempt_number = 0
    current_steps = steps

    while attempts_without_improvement < patience:

        attempt_number += 1

        S1, S2, steps_to_pair,pair_score = (
            get_two_single_sets(
                graph=graph,
                uncolored=uncolored,
                steps=current_steps,
                start_temp=start_temp,
                cooling=cooling,
                #remove_prob=remove_prob,
                #alpha=alpha,
                w_boundary=w_boundary,
                w_cross=w_cross,
                w_size=w_size
            )
        )

        # Save copies so later mutations cannot change old results.
        saved_pairs.append({
            "attempt": attempt_number,
            "S1": set(S1),
            "S2": set(S2),
            "score": pair_score,
            "steps_allowed": current_steps,
            "steps_to_pair": steps_to_pair
        })

        if best_score == float("-inf"):
            substantial_improvement = True
        else:
            substantial_improvement = ((pair_score - best_score) / max(abs(best_score), 1e-12) >= min_improvement)


        if substantial_improvement:
            best_S1 = set(S1)
            best_S2 = set(S2)
            best_score = pair_score

            # Reset consecutive unsuccessful attempts.
            attempts_without_improvement = 0

            # The next attempt receives the same number of steps
            # used to reach this improved pair.
            if steps_to_pair > 0:
                current_steps = steps_to_pair

            print(
                f"Attempt {attempt_number}: new best score "
                f"{best_score:.3f}; next step budget = {current_steps}"
            )
        else:
            attempts_without_improvement += 1

            print(
                f"Attempt {attempt_number}: score={pair_score:.3f}; "
                f"no substantial improvement "
                f"({attempts_without_improvement}/{patience})"
            )

    actual_best = max(saved_pairs, key=lambda result: result["score"])

    return (
        set(actual_best["S1"]),
        set(actual_best["S2"]),
        actual_best["score"],
        saved_pairs
    )


def color_graph_by_independent_sets_two(
    graph,
    steps=40000,
    start_temp=10.0,
    cooling=0.998,
    remove_prob=0.4,
    alpha=1.1,
    w_boundary=1.0,
    w_cross=0.15,
    w_size=5.0
):
    uncolored = set(graph.keys())
    colors = {}
    color_id = 0

    while uncolored:

        # All remaining vertices can share one color.
        if is_independent_set(graph, uncolored):
            outgoing_edges = set()
            for node in uncolored:
                for neighbor in graph[node]:
                    if neighbor not in uncolored:
                        outgoing_edges.add(tuple(sorted((node, neighbor))))

            print(f"\nIndependent set {color_id}:")
            print(f"  Nodes: {sorted(uncolored)}")
            print(f"  Number of nodes: {len(uncolored)}")
            print(f"  Number of outgoing edges: {len(outgoing_edges)}")

            for node in uncolored:
                colors[node] = color_id
            color_id += 1
            uncolored.clear()
            break

        S1, S2, selected_score, saved_pairs = (
            find_best_independent_set_pair(
            graph=graph,
            uncolored=uncolored,
            steps=steps,
            start_temp=start_temp,
            cooling=cooling,
            remove_prob=remove_prob,
            alpha=alpha,
            w_boundary=w_boundary,
            w_cross=w_cross,
            w_size=w_size,
            patience=20,
            min_improvement=0.05
            )
        )

        print(f"\nSelected pair score: {selected_score:.3f}")
        print(f"Number of pairs tested: {len(saved_pairs)}")    

        print("\nIndependent-set statistics S1:")


        outgoing_edges = set()

        for node in S1:
            for neighbor in graph[node]:
                if neighbor not in S1:
                    outgoing_edges.add(tuple(sorted((node, neighbor))))

        print(f"\nIndependent set {color_id}:")
        print(f"  Nodes: {sorted(S1)}")
        print(f"  Number of nodes: {len(S1)}")
        print(f"  Number of outgoing edges: {len(outgoing_edges)}")

        print("\nIndependent-set statistics S2:")

        outgoing_edges = set()

        for node in S2:
            for neighbor in graph[node]:
                if neighbor not in S2:
                    outgoing_edges.add(tuple(sorted((node, neighbor))))

        print(f"\nIndependent set {color_id+1}:")
        print(f"  Nodes: {sorted(S2)}")
        print(f"  Number of nodes: {len(S2)}")
        print(f"  Number of outgoing edges: {len(outgoing_edges)}")

        
        
        # Safety fallback
        if not S1 and not S2:
            node = random.choice(list(uncolored))
            S1 = {node}
    
        #if can be merged color as one, else color two seperately
        combined_set = S1 | S2

        if is_independent_set(graph, combined_set):
            for node in combined_set:
                colors[node] = color_id

            uncolored -= combined_set
            color_id += 1
        else:
            for node in S1:
                colors[node] = color_id

            for node in S2:
                colors[node] = color_id + 1

            uncolored -= S1
            uncolored -= S2
            color_id += 2


        print("round done")

    return colors, color_id























def simulated_annealing_two_independent_sets(
    graph,
    uncolored,
    steps,
    start_temp,
    cooling,
    remove_prob,
    alpha,
    w_boundary,
    w_cross,
    w_size
):
    uncolored = set(uncolored)

    if len(uncolored) == 0:
        return set(), set()

    if len(uncolored) == 1:
        return set(uncolored), set()

    nodes = list(uncolored)

    first = random.choice(nodes)
    S1 = {first}

    second = random.choice(list(uncolored - S1))
    S2 = {second}

    current_S1 = set(S1)
    current_S2 = set(S2)

    current_score = score_two_sets(
        graph,
        current_S1,
        current_S2,
        uncolored,
        w_boundary,
        w_cross,
        w_size
    )

    best_S1 = set(current_S1)
    best_S2 = set(current_S2)
    best_score = current_score

    T = start_temp

    for step in range(steps):
        '''
        # Alternate between S1 and S2
        if step % 2 == 0:
            target = 1
        else:
            target = 2
        #print(step)
        #print(current_score)
        '''
        target = random.choice([1, 2])

        candidate_S1, candidate_S2 = random_valid_neighbor_two_sets(
            graph,
            current_S1,
            current_S2,
            uncolored,
            target,
            remove_prob,
            alpha,
            w_boundary,
            w_cross,
            w_size
        )

        candidate_score = score_two_sets(
            graph,
            candidate_S1,
            candidate_S2,
            uncolored,
            w_boundary,
            w_cross,
            w_size
        )

        delta = candidate_score - current_score

        if delta > 0 or random.random() < math.exp(delta / T):
            current_S1 = candidate_S1
            current_S2 = candidate_S2
            current_score = candidate_score

        if current_score > best_score:
            best_S1 = set(current_S1)
            best_S2 = set(current_S2)
            best_score = current_score

        T *= cooling

        if T < 1e-8:
            T = 1e-8

    return best_S1, best_S2












#check if coloring is legal
def is_valid_coloring(graph, colors,ccolors):
    if set(colors) != set(graph):
        return False
    
    for u in graph:
        for v in graph[u]:
            if colors[u] == colors[v]:
                return False
            
    if len(set(colors.values()))!=ccolors:
        return False
    return True