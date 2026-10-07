import random
import math
import matplotlib.pyplot as plt


def generate_uniform_graph(n, m):
    """
    Generate an undirected simple graph with n vertices and exactly m edges.
    Vertices are labeled 0, 1, ..., n-1.
    Returns an adjacency list.
    """
    possible_edges = []

    for i in range(n):
        for j in range(i + 1, n):
            possible_edges.append((i, j))

    if m > len(possible_edges):
        raise ValueError("Too many edges requested.")

    edges = random.sample(possible_edges, m)

    graph = {i: [] for i in range(n)}

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    return graph


def generate_geometric_graph(n, radius):
    
    #create random 2D positions
    positions = {}

    for i in range(n):
        x = random.random()
        y = random.random()
        positions[i] = (x, y)

    #create empty graph
    graph = {i: [] for i in range(n)}

    #connect points that are close enough
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]

            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            if distance <= radius:
                graph[i].append(j)
                graph[j].append(i)

    return graph, positions




'''
def generate_graph_min_k(n,k):
    #node and its color
    colors = {}
    #create empty graph
    graph = {i: set() for i in range(n)}
    #nodes reserved for clique
    clique_nodes = random.sample(range(n), k)
    
    #first k clique
    for color, node in enumerate(clique_nodes):
        colors[node]=color
    
    #edge between clique
    for i in range(k):
        for j in range(i+1,k):
            graph[clique_nodes[i]].add(clique_nodes[j])
            graph[clique_nodes[j]].add(clique_nodes[i])

    #assign rest randomly
    for node in range(n):
        if node not in colors:
            colors[node] = random.randint(0,k-1)

    #randomly put edge between if colors DONT match
    for i in range(n):
        for j in range(i+1,n):
            if colors[i]!=colors[j]:
                if random.random()<0.30:
                    graph[i].add(j)
                    graph[j].add(i)


    # Build the independent sets from the color assignments
    independent_sets = {color: set() for color in range(k)}

    for node, color in colors.items():
        independent_sets[color].add(node)

    
    # Print information about every independent set
    print("\nIndependent-set statistics:")

    for color, nodes in independent_sets.items():
        outgoing_edges = set()

        for node in nodes:
            for neighbor in graph[node]:
                if neighbor not in nodes:
                    outgoing_edges.add(tuple(sorted((node, neighbor))))

        print(f"\nIndependent set {color}:")
        print(f"  Nodes: {sorted(nodes)}")
        print(f"  Number of nodes: {len(nodes)}")
        print(f"  Number of outgoing edges: {len(outgoing_edges)}")
        #print(f"  Outgoing edges: {sorted(outgoing_edges)}")

    

    return graph, colors
'''

def generate_graph_min_k(k, group_size, edge_prob=0.30):
    """
    group_size: int (fixed size per color group) or (min, max) tuple for variable sizes
    k: number of colors/groups (this becomes the true chromatic number)
    """
    if isinstance(group_size, tuple):
        group_sizes = [random.randint(*group_size) for _ in range(k)]
    else:
        group_sizes = [group_size] * k

    n = sum(group_sizes)

    colors = {}
    graph = {i: set() for i in range(n)}

    # assign nodes to color groups up front (balanced by construction)
    node = 0
    independent_sets = {c: set() for c in range(k)}
    for c, size in enumerate(group_sizes):
        for _ in range(size):
            colors[node] = c
            independent_sets[c].add(node)
            node += 1

    # pick one representative node per color to form the guaranteeing clique
    clique_nodes = [next(iter(independent_sets[c])) for c in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            a, b = clique_nodes[i], clique_nodes[j]
            graph[a].add(b)
            graph[b].add(a)

    # random edges between differently-colored nodes
    for i in range(n):
        for j in range(i + 1, n):
            if colors[i] != colors[j] and random.random() < edge_prob:
                graph[i].add(j)
                graph[j].add(i)


    print(f"\nGenerated graph: n={n}, k={k}, group_sizes={group_sizes}")
    print("Independent-set statistics:")
    for c, nodes in independent_sets.items():
        outgoing = set()
        for node in nodes:
            for nb in graph[node]:
                if nb not in nodes:
                    outgoing.add(tuple(sorted((node, nb))))
        print(f"  Set {c}: {len(nodes)} nodes, {len(outgoing)} outgoing edges")

    return graph, colors



