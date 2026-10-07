from graph_utils import generate_uniform_graph, generate_geometric_graph, generate_graph_min_k
from solving_twosets import is_valid_coloring, color_graph_by_independent_sets_two
from solving_oneset import color_graph_by_independent_sets
import matplotlib.pyplot as plt

nodes=5
k=5
#graph, colors = generate_graph(10, 5)
#print(graph, colors)
print()
before = False
#graph, positions = generate_geometric_graph(nodes, 0.3)
data = []

graph, pcolors = generate_graph_min_k(k, nodes)

print()
print("Solved independent sets: ")

#scolors1 = color_graph_by_independent_sets(graph)
scolors2, ccolors = color_graph_by_independent_sets_two(graph)
'''
if is_valid_coloring(graph, scolors1):
    if len(set(pcolors.values())) == len(set(scolors1.values())):
        print(True)
    else:
        print(len(set(pcolors.values())))
        print(len(set(scolors1.values())))

print()
'''



if is_valid_coloring(graph, scolors2,ccolors):
    if len(set(pcolors.values())) == len(set(scolors2.values())):
        print(True)
    else:
        print(len(set(pcolors.values())))
        print(len(set(scolors2.values())))
else:
    print("invalid")
