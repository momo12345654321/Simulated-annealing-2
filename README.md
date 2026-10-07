# Graph Coloring Simulated Annealing Algorithm using Two Independent Sets

A simulated annealing algorithm for finding optimal graph coloring solutions by finding two independent sets per iteration. The points with no connections between them can all share a color, so the program colors a graph one group at a time. Simulated annealing is used to repeatedly find two large groups of unconnected points that also cut many connections from the rest of the graph. Each group gets a color and is removed, and this repeats until every point is colored.

# Researcher

Jasper Ding

# How to run

1.Set the number of colors and the group size at the top of main.py.
2. Run python main.py.

The program prints whether the coloring is valid and whether it used the minimum number of colors.

