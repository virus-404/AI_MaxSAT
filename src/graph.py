#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import collections
import itertools
import msat_runner
import wcnf


# Graph class
###############################################################################

class Graph(object):
    """This class represents an undirected graph. The graph nodes are
    labeled 1, ..., n, where n is the number of nodes, and the edges are
    stored as pairs of nodes.
    """

    def __init__(self, file_path=""):
        self.edges = []
        self.n_nodes = 0

        if file_path:
            self.read_file(file_path)

    def read_file(self, file_path):
        """Loads a graph from the given file.

        :param file_path: Path to the file that contains a graph definition.
        """
        with open(file_path, 'r') as stream:
            self.read_stream(stream)

    def read_stream(self, stream):
        """Loads a graph from the given stream.

        :param stream: A data stream from which read the graph definition.
        """
        n_edges = -1
        edges = set()

        reader = (l for l in (ll.strip() for ll in stream) if l)
        for line in reader:
            l = line.split()
            if l[0] == 'p':
                self.n_nodes = int(l[2])
                n_edges = int(l[3])
            elif l[0] == 'c':
                pass  # Ignore comments
            else:
                edges.add(frozenset([int(l[1]), int(l[2])]))

        self.edges = tuple(tuple(x) for x in edges)
        if n_edges != len(edges):
            print("Warning incorrect number of edges")

    def min_vertex_cover(self, solver, n_solutions):
        """Computes the minimum vertex cover of the graph.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of nodes.
        """
        formula = wcnf.WCNFFormula()
        n_vars = [formula.new_var() for _ in range(self.n_nodes)]

        # soft: including a vertex in the cover has a cost of 1
        for i in range(self.n_nodes):
            formula.add_clause([-n_vars[i]], 1)  # clause weight = 1

        # hard: all edges must be covered
        for node1, node2 in self.edges:
            vertex1, vertex2 = n_vars[node1-1], n_vars[node2-1]
            formula.add_clause([vertex1, vertex2], wcnf.TOP_WEIGHT)

        # this is just an example, only one solution is computed
        n_solutions = 1
        all_solutions = []
        opt, model = solver.solve(formula)
        if opt >= 0:
            solution = [x for x in range(1, self.n_nodes + 1) if model[x-1] > 0]
            all_solutions.append(solution)

        return all_solutions

#python3 graph.py ./WPM1-2012 ../graphs/petersen.dmg

    def min_dominating_set(self, solver, n_solutions):
        """Computes the minimum dominating set of the graph.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of nodes.
        """
        formula = wcnf.WCNFFormula()
        n_vars = [formula.new_var() for _ in range(self.n_nodes)]
        neighbors = {} #neighbors
        all_solutions = []

        # soft: including a vertex in the cover has a cost of 1
        for i in range(self.n_nodes):
            neighbors[i + 1] = [i+1]
            formula.add_clause([-n_vars[i]], 1)  # clause weight = 1

        # hard: all edges must be covered
        for node1, node2 in self.edges:
            neighbors[node1].append(node2)
            neighbors[node2].append(node1)

        for edge in neighbors.values():
            formula.add_at_least_one(edge)

        # Shows all solutions possible
        opt, model = solver.solve(formula)
        curr_opt = opt
        counter = 0

        if opt <= 0: return all_solutions

        while (n_solutions < 1 and curr_opt == opt) or (n_solutions > 0 and curr_opt == opt and counter < n_solutions):
            counter += 1
            solution = [x for x in range(1, self.n_nodes+1) if model[x-1] > 0]
            all_solutions.append(solution)
            formula.add_clause(list(map(lambda x: -x, model)), wcnf.TOP_WEIGHT)
            curr_opt, model = solver.solve(formula)

        return all_solutions
        raise NotImplementedError("Your Code Here")

    def max_independent_set(self, solver, n_solutions):
        """Computes the maximum independent set of the graph.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of nodes.
        """
        formula = wcnf.WCNFFormula()
        n_vars = [formula.new_var() for _ in range(self.n_nodes)]
        all_solutions = []

        # soft: including a vertex in the cover has a cost of 1
        for i in range(self.n_nodes):
            formula.add_clause([n_vars[i]], 1)  # clause weight = 1

        #hard: Only one vertex maximum of each edge
        for node1, node2 in self.edges:
            vertex1, vertex2 = n_vars[node1-1], n_vars[node2-1]
            formula.add_at_most_one([vertex1, vertex2])

        # Shows all solutions possible
        opt, model = solver.solve(formula)
        curr_opt = opt
        counter = 0

        if opt <= 0: return all_solutions

        while (n_solutions < 1 and curr_opt == opt) or (n_solutions > 0 and curr_opt == opt and counter < n_solutions):
            counter += 1
            solution = [x for x in range(1, self.n_nodes+1) if model[x-1] > 0]
            all_solutions.append(solution)
            formula.add_clause(list(map(lambda x: -x, model)), wcnf.TOP_WEIGHT)
            curr_opt, model = solver.solve(formula)

        return all_solutions
        raise NotImplementedError("Your Code Here")

    def min_graph_coloring(self, solver, n_solutions):
        """Computes the sets of nodes that can be painted using the
        same color, such that two adjacent nodes do not use the same
        color.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of lists of
            nodes, where all the nodes in the same list are painted
            using the same color.
        """

        formula = wcnf.WCNFFormula()
        all_solutions = []
        solution = {}
        matrix = []
        neighbors = {}
        counter = 0

        for i in range(1, self.n_nodes + 1): #Starting the counter for each vertex
            neighbors[i] = 0

        for node1, node2 in self.edges: #counting
            neighbors[node1] += 1
            neighbors[node2] += 1

        max_color = max(neighbors.values())
        last_node = 1

        #We generate a matrix of literals (rows = vertex, columns = colors)
        for i in range(self.n_nodes + 1):
            temporal = []
            for j in range(max_color):
                temporal.append(last_node + j)
                formula.new_var()
            matrix.append(temporal)
            last_node = temporal[-1] + 1

        #soft(1): Every new color has a weight of the previous color plus one
        weight = 1
        for node in matrix[-1]:
            formula.add_clause([-node], weight)
            weight += 1

        #hard(1): One color for each vertex
        for row in matrix[:-1]:
            formula.add_exactly_one(row)

        #hard(2): No neighbors share the same color
        for node1, node2 in self.edges:
            for i in range(len(matrix[1])):
                formula.add_at_most_one([matrix[node1-1][i], matrix[node2-1][i]])

        #hard(3): A boolean is true if and only if the color is set
        for row in matrix[:-1]:
            for i, item in enumerate(row):
                formula.add_clause([matrix[-1][i], -item], wcnf.TOP_WEIGHT)

        opt, model = solver.solve(formula)
        curr_opt = opt

        if opt <= 0: return []

        while (n_solutions < 1 and curr_opt == opt) or (n_solutions > 0 and curr_opt == opt and counter < n_solutions):
            counter += 1
            solution.clear()
            interpretation = [x for x in model if x > 0]
            for i in range(len(interpretation)):
                for row in matrix [:-1]:
                    if interpretation[i] in row:
                        if row.index(interpretation[i]) not in solution.keys():
                            solution[row.index(interpretation[i])] = []
                        solution[row.index(interpretation[i])].append(matrix.index(row) + 1)

            if list(solution.values()) not in all_solutions:
                all_solutions.append(list(solution.values()))
            formula.add_clause(list(map(lambda x: -x, model)), wcnf.TOP_WEIGHT)
            curr_opt, model = solver.solve(formula)


        return all_solutions

# Program main
###############################################################################

def main(argv=None):
    args = parse_command_line_arguments(argv)

    solver = msat_runner.MaxSATRunner(args.solver)
    graph = Graph(args.graph)

    mds_all = graph.min_dominating_set(solver, args.n_solutions)
    assert all(len(mds_all[0]) == len(x) for x in mds_all)

    mis_all = graph.max_independent_set(solver, args.n_solutions)
    assert all(len(mis_all[0]) == len(x) for x in mis_all)

    mgc_all = graph.min_graph_coloring(solver, args.n_solutions)
    assert all(len(mgc_all[0]) == len(x) for x in mgc_all)

    print("INDEPENDENT DOMINATION NUMBER", len(mds_all[0]))
    for mds in mds_all:
        print("MDS", " ".join(map(str, mds)))

    print("INDEPENDENCE NUMBER", len(mis_all[0]))
    for mis in mis_all:
        print("MIS", " ".join(map(str, mis)))

    print("CHROMATIC INDEX", len(mgc_all[0]))
    for mgc in mgc_all:
        nodes = (" ".join(map(str, x)) for x in mgc)
        print("GC", " | ".join(nodes))

# Utilities
###############################################################################

def parse_command_line_arguments(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("solver", help="Path to the MaxSAT solver.")

    parser.add_argument("graph", help="Path to the file that descrives the"
                                      " input graph.")

    parser.add_argument("-n", "--n-solutions", type=int, default=1,
                        help="Number of solutions to compute, 0 or negative"
                             " means all solutions.")

    return parser.parse_args(args=argv)


# Entry point
###############################################################################

if __name__ == "__main__":
    sys.exit(main())
