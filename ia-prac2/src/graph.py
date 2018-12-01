#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import itertools
import sys

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
        for n1, n2 in self.edges:
            v1, v2 = n_vars[n1-1], n_vars[n2-1]
            formula.add_clause([v1, v2], wcnf.TOP_WEIGHT)

        # this is just an example, only one solution is computed
        all_solutions = []
        opt, model = solver.solve(formula)
        if opt >= 0:
            solution = [x for x in range(1, self.n_nodes + 1) if model[x-1] > 0]
            all_solutions.append(solution)

        return all_solutions

    def min_dominating_set(self, solver, n_solutions):
        """Computes the minimum dominating set of the graph.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of nodes.
        """
        raise NotImplementedError("Your Code Here")

    def max_independent_set(self, solver, n_solutions):
        """Computes the maximum independent set of the graph.

        :param solver: Path to a MaxSAT solver.
        :param n_solutions: Number of solutions to compute, 0 or negative
            means all posible solutions.
        :return: A list of solutions, each solution is a list of nodes.
        """
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
        raise NotImplementedError("Your Code Here")


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
