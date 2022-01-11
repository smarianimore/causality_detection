import networkx as nx

from scipy.stats import chisquare
from pgmpy.estimators.CITests import chi_square as ci_chi2
from itertools import combinations
from warnings import warn
from pgmpy.base import PDAG


def simulate(src: str, dst: list, env, cond_evidence: dict = {}, do_size=100):
    """
    fix src node value and do many simulation and return the data
    """
    result = {}
    evidences = [{src: value, **cond_evidence} for value in [0, 1]]  # QUESTION: is the hardcoded value the admissible observed value (on/off)?
    print(f'\t\tevidencies for {src}: {evidences}')

    for i, evidence in enumerate(evidences):
        do_result = env.do_evidence(evidence=evidence, size=do_size)

        ev = evidence[src]
        result[ev] = {}
        for node in dst:
            counts = do_result[node].value_counts()  # DOC: count occurrences of values (0/1 in this case) simulated by do_evidence()
            a = counts[0] + 1 if 0 in counts.index else 1
            b = counts[1] + 1 if 1 in counts.index else 1  # for a beta dist we need to add 1
            # _, _, mod = beta_dist(a, b)
            result[ev][node] = [a, b]

    return result


def has_influence(src: str,  # DOC: node to intervene on
                  dst: list,  # DOC: nodes to check whether src has influence on
                  cond_node: list,  # DOC: list of conditioning nodes
                  env,  # DOC: whole Bayesian network (with CPD)
                  do_size=200,
                  alpha=0.05):
    """
    test if src node have causal effect on dst node
    """
    n_node = len(cond_node)
    n = 2 ** n_node  # number of combinations
    binaries = [f"%0{n_node}d" % int(format(i, f'#0{n}b')[2:]) for i in range(n)]  # write in binary QUESTION: what is?
    print(f'\tbinaries = {binaries}')
    conditional_evidences = [{cond_node[i]: int(b[i]) for i in range(len(cond_node))} for b in binaries]
    print(f'\tconditional_evidences for {src} on {dst} = {conditional_evidences}')  # QUESTION: are these the combinations of values for conditioning variables?
    test = {node: [] for node in dst}
    for cond_evidence in conditional_evidences:
        res = simulate(src, dst, env, cond_evidence, do_size=do_size)
        #  QUESTION: is res format
        #       {src_node_val: {dst_node_1: [#_times_0, #_times_1], ...}}
        #       ?
        print(f'\tres of {cond_evidence} = {res}')
        for node in dst:
            # QUESTION: what is the hardcoded index? is it the value assumed by the variable? what if it is not boolean?
            test[node].append(chisquare(
                f_obs=res[0][node],  # DOC: observed distribution
                f_exp=res[1][node]  # DOC: expected distribution
            )[1] <= alpha)  # DOC: p-value of test <= chosen confidence interval
    influence = {node: any(test[node]) for node in dst}
    return influence


class CausalLeaner:
    def __init__(self, nodes: list, non_dobale: list, env, obs_data=None):
        self.env = env
        self.nodes = nodes
        self.non_doable = non_dobale
        self.obs_data = obs_data

        if len(non_dobale) != 0 and obs_data is None:
            raise Exception('with non doable variables we need observational data')

    def learn(self, do_size=100,
              do_conf=0.4,
              ci_conf=0.1,
              max_cond_vars=4,  # DOC: max # of vars to condition (incrementally, through lim_neighbours = Order)
              post=True, trace=False, verbose=False):
        # Initialize initial values and structures.
        ci_test = ci_chi2
        lim_neighbors = 0
        separating_sets = dict()

        # Step 1: Initialize a fully connected directed graph
        completeG = nx.complete_graph(n=self.nodes, create_using=nx.Graph)
        G = nx.DiGraph(list(completeG.edges()))
        graph = nx.DiGraph(list(G.edges()) + list(G.reverse().edges()))

        non_doable = set(self.non_doable)
        doable = set(graph.nodes) - non_doable

        if verbose:
            print('doable:', doable)
            print('non-doable:', non_doable)

        if trace:
            track = [list(graph.edges())]

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
                [len(list(graph.neighbors(var))) < lim_neighbors for var in self.nodes]
        ):

            if verbose: print("Order : ", lim_neighbors)
            #######################################
            #  is there a causal relationship?    #
            #######################################
            if lim_neighbors == 0:  # DOC start without conditioning neighbours
                #  precompute influence between node  #
                influence = {}
                for node in doable:
                    influence[node] = has_influence(node, list(graph.neighbors(node)),  # DOC influence of node on neighbours
                                                    [], self.env, do_size=do_size,  # DOC empty list -> no conditioning nodes
                                                    alpha=do_conf)

                print(f'[k = 0 case] influence: {influence}')

                #  delete node base on influence result  #
                edges = list(graph.edges())
                for i, (X, Y) in enumerate(edges):
                    if X in doable:  # DOC check influence of do-able nodes
                        X2Y = influence[X][Y]
                        if not X2Y and graph.has_edge(X, Y):
                            graph.remove_edge(X, Y)
                            print('do', X, Y, 'remove')
                    else:  # DOC check influence of NON do-able nodes
                        if ci_test(X, Y, [], data=self.obs_data, significance_level=ci_conf):
                            graph.remove_edge(X, Y)
                            print('ci', X, Y, 'remove')

            #######################################
            #      what if we block k path        #
            #######################################
            elif lim_neighbors > 0:  # DOC condition K = lim_neighbours neighbours
                #  precompute influence between node  #
                influence = {}
                for node in doable:
                    neigh = set(graph.neighbors(node))
                    all_sep_set = set()
                    for v in neigh:  # DOC for each neighbour v, combinations of K do-able other (not v) neighbours
                        all_sep_set = all_sep_set | set(combinations(set(graph.neighbors(node))
                                                                     - {v} - non_doable, lim_neighbors))

                    print(f'node = {node} => all_sep_set = {list(all_sep_set)}')
                    influence[node] = {}
                    for sep_set in list(all_sep_set):  # DOC for each combination
                        on = neigh - set(sep_set)
                        if len(on) > 0:  # DOC check influence of node on remaining neighbours (no sep_set) conditioning on sep_set
                            influence[node][sep_set] = has_influence(node, list(on), sep_set, self.env,
                                                                     do_size=do_size, alpha=do_conf)

                print(f'[k > 0 case] influence: {influence}')

                #  delete node base on influence result  #
                edges = list(graph.edges())
                for (X, Y) in edges:
                    if X not in non_doable:
                        for separating_set in list(
                                set(combinations(set(graph.neighbors(X)) - {Y}, lim_neighbors))
                        ):
                            if len(non_doable & set(separating_set)) == 0:
                                print('[doable case] test :', X, '->', Y, 'sep:', separating_set)
                                # If a conditioning set exists remove the edge, store the
                                # separating set and move on to finding conditioning set for next edge.
                                the_sep_set = None
                                for sep_set in influence[X].keys():
                                    if set(sep_set) == set(separating_set):
                                        the_sep_set = sep_set
                                        break

                                assert the_sep_set is not None

                                X2Y = influence[X][the_sep_set][Y]
                                if not X2Y and graph.has_edge(X, Y):
                                    graph.remove_edge(X, Y)
                                    print('do', X, Y, 'remove')
                            else:
                                if ci_test(X, Y, separating_set, data=self.obs_data, significance_level=ci_conf) \
                                        and graph.has_edge(X, Y):
                                    graph.remove_edge(X, Y)
                                    print('ci', X, Y, 'remove')
                                    break

                    else: # DOC node X is NOT do-able
                        undir_graph = graph.to_undirected()
                        for separating_set in list(
                                set(combinations(set(undir_graph.neighbors(X)) - {Y}, lim_neighbors)) | \
                                set(combinations(set(undir_graph.neighbors(Y)) - {X}, lim_neighbors))
                        ):
                            if ci_test(X, Y, separating_set, data=self.obs_data, significance_level=ci_conf) \
                                    and graph.has_edge(X, Y):
                                graph.remove_edge(X, Y)
                                print('[non-doable case] ci', X, Y, 'remove')
                                break

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                warn("Reached maximum number of allowed conditional variables. Exiting")
                break

            if trace:
                track.append(list(graph.edges()))

            lim_neighbors += 1

        G = self.postprocessing(graph) if post else graph
        if trace:
            track.append(list(G.edges()))

        if not trace:
            return G
        else:
            return G, track  # add last graph

    def postprocessing(self, graph: nx.DiGraph):
        non_decidable_arrow = []
        for src in self.non_doable:
            non_decidable_arrow += [(src, dst) for dst in list(graph.neighbors(src))]

        undirected_edge = set()
        for (src, dst) in non_decidable_arrow:
            # two undecidable case
            if (dst, src) not in undirected_edge and (dst, src) in non_decidable_arrow:
                undirected_edge = undirected_edge | {(src, dst)}
                graph.remove_edge(src, dst) if graph.has_edge(src, dst) else None
                graph.remove_edge(dst, src) if graph.has_edge(dst, src) else None

            # one decidable with one undecidable
            elif (dst, src) not in non_decidable_arrow and graph.has_edge(dst, src):
                graph.remove_edge(src, dst)
        print('undirected', undirected_edge)
        if len(undirected_edge) == 0:
            return graph
        else:
            pdag = PDAG(directed_ebunch=list(graph.edges()), undirected_ebunch=list(undirected_edge))
            return pdag.to_dag(required_edges=list(graph.edges()))
