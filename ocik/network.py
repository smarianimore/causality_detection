import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm


def mask(df, key, value):
    return df[df[key] == value]


pd.DataFrame.mask = mask


class BayesianNetwork:
    def __init__(self, edges):
        self.G = nx.DiGraph(edges)
        self.node_sort = list(nx.topological_sort(self.G))
        self.name = lambda k, v: f"{k}={v}"  # pour nommer les variables
        self.initialize_cpd()

    def nodes(self):
        return list(self.G.nodes())

    def edges(self):
        return list(self.G.edges())

    def _build_cpd(self, node: str):
        """
        create the cpd table for a node in the graph
        :param node: node name
        :return: parent[list], cpd[Dataframe]
        """
        parent = list(nx.neighbors(self.G.reverse(), node))
        node_1 = self.name(node, 1)
        node_0 = self.name(node, 0)
        if len(parent) == 0:
            return parent, pd.DataFrame.from_dict({node_1: [0], node_0: [0]})
        else:
            n_par = len(parent)
            n = 2 ** n_par  # number of combinations
            binaries = [f"%0{n_par}d" % int(format(i, f'#0{n}b')[2:]) for i in range(n)]  # write in binary
            combination = {parent[i]: [int(b[i]) for b in binaries] for i in range(len(parent))}
            combination[node_0] = [0] * n
            combination[node_1] = [0] * n
            cpd = pd.DataFrame.from_dict(combination)
            cpd = cpd.set_index(parent)
            return parent, cpd

    def initialize_cpd(self):
        """
        create cpd table for all nodes and initialize probabitity with zero
        """
        for node in self.G.nodes():
            par, cpd = self._build_cpd(node)
            self.G.add_node(node, par=par, cpd=cpd)

    def get_cpd_count(self, node: str):
        """
        retourne les occurences des différentes etats obtenues à partir des données
        :param node: the node name in the graph
        :return: parent [list], cpd [dataframe]
        """
        par, cpd = self.G.nodes(data=True)[node].values()
        return par, cpd.copy()

    def get_cpd(self, node: str):
        """
        retourne le tableau de probabilité conditionnelle d'un noeud
        :param node: the node name in the graph
        :return: parent [list], cpd [dataframe]
        """
        par, cpd_count = self.get_cpd_count(node)

        node_1 = self.name(node, 1)
        node_0 = self.name(node, 0)
        cpd = cpd_count.copy()

        cpd[node_1] = cpd_count.apply(
            lambda x: 0 if x[node_1] + x[node_0] == 0 else x[node_1] / (x[node_1] + x[node_0]), axis=1)
        cpd[node_0] = cpd_count.apply(
            lambda x: 0 if x[node_1] + x[node_0] == 0 else x[node_0] / (x[node_1] + x[node_0]), axis=1)
        return par, cpd

    def _set_cpd(self, node, cpd):
        self.G.nodes(data=True)[node]["cpd"] = cpd

    def __call__(self, *args, **kwargs):
        return self.G.edges()

    def set_cpd(self, node: str, cpd_prob: list, parent: list):
        """
        recupère et remplit la table de probabilité conditinnelle
        :param cpd_prob: arary of size [2, nb_of_parent_combination]
        :param parent: list of parent node name
        :return:
        """
        node_1 = self.name(node, 1)
        node_0 = self.name(node, 0)
        if len(parent) == 0:
            self.G.nodes(data=True)[node]["cpd"] = pd.DataFrame.from_dict({node_1: cpd_prob[0], node_0: cpd_prob[1]})
        else:
            n_par = len(parent)
            n = 2 ** n_par  # number of combinations
            binaries = [f"%0{n_par}d" % int(format(i, f'#0{n}b')[2:]) for i in range(n)]  # write in binary
            combination = {parent[i]: [int(b[i]) for b in binaries] for i in range(len(parent))}
            combination[node_1] = cpd_prob[0]
            combination[node_0] = cpd_prob[1]
            cpd = pd.DataFrame.from_dict(combination)
            cpd = cpd.set_index(parent)
            self.G.nodes(data=True)[node]["cpd"] = cpd

    def copy(self):
        """
        duplinamee the bayesian network
        """
        bn = BayesianNetwork(self.G.edges())
        for node in bn.G.nodes():
            _, bn.G.nodes(data=True)[node]["cpd"] = self.get_cpd_count(node)
        return bn

    def _sample(self, evidence: dict = None):
        """
        generate one simulation data. propagate data from node without parent to all the network
        """
        if evidence is None:
            evidence = {}
        else:
            evidence = evidence.copy()

        node_available = set([]) | set(evidence.keys())
        to_compute = set(self.G.nodes()) - set(evidence.keys())  # DOC: non-conditioned nodes

        data = dict(self.G.nodes(data=True))
        #print(f'\t\t\t\tnodes data: {data}')  # NB: this method is executed many many times, print carefully!
        dependancies = {node: d for node, d in data.items()}  # QUESTION: are d the parents of node?

        for node in self.node_sort:
            if node in to_compute:
                par, cpd = dependancies[node]['par'], dependancies[node]['cpd']

                # raise if topological sort fails
                assert node_available & set(par) == set(par), "parent evidence not available"

                node_1 = self.name(node, 1)
                node_0 = self.name(node, 0)
                if len(dependancies[node]['par']) == 0:  # QUESTION: no parents?
                    prob_1, prob_0 = cpd[node_1][cpd.index[0]], cpd[node_0][cpd.index[0]]
                else:
                    select = tuple(evidence[p] for p in par)
                    prob = cpd.loc[select]
                    prob_1, prob_0 = prob[node_1], prob[node_0]

                evidence[node] = np.random.choice([1, 0], p=[prob_1, prob_0])  # QUESTION: hard-coded simulated values?
                node_available = node_available | {node}
                to_compute = to_compute - {node}

        assert len(to_compute) == 0, "not every node have been sample"

        return pd.DataFrame({k: [v] for k, v in evidence.items()})

    def sample(self, size=20):

        df = pd.DataFrame({node: [] for node in sorted(self.G.nodes())}).astype(int)
        for _ in tqdm(range(size)):
            df = pd.concat((df, self._sample()))
        df.reset_index(drop=True, inplace=True)
        return df

    def do_evidence(self, evidence=None, size=100, seed=12):
        np.random.seed(seed)

        if evidence is None:
            evidence = {}

        df = pd.DataFrame({node: [] for node in sorted(self.G.nodes())}).astype(int)
        for i in range(size):
            df = pd.concat((df, self._sample(evidence)))
        #print(f'\t\t\tsimulated data: {df.head()}')  # DOC: simulated values (0/1 in this case)
        df.reset_index(drop=True, inplace=True)

        return df

    def do(self, node, size=15):
        n_node = len(node)
        n = 2 ** n_node  # number of combinations
        binaries = [f"%0{n_node}d" % int(format(i, f'#0{n}b')[2:]) for i in range(n)]  # write in binary
        combination = [{node[i]: int(b[i]) for i in range(len(node))} for b in binaries]
        df = pd.concat([self.do_evidence(evidence=ev, size=size) for ev in combination])
        df.reset_index(drop=True, inplace=True)
        return df