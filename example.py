from ocik import Asia, Room, Circuit
from ocik import CausalLeaner
from graphviz import Digraph, Graph
import pandas as pd


def draw(edge, directed=True):
    dot = Digraph(graph_attr={'rankdir': 'LR'}) if directed else Graph()
    dot.edges(edge)
    return dot


def difference(gt, pred):
    '''
    Difference between directed graphs
    :param gt: ground truth graph
    :param pred: predicted graph
    :return: the difference graph
    '''
    f = Digraph(graph_attr={'rankdir': 'LR'})
    new_edges = [ed for ed in pred if ed not in gt]  # DOC predicted edge not in ground truth
    f.attr('edge', color='blue')
    f.edges(new_edges)

    missed_edges = [ed for ed in gt if ed not in pred]  # DOC edge in ground truth not predicted
    f.attr('edge', color='red')
    f.edges(missed_edges)

    recovered_edges = [ed for ed in pred if ed in gt]  # DOC predicted edge in ground truth
    f.attr('edge', color='green')
    f.edges(recovered_edges)
    return f


room = Room()
bn = room.get_network()
# obs_data = bn.sample(5000)
obs_data = pd.read_csv('tmp/room.csv')

estimator = CausalLeaner(bn.nodes(), non_dobale=['L', 'T'], env=bn, obs_data=obs_data)
model, trace = estimator.learn(max_cond_vars=4, do_size=100, trace=True, verbose=True)

dot = difference(bn.edges(), model.edges())
dot.view(directory='tmp/')
