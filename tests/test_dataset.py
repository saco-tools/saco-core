import numpy as np
import networkx as nx

from saco import Dataset


DATA_FOLDER = './tests/data/01/base'


def test_infer_percentile_impact(data_folder=DATA_FOLDER):
    ds = Dataset(data_folder)
    ds.load_data()

    ds.gwabs.data['GWQ95FLWR__ORIGINAL'] = ds.gwabs.data['GWQ95FLWR']
    ds.swabs.data['SWQ95FLWR__ORIGINAL'] = ds.swabs.data['SWQ95FLWR']

    ds.infer_percentile_impact(scenarios=['FL'], percentiles=[95])

    assert np.all(np.abs(ds.gwabs.data['GWQ95FLWR'] - ds.gwabs.data['GWQ95FLWR__ORIGINAL']) < 1e-02)

    mask = ds.swabs.data['HOFMLD'] > 0.0
    df = ds.swabs.data.loc[~mask]
    df = df.loc[np.abs(df['SWQ95FLWR'] - df['SWQ95FLWR__ORIGINAL']) >= 1e-02]
    assert df.shape[0] == 0


def test_find_outlet_waterbodies_on_acyclic_graph():
    ds = Dataset()
    ds.set_graph(nx.DiGraph([('upstream', 'outlet')]))

    assert ds.find_outlet_waterbodies(assessed_outlets_only=False) == ['outlet']


def test_find_outlet_waterbodies_preserves_cyclic_graph_behavior():
    graph = nx.DiGraph([('cycle-a', 'cycle-b'), ('cycle-b', 'cycle-a')])
    graph.add_edge('self-loop', 'self-loop')
    graph.add_node('outlet')
    ds = Dataset()
    ds.set_graph(graph)

    assert ds.find_outlet_waterbodies(assessed_outlets_only=False) == [
        'self-loop', 'outlet',
    ]
