import numpy as np
import pandas as pd
import networkx as nx
import pytest

from causalexplain.generators import generators
from causalexplain.generators.mechanisms import LinearMechanism


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


@pytest.fixture
def deterministic_graph(monkeypatch):
    """Force adjacency sampling to succeed by deterministically selecting edges."""

    def fake_randint(low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        return low + 1 if (high - low) > 1 else low

    def fake_choice(seq, size, replace=False):
        seq = list(seq)
        size = int(size)
        if size <= 0:
            return np.array([], dtype=int)
        return np.array(seq[-size:])

    monkeypatch.setattr(generators.np.random, "randint", fake_randint)
    monkeypatch.setattr(generators.np.random, "choice", fake_choice)


def test_init_variables_creates_mechanisms(deterministic_graph):
    gen = generators.AcyclicGraphGenerator(
        causal_mechanism="linear", nodes=3, parents_max=2, points=10
    )
    gen.init_variables()
    assert isinstance(gen.g, nx.DiGraph)
    # Leaf node should use linear mechanism, roots keep initial generator callable
    assert any(isinstance(func, LinearMechanism) for func in gen.cfunctions)


def test_generate_initializes_on_demand(monkeypatch):
    gen = generators.AcyclicGraphGenerator("linear", nodes=3, points=5)

    def fake_init():
        gen.adjacency_matrix = np.zeros((gen.nodes, gen.nodes))
        gen.g = nx.DiGraph(gen.adjacency_matrix)
        gen.cfunctions = [lambda n, verbose=False: np.arange(n) for _ in range(gen.nodes)]

    monkeypatch.setattr(gen, "init_variables", fake_init)
    graph, data = gen.generate(rescale=False)
    assert graph.number_of_nodes() == 3
    assert (data.iloc[:, 0] == np.arange(5)).all()


def test_generate_rescales_columns(deterministic_graph):
    gen = generators.AcyclicGraphGenerator("linear", nodes=3, parents_max=2, points=30)
    gen.init_variables()
    _, data = gen.generate(rescale=True)
    assert pytest.approx(data.iloc[:, 0].mean(), abs=1e-6) == 0


def test_init_variables_handles_small_graphs_without_recursion():
    gen = generators.AcyclicGraphGenerator("linear", nodes=5, parents_max=3, points=20)
    gen.init_variables()
    assert isinstance(gen.g, nx.DiGraph)
    indegrees = gen.adjacency_matrix.sum(axis=0)
    assert indegrees.max() <= 3


def test_init_variables_clamps_impossible_parent_counts():
    gen = generators.AcyclicGraphGenerator("linear", nodes=2, parents_max=5, points=10)
    gen.init_variables()
    assert gen.parents_max == 1
    indegrees = gen.adjacency_matrix.sum(axis=0)
    assert indegrees.max() <= 1


def test_to_csv_requires_generated_data(tmp_path):
    gen = generators.AcyclicGraphGenerator("linear")
    gen.data = None
    with pytest.raises(ValueError):
        gen.to_csv(str(tmp_path / "sample"))


def test_to_csv_writes_all_artifacts(tmp_path):
    gen = generators.AcyclicGraphGenerator("linear", nodes=2, points=3)
    gen.data = pd.DataFrame({"V0": [0, 1], "V1": [1, 2]})
    gen.adjacency_matrix = np.array([[0, 1], [0, 0]])
    gen.g = nx.DiGraph(gen.adjacency_matrix)
    gen.to_csv(str(tmp_path / "dataset"))
    assert (tmp_path / "dataset_data.csv").exists()
    assert (tmp_path / "dataset_target.csv").exists()
    assert (tmp_path / "dataset_target.dot").exists()
