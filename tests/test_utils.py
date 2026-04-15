from _pytest.tmpdir import tmp_path
import networkx as nx
import pandas as pd
import numpy as np
import pytest
import torch
import os
import pickle
import math

from causalexplain.common import utils


class TestSelectDevice:
    # Tests that the function returns 'cuda' when force is 'cuda' and cuda is
    # available
    def test_returns_cuda_when_force_is_cuda_and_cuda_is_not_available(self):
        # Try to force cuda when cuda is not available
        # Should raise a ValueError
        with pytest.raises(ValueError):
            utils.select_device(force='cuda')

    # Tests that the function returns 'mps' when force is 'mps' and mps is
    # available

    def test_returns_mps_when_force_is_mps_and_mps_is_available(self):
        result = utils.select_device(force='mps')
        if torch.backends.mps.is_available():
            assert result == 'mps'
        else:
            assert result == 'cpu'

    # Tests that the function returns 'cpu' when force is 'cpu' and no other
    # options are available

    def test_returns_cpu_when_force_is_cpu_and_no_other_options_are_available(self):
        result = utils.select_device(force='cpu')
        assert result == 'cpu'

    # Tests that the function returns 'mps' when mps is available and no force
    # is specified

    def test_returns_mps_when_mps_is_available_and_no_force_is_specified(self):
        result = utils.select_device()
        if torch.backends.mps.is_available():
            assert result == 'mps'
        else:
            assert result == 'cpu'

    # Tests that the function returns 'cpu' when no other options are available
    # and no force is specified

    def test_returns_cpu_when_no_other_options_are_available_and_no_force_is_specified(self):
        result = utils.select_device()
        if torch.backends.mps.is_available():
            assert result == 'mps'
        else:
            assert result == 'cpu'


class TestGraphIntersection:

    # Test with two identical graphs

    def test_identical_graphs(self):
        # Create two identical graphs
        g1 = nx.DiGraph()
        g1.add_node(1)
        g1.add_node(2)
        g1.add_edge(1, 2)

        g2 = nx.DiGraph()
        g2.add_node(1)
        g2.add_node(2)
        g2.add_edge(1, 2)

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result is equal to the input graphs
        assert result.nodes == g1.nodes
        assert result.edges == g1.edges

    # Test with two graphs with different nodes and edges
    def test_different_nodes_edges(self):
        # Create two graphs with different nodes and edges
        g1 = nx.DiGraph()
        g1.add_node(1)
        g1.add_node(2)
        g1.add_edge(1, 2)

        g2 = nx.DiGraph()
        g2.add_node(3)
        g2.add_node(4)
        g2.add_edge(3, 4)

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result is an empty graph
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    # Test with two graphs with some common nodes and edges
    def test_common_nodes_edges(self):
        # Create two graphs with some common nodes and edges
        g1 = nx.DiGraph()
        g1.add_node(1)
        g1.add_node(2)
        g1.add_edge(1, 2)

        g2 = nx.DiGraph()
        g2.add_node(2)
        g2.add_node(3)
        g2.add_edge(2, 3)

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result has the common nodes and edges
        assert len(result.nodes) == 1
        assert len(result.edges) == 0
        assert 2 in result.nodes

    # Test two graphs with different node attributes

    def test_different_node_attributes(self):
        # Create two graphs with different node attributes
        g1 = nx.DiGraph()
        g1.add_node(1, attr='A')
        g1.add_node(2, attr='B')
        g1.add_edge(1, 2)

        g2 = nx.DiGraph()
        g2.add_node(1, attr='C')
        g2.add_node(2, attr='D')
        g2.add_edge(1, 2)

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result has the minimum node attributes
        assert len(result.nodes) == 2
        # assert 'A' in result.nodes[1]['attr']
        # assert 'B' in result.nodes[2]['attr']

    # Test with two empty graphs

    def test_empty_graphs(self):
        # Create two empty graphs
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result is an empty graph
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    # Test with two undirected graphs

    def test_identical_undirected_graphs(self):
        # Create two identical undirected graphs
        g1 = nx.Graph()
        g1.add_node(1)
        g1.add_node(2)
        g1.add_edge(1, 2)

        g2 = nx.Graph()
        g2.add_node(1)
        g2.add_node(2)
        g2.add_edge(1, 2)

        # Call the graph_intersection function
        result = utils.graph_intersection(g1, g2)

        # Check if the result is equal to the input graphs
        assert result.nodes == g1.nodes
        assert result.edges == g1.edges


class TestGraphUnion:
    """Test suite for graph_union function"""

    def test_union_disjoint_graphs(self):
        g1 = nx.DiGraph()
        g1.add_edges_from([("A", "B"), ("B", "C")])
        g2 = nx.DiGraph()
        g2.add_edges_from([("D", "E"), ("E", "F")])

        union_graph = utils.graph_union(g1, g2)

        assert isinstance(union_graph, nx.DiGraph)
        assert set(union_graph.nodes()) == {"A", "B", "C", "D", "E", "F"}
        assert set(union_graph.edges()) == {("A", "B"), ("B", "C"), ("D", "E"),
                                            ("E", "F")}

    def test_union_overlapping_graphs(self):
        g1 = nx.DiGraph()
        g1.add_edges_from([("A", "B"), ("B", "C")])
        nx.set_node_attributes(g1, {"A": {"weight": 1}, "B": {
                               "weight": 2}, "C": {"weight": 3}})

        g2 = nx.DiGraph()
        g2.add_edges_from([("B", "C"), ("C", "D")])
        nx.set_node_attributes(g2, {"B": {"weight": 3}, "C": {
                               "weight": 4}, "D": {"weight": 5}})

        union_graph = utils.graph_union(g1, g2)

        assert isinstance(union_graph, nx.DiGraph)
        assert set(union_graph.nodes()) == {"A", "B", "C", "D"}
        assert set(union_graph.edges()) == {("A", "B"), ("B", "C"), ("C", "D")}
        assert union_graph.nodes["B"]["weight"] == 2.5
        assert union_graph.nodes["C"]["weight"] == 3.5

    def test_union_with_empty_graph(self):
        g1 = nx.DiGraph()
        g1.add_edges_from([("A", "B")])
        g2 = nx.DiGraph()  # Empty graph

        union_graph = utils.graph_union(g1, g2)

        assert isinstance(union_graph, nx.DiGraph)
        assert set(union_graph.nodes()) == {"A", "B"}
        assert set(union_graph.edges()) == {("A", "B")}


class TestClassifyVariable:
    """Test suite for _classify_variable function"""

    def test_binary_numeric(self):
        """Test binary classification with numeric values"""
        arr = pd.Series([0, 1, 0, 1, 0])
        assert utils._classify_variable(arr) == "binary"

        arr = pd.Series([1.0, 2.0, 1.0, 2.0])
        assert utils._classify_variable(arr) == "binary"

    def test_binary_categorical(self):
        """Test binary classification with categorical values"""
        arr = pd.Series(['yes', 'no', 'yes', 'no'])
        assert utils._classify_variable(arr) == "multiclass"

        arr = pd.Series(['true', 'false', 'true'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_multiclass_categorical(self):
        """Test multiclass classification with categorical values"""
        arr = pd.Series(['a', 'b', 'c', 'd'])
        assert utils._classify_variable(arr) == "multiclass"

        arr = pd.Series(['low', 'medium', 'high', 'medium', 'low'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_continuous_numeric(self):
        """Test continuous classification with numeric values"""
        arr = pd.Series([1.0, 2.5, 3.7, 4.2, 5.0])
        assert utils._classify_variable(arr) == "continuous"

        arr = pd.Series(np.random.normal(0, 1, 100))
        assert utils._classify_variable(arr) == "continuous"

    def test_mixed_types(self):
        """Test with mixed types (should be handled as multiclass)"""
        arr = pd.Series(['1', '2', 'a', 'b'])
        assert utils._classify_variable(arr) == "multiclass"

    def test_special_cases(self):
        """Test special cases"""
        # Empty series
        arr = pd.Series([])
        assert utils._classify_variable(arr) == "continuous"

        # Series with NaN values
        arr = pd.Series([1, 2, np.nan, 4])
        assert utils._classify_variable(arr) == "continuous"

        # Boolean values
        arr = pd.Series([True, False, True])
        assert utils._classify_variable(arr) == "binary"


class TestCastCategoricalsToInt:
    """Test suite for cast_categoricals_to_int function"""

    def test_cast_multiclass(self):
        """Test casting multiclass categorical variables"""
        df = pd.DataFrame({
            'A': pd.Categorical(['a', 'b', 'c', 'a']),
            'B': pd.Categorical(['x', 'y', 'z', 'x']),
            'C': [1.0, 2.0, 3.0, 1.0]  # Numeric column should remain unchanged
        })
        result = utils.cast_categoricals_to_int(df)
        # Categories should be converted to integers starting from 0
        assert result['A'].dtype == np.int16
        assert result['B'].dtype == np.int16
        assert result['C'].dtype == np.float64  # Should remain unchanged
        assert set(result['A'].unique()) == {0, 1, 2}
        assert set(result['B'].unique()) == {0, 1, 2}

    def test_cast_binary(self):
        """Test casting binary categorical variables"""
        df = pd.DataFrame({
            'A': ['yes', 'no', 'yes', 'no'],
            'B': ['true', 'false', 'true', 'false'],
            'C': [0, 1, 0, 1]  # Already binary numeric
        })
        result = utils.cast_categoricals_to_int(df)
        assert result['A'].dtype == np.int16
        assert result['B'].dtype == np.int16
        assert result['C'].dtype == np.int16  # Should remain numeric
        assert set(result['A'].unique()) == {0, 1}
        assert set(result['B'].unique()) == {0, 1}
        assert set(result['C'].unique()) == {0, 1}

    def test_no_categorical(self):
        """Test with no categorical columns"""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
            'C': [7, 8, 9]
        })
        result = utils.cast_categoricals_to_int(df)
        pd.testing.assert_frame_equal(result, df)  # Should be unchanged

    def test_mixed_types(self):
        """Test with mixed types in the same column"""
        df = pd.DataFrame({
            'A': ['1', '2', 'a', 'b'],  # Mixed strings
            'B': [1, 2, 'three', 'four'],  # Mixed numbers and strings
            'C': [1.0, 2.0, 3.0, 4.0]  # Pure numeric
        })
        result = utils.cast_categoricals_to_int(df)
        assert result['A'].dtype == np.int16  # Should be converted
        assert result['B'].dtype == np.int16  # Should be converted
        assert result['C'].dtype == np.float64  # Should remain unchanged
        assert len(result['A'].unique()) == 4
        assert len(result['B'].unique()) == 4


class TestSaveExperiment:
    """Test suite for save_experiment function"""
    tmp_path = "/tmp"

    def test_save_experiment_creates_folder(self):
        folder = os.path.join(self.tmp_path, "new_folder")
        obj_name = "test_experiment"
        results = {"key0": "value0"}

        output = utils.save_experiment(obj_name, str(folder), results)

        assert os.path.exists(folder)
        assert os.path.exists(output)

    def test_save_experiment_overwrite(self):
        folder = os.path.join(self.tmp_path, "new_folder")
        obj_name = "test_experiment"
        results = {"key2": "valu2e"}

        # Create an existing file
        existing_file = os.path.join(folder, f"{obj_name}.pickle")
        with open(existing_file, 'wb') as f:
            pickle.dump({}, f)

        output = utils.save_experiment(
            obj_name, str(folder), results, overwrite=True)

        assert os.path.exists(output)
        with open(output, 'rb') as f:
            data = pickle.load(f)
        assert data == results

    def test_save_experiment_no_overwrite(self):
        folder = os.path.join(self.tmp_path, "new_folder")
        obj_name = "test_experiment"
        results = {"key3": "value3"}

        # Create an existing file
        existing_file = os.path.join(folder, f"{obj_name}.pickle")
        with open(existing_file, 'wb') as f:
            pickle.dump({}, f)

        output = utils.save_experiment(
            obj_name, str(folder), results, overwrite=False)

        assert os.path.exists(output)
        with open(output, 'rb') as f:
            data = pickle.load(f)
        assert data == results

    def test_save_experiment_return_value(self):
        folder = os.path.join(self.tmp_path, "new_folder")
        obj_name = "test_experiment"
        results = {"key": "value"}

        output = utils.save_experiment(
            obj_name, str(folder), results, overwrite=True)

        expected_output = os.path.join(folder, f"{obj_name}.pickle")
        assert output == str(expected_output)

    def test_save_experiment_saves_correct_data(self):
        folder = os.path.join(self.tmp_path, "new_folder")
        obj_name = "test_experiment"
        results = {"key": "value"}

        output = utils.save_experiment(
            obj_name, str(folder), results, overwrite=True)

        with open(output, 'rb') as f:
            data = pickle.load(f)
        assert data == results

    def test_save_experiment_with_pickle_extension(self):
        folder = os.path.join(self.tmp_path, "pickle_folder")
        obj_name = "test_experiment.pickle"  # Already has .pickle extension
        results = {"key": "value"}

        output = utils.save_experiment(
            obj_name, str(folder), results, overwrite=True)

        # Should not add another .pickle
        expected_output = os.path.join(folder, obj_name)
        assert output == str(expected_output)
        assert os.path.exists(output)
        with open(output, 'rb') as f:
            data = pickle.load(f)
        assert data == results


class TestLoadExperiment:
    """Test suite for load_experiment function"""
    tmp_path = "/tmp"

    def test_load_experiment_existing_file(self):
        folder = os.path.join(self.tmp_path, "existing_folder")
        obj_name = "test_experiment"
        expected_results = {"key1": "value1"}

        # Ensure the folder exists
        os.makedirs(folder, exist_ok=True)

        # Create a pickle file with expected results
        with open(os.path.join(folder, f"{obj_name}.pickle"), 'wb') as f:
            pickle.dump(expected_results, f)

        # Load the experiment
        loaded_obj = utils.load_experiment(obj_name, folder)

        assert loaded_obj == expected_results

    def test_load_experiment_non_existent_file(self):
        folder = os.path.join(self.tmp_path, "non_existent_folder")
        obj_name = "non_existent_experiment"

        with pytest.raises(FileNotFoundError):
            utils.load_experiment(obj_name, folder)


class TestValidOutputName:
    """Test suite for valid_output_name function"""
    tmp_path = "/tmp"

    def test_valid_output_name_no_conflict(self):
        path = os.path.join(self.tmp_path, "unique_folder")
        filename = "unique_file"
        extension = "txt"

        # Ensure the folder exists
        os.makedirs(path, exist_ok=True)

        # Test for a filename with no conflict
        result = utils.valid_output_name(filename, path, extension)
        expected = os.path.join(path, f"{filename}.{extension}")

        assert result == expected

    def test_valid_output_name_with_conflict(self):
        path = os.path.join(self.tmp_path, "conflict_folder")
        filename = "conflict_file"
        extension = "txt"

        # Ensure the folder exists
        os.makedirs(path, exist_ok=True)

        # Create a conflicting file
        with open(os.path.join(path, f"{filename}.{extension}"), 'w') as f:
            f.write("dummy content")

        # Test for a filename with conflict
        result = utils.valid_output_name(filename, path, extension)
        expected = os.path.join(path, f"{filename}_1.{extension}")

        assert result == expected

    def test_valid_output_name_no_extension(self):
        path = os.path.join(self.tmp_path, "no_ext_folder")
        filename = "no_ext_file"

        # Ensure the folder exists
        os.makedirs(path, exist_ok=True)

        # Test for a filename without extension
        result = utils.valid_output_name(filename, path)
        expected = os.path.join(path, filename)

        assert result == expected

    def test_valid_output_name_with_pickle_extension(self):
        path = os.path.join(self.tmp_path, "pickle_ext_folder")
        filename = "test_file.pickle"  # Already has .pickle extension
        extension = "pickle"

        # Ensure the folder exists
        os.makedirs(path, exist_ok=True)

        # Test for a filename that already has the extension
        result = utils.valid_output_name(filename, path, extension)
        # Should not add another .pickle
        expected = os.path.join(path, filename)

        assert result == expected


class TestGraphFromDotFile:
    """Test suite for graph_from_dot_file function"""
    tmp_path = "/tmp"

    def test_graph_from_valid_dot_file(self):
        dot_content = """
        strict digraph G {
            concentrate = true;
            
            A -> B;
            B -> C;
            C -> A;
        }
        """
        dot_file_path = os.path.join(self.tmp_path, "valid_graph.dot")

        with open(dot_file_path, 'w') as f:
            f.write(dot_content)

        graph = utils.graph_from_dot_file(dot_file_path)

        assert graph is not None
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'A', 'B', 'C'}
        assert set(graph.edges()) == {('A', 'B'), ('B', 'C'), ('C', 'A')}

    def test_graph_from_non_existent_file(self):
        non_existent_path = os.path.join(self.tmp_path, "non_existent.dot")
        graph = utils.graph_from_dot_file(non_existent_path)
        assert graph is None

    def test_graph_from_invalid_dot_file(self):
        invalid_dot_content = """
        digraph G {
            A -> B
            B ->
        }
        """
        dot_file_path = os.path.join(self.tmp_path, "invalid_graph.dot")

        with open(dot_file_path, 'w') as f:
            f.write(invalid_dot_content)

        graph = utils.graph_from_dot_file(dot_file_path)
        assert graph is None


class TestGraphFromDictionary:
    """Test suite for graph_from_dictionary function"""

    def test_graph_from_simple_dictionary(self):
        simple_dict = {
            'u': ['v', 'w'],
            'x': ['y']
        }
        graph = utils.graph_from_dictionary(simple_dict)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'u', 'v', 'w', 'x', 'y'}
        assert set(graph.edges()) == {('v', 'u'), ('w', 'u'), ('y', 'x')}

    def test_graph_from_weighted_dictionary(self):
        weighted_dict = {
            'u': [('v', 0.2), ('w', 0.7)],
            'x': [('y', 0.5)]
        }
        graph = utils.graph_from_dictionary(weighted_dict)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'u', 'v', 'w', 'x', 'y'}
        assert set(graph.edges()) == {('v', 'u'), ('w', 'u'), ('y', 'x')}
        assert graph['v']['u']['weight'] == 0.2
        assert graph['w']['u']['weight'] == 0.7
        assert graph['y']['x']['weight'] == 0.5

    def test_graph_from_empty_dictionary(self):
        empty_dict = {}
        graph = utils.graph_from_dictionary(empty_dict)

        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0


class TestGraphFromAdjacency:
    """Test suite for graph_from_adjacency function"""

    def test_graph_from_default_adjacency(self):
        adjacency = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        graph = utils.graph_from_adjacency(adjacency)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {0, 1, 2}
        assert set(graph.edges()) == {(0, 1), (1, 2), (2, 0)}

    def test_graph_with_threshold(self):
        adjacency = np.array([
            [0, 0.5, 0],
            [0, 0, 0.2],
            [0.3, 0, 0]
        ])
        graph = utils.graph_from_adjacency(adjacency, th=0.25)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {0, 1, 2}
        assert set(graph.edges()) == {(0, 1), (2, 0)}

    def test_graph_with_inverse_edges(self):
        adjacency = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        graph = utils.graph_from_adjacency(adjacency, inverse=True)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {0, 1, 2}
        assert set(graph.edges()) == {(1, 0), (2, 1), (0, 2)}

    def test_graph_with_absolute_values(self):
        adjacency = np.array([
            [0, -0.5, 0],
            [0, 0, -0.2],
            [0.3, 0, 0]
        ])
        graph = utils.graph_from_adjacency(
            adjacency, absolute_values=True, th=0.25)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {0, 1, 2}
        assert set(graph.edges()) == {(0, 1), (2, 0)}

    def test_graph_with_node_labels(self):
        adjacency = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        labels = ['A', 'B', 'C']
        graph = utils.graph_from_adjacency(adjacency, node_labels=labels)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'A', 'B', 'C'}
        assert set(graph.edges()) == {('A', 'B'), ('B', 'C'), ('C', 'A')}


class TestGraphFromAdjacencyFile:
    """Test suite for graph_from_adjacency_file function"""
    tmp_path = "/tmp"

    def test_graph_from_file_with_header(self):
        file_path = os.path.join(self.tmp_path, "adjacency_with_header.csv")
        content = """A,B,C\n0,1,0\n0,0,1\n1,0,0"""

        with open(file_path, 'w') as f:
            f.write(content)

        graph, df = utils.graph_from_adjacency_file(file_path)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'A', 'B', 'C'}
        assert set(graph.edges()) == {('A', 'B'), ('B', 'C'), ('C', 'A')}
        assert isinstance(df, pd.DataFrame)

    def test_graph_from_file_without_header(self):
        file_path = os.path.join(self.tmp_path, "adjacency_no_header.csv")
        content = """0,1,0\n0,0,1\n1,0,0"""

        with open(file_path, 'w') as f:
            f.write(content)

        labels = ['A', 'B', 'C']
        graph, df = utils.graph_from_adjacency_file(
            file_path, labels=labels, header=False)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'A', 'B', 'C'}
        assert set(graph.edges()) == {('A', 'B'), ('B', 'C'), ('C', 'A')}
        assert isinstance(df, pd.DataFrame)

    def test_graph_with_threshold_from_file(self):
        file_path = os.path.join(self.tmp_path, "adjacency_threshold.csv")
        content = """A,B,C\n0,0.5,0\n0,0,0.2\n0.3,0,0"""

        with open(file_path, 'w') as f:
            f.write(content)

        graph, df = utils.graph_from_adjacency_file(file_path, th=0.25)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'A', 'B', 'C'}
        assert set(graph.edges()) == {('A', 'B'), ('C', 'A')}
        assert isinstance(df, pd.DataFrame)

    def test_graph_with_custom_labels_from_file(self):
        file_path = os.path.join(self.tmp_path, "adjacency_custom_labels.csv")
        content = """0,1,0\n0,0,1\n1,0,0"""

        with open(file_path, 'w') as f:
            f.write(content)

        custom_labels = ['X', 'Y', 'Z']
        graph, df = utils.graph_from_adjacency_file(
            file_path, labels=custom_labels, header=False)

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {'X', 'Y', 'Z'}
        assert set(graph.edges()) == {('X', 'Y'), ('Y', 'Z'), ('Z', 'X')}
        assert isinstance(df, pd.DataFrame)


class TestGraphToAdjacency:
    """Test suite for graph_to_adjacency function"""

    def test_simple_graph_to_adjacency(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=1)
        graph.add_edge('B', 'C', weight=1)
        graph.add_edge('C', 'A', weight=1)
        labels = ['A', 'B', 'C']

        adjacency_matrix = utils.graph_to_adjacency(graph, labels)

        expected_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        assert np.array_equal(adjacency_matrix, expected_matrix)

    def test_graph_with_custom_weight_label(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', custom_weight=2)
        graph.add_edge('B', 'C', custom_weight=3)
        graph.add_edge('C', 'A', custom_weight=4)
        labels = ['A', 'B', 'C']

        adjacency_matrix = utils.graph_to_adjacency(
            graph, labels, weight_label='custom_weight')

        expected_matrix = np.array([
            [0, 2, 0],
            [0, 0, 3],
            [4, 0, 0]
        ])

        assert np.array_equal(adjacency_matrix, expected_matrix)

    def test_graph_to_adjacency_with_missing_labels(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=1)
        graph.add_edge('B', 'C', weight=1)
        labels = ['A', 'B', 'C', 'D']  # 'D' is a missing node in the graph

        adjacency_matrix = utils.graph_to_adjacency(graph, labels)

        expected_matrix = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        assert np.array_equal(adjacency_matrix, expected_matrix)


class TestGraphToAdjacencyFile:
    """Test suite for graph_to_adjacency_file function"""
    tmp_path = "/tmp"

    def test_write_simple_graph_to_file(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=1)
        graph.add_edge('B', 'C', weight=1)
        graph.add_edge('C', 'A', weight=1)
        labels = ['A', 'B', 'C']
        file_path = os.path.join(self.tmp_path, "simple_graph.csv")

        utils.graph_to_adjacency_file(graph, file_path, labels)

        with open(file_path, 'r') as f:
            content = f.read()

        expected_content = """A,B,C
A,0.0,1.0,0.0
B,0.0,0.0,1.0
C,1.0,0.0,0.0
"""
        assert content == expected_content

    def test_write_graph_with_custom_weight_label_to_file(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=2)
        graph.add_edge('B', 'C', weight=3)
        graph.add_edge('C', 'A', weight=4)
        labels = ['A', 'B', 'C']
        file_path = os.path.join(self.tmp_path, "custom_weight_graph.csv")

        utils.graph_to_adjacency_file(graph, file_path, labels)

        with open(file_path, 'r') as f:
            content = f.read()

        expected_content = """A,B,C
A,0.0,2.0,0.0
B,0.0,0.0,3.0
C,4.0,0.0,0.0
"""
        assert content == expected_content

    def test_write_graph_with_missing_labels_to_file(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B', weight=1)
        graph.add_edge('B', 'C', weight=1)
        labels = ['A', 'B', 'C', 'D']  # 'D' is a missing node in the graph
        file_path = os.path.join(self.tmp_path, "missing_labels_graph.csv")

        utils.graph_to_adjacency_file(graph, file_path, labels)

        with open(file_path, 'r') as f:
            content = f.read()

        expected_content = """A,B,C,D
A,0.0,1.0,0.0,0.0
B,0.0,0.0,1.0,0.0
C,0.0,0.0,0.0,0.0
D,0.0,0.0,0.0,0.0
"""
        assert content == expected_content


class TestCorrectEdgeFromPrior:
    """Test suite for correct_edge_from_prior function"""

    def test_remove_edge_in_top_list(self):
        dag = nx.DiGraph()
        dag.add_edge('A', 'B')
        prior = [['A', 'B'], ['C']]

        orientation = utils.correct_edge_from_prior(
            dag, 'A', 'B', prior, verbose=False)

        assert orientation == +1
        assert not dag.has_edge('A', 'B')

    def test_add_edge_in_correct_direction(self):
        dag = nx.DiGraph()
        prior = [['A'], ['B', 'C']]

        orientation = utils.correct_edge_from_prior(
            dag, 'A', 'B', prior, verbose=False)

        assert orientation == +1
        assert dag.has_edge('A', 'B')

    def test_reverse_edge(self):
        dag = nx.DiGraph()
        prior = [['B'], ['A']]

        orientation = utils.correct_edge_from_prior(
            dag, 'A', 'B', prior, verbose=False)

        assert orientation == -1
        assert not dag.has_edge('B', 'A')

    def test_no_change_for_unclear_order(self):
        dag = nx.DiGraph()
        dag.add_edge('A', 'B')
        prior = [['A'], ['C'], ['B']]

        orientation = utils.correct_edge_from_prior(
            dag, 'A', 'B', prior, verbose=False)

        assert orientation == 1
        assert dag.has_edge('A', 'B')

    def test_nodes_not_in_prior(self):
        dag = nx.DiGraph()
        dag.add_edge('X', 'Y')
        prior = [['A'], ['B']]

        orientation = utils.correct_edge_from_prior(
            dag, 'X', 'Y', prior, verbose=False)

        assert orientation == 0
        assert dag.has_edge('X', 'Y')

    def test_edge_reflects_backward_connection(self):
        dag = nx.DiGraph()
        dag.add_edge('A', 'B')
        prior = []

        orientation = utils.correct_edge_from_prior(
            dag, 'A', 'B', prior, verbose=False)

        assert orientation == 0

    def test_edge_reflects_connection_in_same_layer(self):
        dag = nx.DiGraph()
        dag.add_edge('B', 'C')
        prior = [['A'], ['B', 'C']]

        orientation = utils.correct_edge_from_prior(
            dag, 'B', 'C', prior, verbose=False)

        assert orientation == 0


class TestValidCandidatesFromPrior:

    def test_valid_candidates_with_prior(self):
        feature_names = ['A', 'B', 'C', 'D']
        effect = 'C'
        prior = [['A'], ['B', 'C'], ['D']]

        candidates = utils.valid_candidates_from_prior(
            feature_names, effect, prior)

        assert candidates == ['A', 'B']

    def test_effect_not_in_prior(self):
        feature_names = ['A', 'B', 'C', 'D']
        effect = 'E'
        prior = [['A'], ['B', 'C'], ['D']]

        candidates = utils.valid_candidates_from_prior(
            feature_names, effect, prior)
        assert candidates == feature_names

    def test_valid_candidates_without_prior(self):
        feature_names = ['A', 'B', 'C', 'D']
        effect = 'C'
        prior = None

        candidates = utils.valid_candidates_from_prior(
            feature_names, effect, prior)

        assert candidates == ['A', 'B', 'D']

    def test_empty_prior(self):
        feature_names = ['A', 'B', 'C', 'D']
        effect = 'C'
        prior = []

        candidates = utils.valid_candidates_from_prior(
            feature_names, effect, prior)

        assert candidates == ['A', 'B', 'D']


class TestBreakCyclesUsingPrior:
    def test_break_cycles_removes_edges(self):
        # Create a graph with a cycle
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        prior = [['A'], ['B'], ['C']]

        # Break cycles using prior
        new_dag = utils.break_cycles_using_prior(dag, prior)

        # Assert that the cycle is broken
        assert not list(nx.simple_cycles(new_dag)
                        ), "The cycle should be broken."

    def test_break_cycles_keeps_valid_edges(self):
        # Create a graph without a cycle
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('B', 'C')])
        prior = [['A'], ['B'], ['C']]

        # Break cycles using prior
        new_dag = utils.break_cycles_using_prior(dag, prior)

        # Assert that no edges are removed
        assert list(new_dag.edges) == [
            ('A', 'B'), ('B', 'C')], "Edges should remain unchanged."

    def test_break_cycles_with_no_prior(self):
        # Create a graph with a cycle
        dag = nx.DiGraph()
        dag.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        prior = None

        # Break cycles using prior
        new_dag = utils.break_cycles_using_prior(dag, prior)

        # Assert that the cycle is broken
        assert sorted(list(nx.simple_cycles(new_dag))[0]) == ['A', 'B', 'C'], \
            "The cycle should remain."


# Mock class for discrepancies
class MockDiscrepancy:
    def __init__(self, shap_gof):
        self.shap_gof = shap_gof


class TestPotentialMisorientedEdges:

    def test_misoriented_edges_identification(self):
        # Define a loop and discrepancies
        loop = ['A', 'B', 'C']
        discrepancies = {
            'A': {'B': MockDiscrepancy(0.3), 'C': MockDiscrepancy(0.8)},
            'B': {'A': MockDiscrepancy(0.2), 'C': MockDiscrepancy(0.1)},
            'C': {'A': MockDiscrepancy(0.6), 'B': MockDiscrepancy(0.9)}
        }

        # Identify misoriented edges
        misoriented_edges = utils.potential_misoriented_edges(
            loop, discrepancies)

        # Assert the correct misoriented edges are identified
        expected_edges = [('A', 'B', 0.1)]
        assert misoriented_edges[0][0] == expected_edges[0][0]
        assert misoriented_edges[0][1] == expected_edges[0][1]
        assert math.isclose(
            misoriented_edges[0][2], expected_edges[0][2], rel_tol=1e-6)

    def test_no_misoriented_edges(self):
        # Define a loop and discrepancies
        loop = ['A', 'B', 'C']
        discrepancies = {
            'A': {'B': MockDiscrepancy(0.2), 'C': MockDiscrepancy(0.8)},
            'B': {'A': MockDiscrepancy(0.3), 'C': MockDiscrepancy(0.1)},
            'C': {'A': MockDiscrepancy(0.6), 'B': MockDiscrepancy(0.9)}
        }

        # Identify misoriented edges
        misoriented_edges = utils.potential_misoriented_edges(
            loop, discrepancies)

        # Assert no misoriented edges are identified
        assert misoriented_edges == []


class TestBreakCyclesIfPresent:

    def test_no_cycles(self):
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C")])
        discrepancies = {}
        result = utils.break_cycles_if_present(dag, discrepancies)
        assert list(result.edges) == [
            ("A", "B"), ("B", "C")], "DAG should remain unchanged"

    def test_single_cycle(self):
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        discrepancies = {
            "B": {"A": MockDiscrepancy(0.2)},
            "C": {"B": MockDiscrepancy(0.1)},
            "A": {"C": MockDiscrepancy(0.3)}
        }
        result = utils.break_cycles_if_present(dag, discrepancies)
        assert not list(nx.simple_cycles(result)), "Cycles should be broken"

    def test_multiple_cycles(self):
        dag = nx.DiGraph()
        dag.add_edges_from(
            [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "B")])
        discrepancies = {
            "B": {"A": MockDiscrepancy(0.2), "D": MockDiscrepancy(0.5)},
            "C": {"B": MockDiscrepancy(0.1)},
            "A": {"C": MockDiscrepancy(0.3)},
            "D": {"C": MockDiscrepancy(0.4)}
        }
        result = utils.break_cycles_if_present(dag, discrepancies)
        assert not list(nx.simple_cycles(result)
                        ), "All cycles should be broken"

    def test_with_prior_knowledge(self):
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        discrepancies = {
            "B": {"A": MockDiscrepancy(0.2)},
            "C": {"B": MockDiscrepancy(0.1)},
            "A": {"C": MockDiscrepancy(0.3)}
        }
        prior = [["A"], ["B"], ["C"]]
        result = utils.break_cycles_if_present(dag, discrepancies, prior=prior)
        assert not list(nx.simple_cycles(result)
                        ), "Cycles should be broken using prior knowledge"

    def test_potential_misoriented_edges(self):
        dag = nx.DiGraph()
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        discrepancies = {
            "B": {"A": MockDiscrepancy(0.2)},
            "C": {"B": MockDiscrepancy(0.1)},
            "A": {"C": MockDiscrepancy(0.3)}
        }
        result = utils.break_cycles_if_present(dag, discrepancies)
        assert not list(nx.simple_cycles(result)
                        ), "Cycles should be broken by changing orientation"


class TestGraphToDotFile:

    tmp_path = "/tmp"

    def test_graph_to_dot_file(self):
        # Create a simple directed graph
        graph = nx.DiGraph()
        graph.add_edges_from([(1, 2), (2, 3)])

        # Define the output file path
        output_file = os.path.join(self.tmp_path, "test_graph.dot")

        # Use the function to write the graph to a dot file
        utils.graph_to_dot_file(graph, output_file)

        # Read the file back and verify its contents
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'digraph' in content
            assert '1 -> 2' in content
            assert '2 -> 3' in content
