import torch.nn as nn
import networkx as nx
from networkx.algorithms.dag import is_directed_acyclic_graph, topological_sort
from collections import OrderedDict

from ...utils import python_utils as pyu


class NNGraph(nx.DiGraph):
    """A NetworkX DiGraph, except that node and edge ordering matters."""
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


class Identity(nn.Module):
    def forward(self, input):
        return input


class Graph(nn.Module):
    def __init__(self):
        super(Graph, self).__init__()
        self._graph = NNGraph()

    def is_node_in_graph(self, name):
        return name in self._graph.node

    def is_source_node(self, name):
        assert self.is_node_in_graph(name)
        return self._graph.in_degree(name) == 0

    def is_sink_node(self, name):
        assert self.is_node_in_graph(name)
        return self._graph.out_degree(name) == 0

    @property
    def output_nodes(self):
        return [name for name, node_attributes in self._graph.node.items()
                if node_attributes.get('is_output_node', False)]

    @property
    def input_nodes(self):
        return [name for name, node_attributes in self._graph.node.items()
                if node_attributes.get('is_input_node', False)]

    @property
    def graph_is_valid(self):
        # Check if the graph is a DAG
        is_dag = is_directed_acyclic_graph(self._graph)
        # Check if output nodes are sinks
        output_nodes_are_sinks = all([self.is_sink_node(name) for name in self.output_nodes])
        # Check inf input nodes are sources
        input_nodes_are_sources = all([self.is_source_node(name) for name in self.input_nodes])
        # TODO Check whether only input nodes are sources and only output nodes are sinks
        # Conclude
        is_valid = is_dag and output_nodes_are_sinks and input_nodes_are_sources
        return is_valid

    def assert_graph_is_valid(self):
        assert is_directed_acyclic_graph(self._graph), "Graph is not a DAG."
        for name in self.output_nodes:
            assert self.is_sink_node(name), "Output node {} is not a sink.".format(name)
            assert not self.is_source_node(name), "Output node {} is a source node. " \
                                                  "Make sure it's connected.".format(name)
        for name in self.input_nodes:
            assert self.is_source_node(name), "Input node {} is not a source.".format(name)
            assert not self.is_sink_node(name), "Input node {} is a sink node. " \
                                                "Make sure it's connected.".format(name)

    def add_node(self, name, module):
        assert isinstance(module, nn.Module)
        self.add_module(name, module)
        self._graph.add_node(name, module=module)
        return self

    def add_input_node(self, name):
        self._graph.add_node(name, module=Identity(), is_input_node=True)
        return self

    def add_output_node(self, name):
        self._graph.add_node(name, is_output_node=True)
        return self

    def add_edge(self, from_node, to_node):
        assert self.is_node_in_graph(from_node)
        assert self.is_node_in_graph(to_node)
        self._graph.add_edge(from_node, to_node)
        assert self.graph_is_valid
        return self

    def forward_through_node(self, name, input=None):
        # If input is a tuple/list, it will NOT be unpacked.
        # Make sure the node is in the graph
        if input is None:
            # Make sure the node is not a source node
            assert not self.is_source_node(name)
            # Get input from payload
            incoming_edges = self._graph.in_edges(name)
            input = [self._graph[incoming][this]['payload']
                     for incoming, this in incoming_edges]
        else:
            assert self.is_node_in_graph(name)
            # Convert input to list
            input = [input]
        # Get outputs
        outputs = pyu.to_iterable(self._graph.node[name]['module'](*input))
        # Distribute outputs to outgoing payloads if required
        if not self.is_sink_node(name):
            outgoing_edges = self._graph.out_edges(name)
            # Make sure the number of outputs check out
            assert len(outputs) == len(outgoing_edges), \
                "Number of outputs from the model ({}) does not match the number " \
                "of out-edges ({}) in the graph for this node ({}).".format(len(outputs),
                                                                            len(outgoing_edges),
                                                                            name)
            for (this, outgoing), output in zip(outgoing_edges, outputs):
                self._graph[this][outgoing].update({'payload': output})
        # Return outputs
        return pyu.from_iterable(outputs)

    def forward(self, *inputs):
        self.assert_graph_is_valid()
        input_nodes = self.input_nodes
        output_nodes = self.output_nodes
        assert len(inputs) == len(input_nodes), "Was expecting {} " \
                                                "arguments for as many input nodes, got {}."\
            .format(len(input_nodes), len(inputs))
        # Unpack inputs to input nodes
        for input, input_node in zip(inputs, input_nodes):
            self.forward_through_node(input_node, input=input)
        # Toposort the graph
        toposorted = topological_sort(self._graph)
        # Remove all input and output nodes
        toposorted = [name for name in toposorted
                      if name not in input_nodes and name not in output_nodes]
        # Forward
        for node in toposorted:
            self.forward_through_node(node)
        # Read outputs from output nodes
        outputs = []
        for output_node in output_nodes:
            # Get all incoming edges to output node
            outputs_from_node = [self._graph[incoming][this]['payload']
                                 for incoming, this in self._graph.in_edges(output_node)]
            outputs.append(pyu.from_iterable(outputs_from_node))
        return pyu.from_iterable(outputs)
