import unittest
from functools import reduce


class TestGraph(unittest.TestCase):
    def setUp(self):
        import torch.nn as nn
        from inferno.utils.python_utils import from_iterable

        class DummyNamedModule(nn.Module):
            def __init__(self, name, history, num_inputs=1):
                super(DummyNamedModule, self).__init__()
                self.name = name
                self.history = history
                self.num_inputs = num_inputs

            def forward(self, *inputs):
                assert len(inputs) == self.num_inputs
                self.history.append(self.name)
                if self.num_inputs > 1:
                    output = reduce(lambda x, y: x + y, inputs)
                else:
                    output = from_iterable(inputs)

                return output

        self.DummyNamedModule = DummyNamedModule

    def test_graph_basic(self):
        import torch
        from torch.autograd import Variable
        from inferno.extensions.layers.graph import Graph

        if not hasattr(self, 'DummyNamedModule'):
            self.setUp()

        DummyNamedModule = self.DummyNamedModule

        history = []
        # Build graph
        model = Graph()
        model.add_input_node('input_0')
        model.add_input_node('input_1')
        model.add_node('conv0_0', DummyNamedModule('conv0_0', history))
        model.add_node('conv0_1', DummyNamedModule('conv0_1', history))
        model.add_node('conv1', DummyNamedModule('conv1', history, 2))
        model.add_node('conv2', DummyNamedModule('conv2', history))
        model.add_output_node('output_0')
        model.add_edge('input_0', 'conv0_0')\
            .add_edge('input_1', 'conv0_1')\
            .add_edge('conv0_0', 'conv1')\
            .add_edge('conv0_1', 'conv1')\
            .add_edge('conv1', 'conv2')\
            .add_edge('conv2', 'output_0')

        input_0 = Variable(torch.rand(10, 10))
        input_1 = Variable(torch.rand(10, 10))
        output = model(input_0, input_1)
        self.assertTrue(history == ['conv0_0', 'conv0_1', 'conv1', 'conv2'] or
                        history == ['conv0_1', 'conv0_0', 'conv1', 'conv2'])


if __name__ == '__main__':
    TestGraph().test_graph_basic()