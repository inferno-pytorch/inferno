import unittest
from functools import reduce
import torch


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

    # @unittest.skip
    def test_graph_dummy_basic(self):
        import torch
        from inferno.extensions.containers.graph import Graph

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

        input_0 = torch.rand(10, 10)
        input_1 = torch.rand(10, 10)
        model(input_0, input_1)
        self.assertTrue(history == ['conv0_0', 'conv0_1', 'conv1', 'conv2'] or
                        history == ['conv0_1', 'conv0_0', 'conv1', 'conv2'])

    # @unittest.skip
    def test_graph_dummy_inception(self):
        import torch
        from inferno.extensions.containers.graph import Graph

        if not hasattr(self, 'DummyNamedModule'):
            self.setUp()

        DummyNamedModule = self.DummyNamedModule

        history = []
        # Build graph
        model = Graph()
        model.add_input_node('input_0')
        model.add_node('conv0', DummyNamedModule('conv0', history), 'input_0')
        model.add_node('conv1_0', DummyNamedModule('conv1_0', history), 'conv0')
        model.add_node('conv1_1', DummyNamedModule('conv1_1', history), 'conv0')
        model.add_node('conv2', DummyNamedModule('conv2', history, 2),
                       ['conv1_0', 'conv1_1'])
        model.add_output_node('output_0', 'conv2')
        input_0 = torch.rand(10, 10)
        model(input_0)
        self.assertTrue(history == ['conv0', 'conv1_0', 'conv1_1', 'conv2'] or
                        history == ['conv0', 'conv1_1', 'conv1_2', 'conv2'])

    # @unittest.skip
    def test_graph_basic(self):
        from inferno.extensions.containers.graph import Graph
        from inferno.extensions.layers.convolutional import ConvELU2D
        from inferno.utils.model_utils import ModelTester
        # Build graph
        model = Graph()
        model.add_input_node('input_0')
        model.add_node('conv0', ConvELU2D(1, 10, 3), previous='input_0')
        model.add_node('conv1', ConvELU2D(10, 1, 3), previous='conv0')
        model.add_output_node('output_0', previous='conv1')
        ModelTester((1, 1, 100, 100), (1, 1, 100, 100))(model)

    @unittest.skipUnless(torch.cuda.is_available(), "No cuda.")
    def test_graph_device_transfers(self):
        from inferno.extensions.containers.graph import Graph
        from inferno.extensions.layers.convolutional import ConvELU2D
        import torch
        # Build graph
        model = Graph()
        model.add_input_node('input_0')
        model.add_node('conv0', ConvELU2D(1, 10, 3), previous='input_0')
        model.add_node('conv1', ConvELU2D(10, 1, 3), previous='conv0')
        model.add_output_node('output_0', previous='conv1')
        # Transfer
        model.to_device('conv0', 'cpu').to_device('conv1', 'cuda', 0)
        x = torch.rand(1, 1, 100, 100)
        y = model(x)
        self.assertIsInstance(y.data, torch.cuda.FloatTensor)

    @unittest.skip("Needs machine with 4 GPUs")
    def test_multi_gpu(self):
        import torch
        import torch.nn as nn
        from torch.nn.parallel.data_parallel import data_parallel
        from inferno.extensions.containers.graph import Graph

        input_shape = [8, 1, 3, 128, 128]
        model = Graph() \
            .add_input_node('input') \
            .add_node('conv0', nn.Conv3d(1, 10, 3, padding=1), previous='input') \
            .add_node('conv1', nn.Conv3d(10, 1, 3, padding=1), previous='conv0') \
            .add_output_node('output', previous='conv1')

        model.cuda()
        input = torch.rand(*input_shape).cuda()
        data_parallel(model, input, device_ids=[0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
