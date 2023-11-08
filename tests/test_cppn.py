import unittest
import sys
sys.path.append('..')
from cppn.cppn import CPPN, CPPNConfig, Node, Connection
from cppn.activation_functions import *
import torch
import json

class TestNode(unittest.TestCase):
    def setUp(self):
        self.activation = IdentityActivation()
        self.node_id = 'n1'
        self.bias = 0.1
        self.node = Node(self.activation, self.node_id, self.bias)

    def test_forward(self):
        input_tensor = torch.tensor([1.0])
        output = self.node(input_tensor)
        self.assertTrue(torch.equal(output, self.activation(input_tensor + self.bias)))

    def test_to_json(self):
        json_repr = self.node.to_json()
        self.assertEqual(json_repr['id'], self.node_id)
        self.assertEqual(json_repr['activation'], self.activation.__class__.__name__)
        self.assertAlmostEqual(json_repr['bias'], self.bias)

    def test_from_json(self):
        json_repr = self.node.to_json()
        new_node = Node(None, None)
        new_node.from_json(json_repr)
        self.assertEqual(new_node.id, self.node_id)
        self.assertIsInstance(new_node.activation, IdentityActivation)
        self.assertAlmostEqual(new_node.bias.item(), self.bias)

class TestConnection(unittest.TestCase):
    def setUp(self):
        self.weight = torch.tensor([0.5])
        self.connection = Connection(self.weight)

    def test_forward(self):
        input_tensor = torch.tensor([2.0])
        output = self.connection(input_tensor)
        self.assertTrue(torch.equal(output, input_tensor * self.weight))

    def test_to_json(self):
        json_repr = self.connection.to_json()
        self.assertEqual(json_repr['weight'], self.weight.item())
        self.assertTrue(json_repr['enabled'])

    def test_from_json(self):
        json_repr = self.connection.to_json()
        new_connection = Connection(None)
        new_connection.from_json(json_repr)
        self.assertEqual(new_connection.weight.item(), self.weight.item())
        self.assertTrue(new_connection.enabled)

class TestCPPN(unittest.TestCase):
    def setUp(self):
        self.config = CPPNConfig()
        self.config.num_inputs = 2
        self.config.num_outputs = 1
        self.config.init_connection_probability = 1.0
        self.cppn = CPPN(self.config).to('cpu')

    def test_initialization(self):
        self.assertEqual(len(self.cppn.nodes), self.cppn.n_input + self.cppn.n_output)
        self.assertEqual(len(self.cppn.connections), 
                         self.config.num_inputs * self.config.num_outputs)

    def test_forward(self):
        input_tensor = torch.randn(1, 1, 2)
        output = self.cppn(input_tensor)
        self.assertEqual(output.shape, (1, 1, 1))

    def test_to_json(self):
        json_repr = self.cppn.to_json()
        self.assertEqual(json_repr['id'], self.cppn.id)
        self.assertIsInstance(json_repr['nodes'], dict)
        self.assertIsInstance(json_repr['connections'], dict)

    def test_from_json(self):
        json_repr = self.cppn.to_json()
        json_repr = json.dumps(json_repr)
        new_cppn = CPPN(self.config, do_init=False)
        new_cppn.from_json(json_repr)
        self.assertEqual(new_cppn.id, self.cppn.id)
        self.assertEqual(len(new_cppn.nodes), len(self.cppn.nodes))
        self.assertEqual(len(new_cppn.connections), len(self.cppn.connections))

        new_cppn = CPPN.create_from_json(json_repr, self.config)
        self.assertEqual(new_cppn.id, self.cppn.id)
        self.assertEqual(len(new_cppn.nodes), len(self.cppn.nodes))
        self.assertEqual(len(new_cppn.connections), len(self.cppn.connections))
        
        

if __name__ == '__main__':
    unittest.main()
