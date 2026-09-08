import unittest
import torch
from core.model_loader import TransformerNet, load_style_model

class TestModelLoader(unittest.TestCase):
    def test_transformer_forward_pass(self):
        model = TransformerNet()
        dummy_input = torch.randn(1, 3, 128, 128)
        output = model(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_load_candy_model(self):
        model = load_style_model("Candy", device="cpu")
        self.assertIsInstance(model, TransformerNet)

if __name__ == '__main__':
    unittest.main()
