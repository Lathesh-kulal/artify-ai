import unittest
from PIL import Image
from core.image_processor import resize_image, preprocess_image, postprocess_tensor

class TestImageProcessor(unittest.TestCase):
    def test_resize_image(self):
        img = Image.new('RGB', (1200, 800))
        resized = resize_image(img, max_dim=600)
        self.assertLessEqual(max(resized.size), 600)
        self.assertEqual(resized.size, (600, 400))

    def test_preprocess_and_postprocess(self):
        img = Image.new('RGB', (100, 100), color=(100, 150, 200))
        tensor = preprocess_image(img, max_dim=100, device="cpu")
        self.assertEqual(tensor.shape, (1, 3, 100, 100))

        reconstructed = postprocess_tensor(tensor)
        self.assertEqual(reconstructed.size, (100, 100))

if __name__ == '__main__':
    unittest.main()
