import unittest
import os
from PIL import Image
from core.utils import get_image_bytes, save_image

class TestUtils(unittest.TestCase):
    def test_get_image_bytes(self):
        img = Image.new('RGB', (50, 50), color='blue')
        data = get_image_bytes(img, format='JPEG')
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_save_image(self):
        test_path = "outputs/test_output.jpg"
        img = Image.new('RGB', (50, 50), color='green')
        saved_path = save_image(img, test_path)
        self.assertTrue(os.path.exists(saved_path))
        if os.path.exists(saved_path):
            os.remove(saved_path)

if __name__ == '__main__':
    unittest.main()
