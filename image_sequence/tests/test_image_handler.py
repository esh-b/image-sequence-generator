import unittest
from unittest.mock import patch

import numpy as np

from image_sequence.image_handler import ImageHandler


class TestImageHandler(unittest.TestCase):

    def test_get_digits_images_no_transform(self):
        """Test to confirm that digit images were not transformed
           given the `transform` config
        """
        image_config = {'transform': False}
        image_handler = ImageHandler(image_config)

        # Arguments
        digits = range(10)
        grouped_labels = (np.random.choice(range(1, 10000), (10, 200), replace=False)).tolist()
        images = np.random.uniform(size=(10000, 28, 28))

        with patch.object(ImageHandler, '_transform_images') as transform_method_mock:
            digits_images = image_handler.get_digits_images(digits, grouped_labels, images)

            # Assert the transform method was not called
            self.assertFalse(transform_method_mock.called)

            # Assert the output image shape is as expected
            self.assertEqual(digits_images.shape, (len(digits), 28, 28))

            # Assert every image in the `digits_image` array comes from their
            # respective labels. In simple terms, assert whether every digit image
            # is taken from the set containing the corresponding `digit` as label
            image_exists = [digits_images[i] in np.take(images, grouped_labels[i], axis=0) for i in range(len(digits))]
            self.assertTrue(np.all(image_exists))

    def test_get_digits_images_with_transform(self):
        """Test whether the images are transformed before return
           given the `transform` config
        """
        image_config = {'transform': True}

        # Arguments
        digits = range(10)
        grouped_labels = (np.random.choice(range(1, 10000), (10, 200), replace=False)).tolist()
        images = np.random.uniform(size=(10000, 28, 28))

        with patch.object(ImageHandler, '_transform_images') as transform_method_mock:
            image_handler = ImageHandler(image_config)

            # Mock return value from pipeline
            transform_method_mock.return_value = np.empty((len(digits), 28, 28))

            # Assert the mocked method was called
            self.assertFalse(transform_method_mock.called)
            digits_images = image_handler.get_digits_images(digits, grouped_labels, images)
            self.assertTrue(transform_method_mock.called)

            # Assert the output image shape is as expected
            self.assertEqual(digits_images.shape, (len(digits), 28, 28))

    def test_resize_image(self):
        """Test image resize method
        """
        image = np.empty((10, 10), dtype=np.float32)
        target_width, target_height = 20, 20

        image_config = {'transform': False}
        image_handler = ImageHandler(image_config)
        resized_image = image_handler.resize_image(image, 20, 20)

        self.assertEqual(resized_image.shape, (20, 20))
