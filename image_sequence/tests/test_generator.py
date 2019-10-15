import os
import json
import tempfile
import unittest
import pkg_resources
from unittest.mock import (
        mock_open, patch
    )

import numpy as np
from PIL import Image

from image_sequence.generator import DigitSequence
from image_sequence.dataset_handler import DatasetHandler


@patch.object(DatasetHandler, '_dataset_exists', return_value=True)
class TestDigitSequenceInitialization(unittest.TestCase):
    """Test initialization of `DigitSequence` class. The above patch 
       helps to initialize class without actually downloading dataset
       i.e. without real side effects.
    """
    def setUp(self):
        self.config = {
            "image": {
                "transform": False
            },
            "spacing": {
                "type": "variable",
                "subtype": "between"
            },
            "output_format": "png"
        }

    def test_invalid_image_config(self, mock_dataset_exists):
        """Test invalid image_config in config
        """
        mock_configs = []
        # Remove image related keys from config and assert exceptions
        self.config['image'].pop('transform', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)
        self.config.pop('image', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)
        
        with patch('image_sequence.generator.open') as m_open:
            m_open.side_effect = mock_configs
            with self.assertRaises(KeyError):
                # Assert KeyError incase `transform` key 
                # in image config was not provided
                generator = DigitSequence()
                # Assert KeyError incase `image` key 
                # was not provided in the config.
                generator = DigitSequence()

    def test_invalid_spacing_config(self, mock_dataset_exists):
        """Test invalid spacing_configs in config
        """
        mock_configs = []
        self.config['spacing'].pop('subtype', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)
        self.config['spacing'].pop('type', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)
        self.config.pop('spacing', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)

        with patch('image_sequence.generator.open') as m_open:
            m_open.side_effect = mock_configs
            with self.assertRaises(KeyError):
                # Assert KeyError incase spacing `subtype` key 
                # was not provided in the config
                generator = DigitSequence()
                # Assert KeyError incase spacing `type` key 
                # was not provided in the config
                generator = DigitSequence()
                # Assert KeyError incase main `spacing` key 
                # was not provided in the config
                generator = DigitSequence()

    def test_invalid_output_format_config(self, mock_dataset_exists):
        """Test giving no `output_format` key in config
        """
        mock_configs = []
        self.config['output_format'] = "invalid_option"
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)
        self.config.pop('output_format', None)
        mock_configs.append(mock_open(read_data=json.dumps(self.config)).return_value)

        with patch('image_sequence.generator.open') as m_open:
            m_open.side_effect = mock_configs
            # Assert ValueError incase `output_format` value was invalid
            with self.assertRaises(ValueError):
                generator = DigitSequence()

            # Assert KeyError if `output_format` is not specified
            with self.assertRaises(KeyError):
                generator = DigitSequence()


class TestDigitSequenceGeneration(unittest.TestCase):
    """Test sequence generation task and image saving task
    """
    def setUp(self):
        self.digits, self.spacing_range, self.image_width = range(10), (0, 100), 390
        self.generator, self.gen_image = self.load_generator_and_get_sequence()

    @patch.object(DatasetHandler, '_dataset_exists', return_value=True)
    @patch.object(DatasetHandler, "_load_images")
    @patch.object(DatasetHandler, "_load_labels")
    def load_generator_and_get_sequence(self, mock_load_labels, mock_load_images, mock_dataset_exists):
        """Many patches are applied to negate the actual database downloading, 
           database loading from filesystem side effects and others like loading/saving 
           `grouped_labels` from filesystem

        Returns:
            generator and the generated image array
        """
        mock_load_images.return_value = np.random.uniform(size=(10000, 28, 28))
        mock_load_labels.return_value = np.random.randint(0, 10, (10000,))

        config_path = pkg_resources.resource_filename('image_sequence', "config.json")

        # resource_filename is called once in the generator (for getting config_path)
        # and other time in dataset_handler (for dataset_path)
        # We mock the method and return the default config.json path the first time
        # (in generator) and return a tmpdir to save grouped_labels.
        # The tmpdir is used so as to not create any side effects in the real
        # dataset directory
        with patch.object(pkg_resources, 'resource_filename') as mock_pkg_method:
            with tempfile.TemporaryDirectory() as tmpdir:
                mock_pkg_method.side_effect = [config_path, tmpdir]

                generator = DigitSequence()
                gen_image = generator.generate_numbers_sequence(
                                self.digits, self.spacing_range, self.image_width
                            )
                return generator, gen_image

    def test_generate_numbers_sequence(self):
        """Check any exception thrown for class initialization using
           valid config and then test the `generate_number_sequence` method
        """
        # assert constraints for the generated image array for
        # the `variable` and `between` config (which is default config)
        # This spacing variant is asked in the question
        self.assertEqual(self.gen_image.shape, (28, 390))
        self.assertEqual(self.gen_image.dtype, np.float32)
        self.assertTrue(np.amin(self.gen_image) >= 0.0)
        self.assertTrue(np.amax(self.gen_image) <= 1.0)

        with self.assertRaises(TypeError) as err:
            # non iterable as digits argument
            self.generator.generate_numbers_sequence(1, (0, 100), 390)
            # non integer in digits argument
            self.generator.generate_numbers_sequence(["hello"], (0, 100), 390)
            # non-tuple type for `spacing_range`
            self.generator.generate_numbers_sequence(range(10), "0,100", 390)
            # non-integer values in tuple
            self.generator.generate_numbers_sequence(range(10), ("0", "100"), 390)
            # `image_width` is non integer
            self.generator.generate_numbers_sequence(range(10), (0, 100), 390.2)

        with self.assertRaises(ValueError):
            # no items in digits argument
            self.generator.generate_numbers_sequence([], (0, 100), 390.2)

    def test_save_image(self):
        # Temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_filepath = self.generator.save_image(self.gen_image, self.digits, tmpdir)
            # Assert file exists in filesystem
            self.assertTrue(os.path.isfile(saved_filepath))

            # Load the saved image and check whether the two arrays are close enough
            # They will not be exactly equal since the image is converted to uint8 before saving
            loaded_image = np.array(Image.open(saved_filepath), dtype=np.float32) / 255
            self.assertTrue(np.allclose(self.gen_image, loaded_image, atol=1e-02))
