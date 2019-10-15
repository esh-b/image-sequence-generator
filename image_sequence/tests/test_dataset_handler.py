import os
import json
import gzip
import tempfile
import unittest
import pkg_resources
from io import BytesIO
from unittest.mock import patch

import numpy as np

from image_sequence.dataset_handler import DatasetHandler


class TestDatasetHandler(unittest.TestCase):

    @patch('logging.Logger.info')
    @patch.object(DatasetHandler, '_dataset_exists', return_value=True)
    def test_class_initialization_without_download(self, mock_exists_method, mock_logger_info):
        """Test class initialization when dataset already exists
        """
        dataset_handler = DatasetHandler()
        self.assertTrue(mock_logger_info.called)

    @patch('logging.Logger.info')
    @patch('image_sequence.dataset_handler.urlopen')
    @patch.object(pkg_resources, 'resource_filename')
    @patch.object(DatasetHandler, '_dataset_exists', return_value=False)
    def test_class_initialization_with_download(self, mock_exists_method, mock_pkg_method,
                                                mock_urlopen, mock_logger_info):
        """Test class initialization by downloading dataset
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # set dataset path (in DatasetHandler when initialized) to the temporary directory
            mock_pkg_method.return_value = tmpdir

            # Write send input_fileobj as alternative to response from urlopen
            input_fileobj = BytesIO(gzip.compress(b'downloaded_content'))
            mock_urlopen().__enter__.return_value = input_fileobj

            # Assert directory was empty before download
            self.assertFalse(os.listdir(tmpdir))

            dataset_handler = DatasetHandler()

            self.assertTrue(mock_logger_info.called)
            # Assert 4 new files were created (2 gz files and 2 unzipped files)
            self.assertEqual(len(os.listdir(tmpdir)), 4)

    @patch.object(DatasetHandler, "_load_images")
    @patch.object(DatasetHandler, "_load_labels")
    @patch.object(DatasetHandler, '_dataset_exists', return_value=True)
    def test_load_dataset(self, mock_exists_method, mock_load_labels, mock_load_images):
        """Test load_dataset method
        """
        # Assert images and labels are returned and have same shape
        mock_load_images.return_value = np.random.uniform(size=(10000, 28, 28))
        mock_load_labels.return_value = np.random.randint(0, 10, (10000,))
        images, labels = DatasetHandler().load_dataset()
        self.assertEqual(images.shape[0], labels.shape[0])

        # Assert images or labels is None
        mock_load_images.return_value = np.random.uniform(size=(10000, 28, 28))
        mock_load_labels.return_value = None
        self.assertRaises(RuntimeError, DatasetHandler().load_dataset)

        # Assert different values for images and labels
        mock_load_images.return_value = np.random.uniform(size=(10000, 28, 28))
        mock_load_labels.return_value = np.random.randint(0, 10, (5000,))
        self.assertRaises(ValueError, DatasetHandler().load_dataset)

    @patch.object(pkg_resources, 'resource_filename')
    @patch.object(DatasetHandler, '_dataset_exists', return_value=True)
    def test_load_grouped_labels(self, mock_exists_method, mock_pkg_method):
        """Test (i) saving created grouped labels in file
            (ii) loading the saved data structure from file
        """
        unique_labels = range(10)
        labels = np.random.choice(unique_labels, 10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_pkg_method.return_value = tmpdir

            # Assert no file in directory
            self.assertFalse(os.listdir(tmpdir))

            grouped_labels = DatasetHandler().load_grouped_labels(labels)

            # # Assert length and type of the returned array
            self.assertEqual(len(grouped_labels), len(unique_labels))
            self.assertIsInstance(grouped_labels, list)
            # # Assert number of indices in the `grouped_labels`
            # # (aggregated from all sublists) is equal to total labels (i.e. 10000)
            self.assertEqual(sum([len(indices_list) for indices_list in grouped_labels]), 10000)

            # (i) FILE SAVE ASSERTION
            files = os.listdir(tmpdir)
            self.assertTrue(files)
            self.assertEqual(len(files), 1)

            # (ii) FILE LOAD ASSERTION
            # Assert the loaded data structure from file is same as the saved one
            with open(os.path.join(tmpdir, files[0]), 'r') as fp:
                loaded_grouped_labels = json.load(fp)
            self.assertTrue(np.array_equal(grouped_labels, loaded_grouped_labels))
