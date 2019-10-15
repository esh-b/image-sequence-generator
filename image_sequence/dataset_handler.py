import os
import gzip
import json
import shutil
import logging
import struct as st
import pkg_resources
from collections import Iterable
from urllib.request import urlopen

import numpy as np


class DatasetHandler:

    """Handle dataset related tasks like downloading
       dataset or loading downloaded dataset
    
    Attributes
    ----------
    DATA_FILES : list
        Files to download (incase dataset does not exist)
    dataset_path : str
        Path to the dataset dir
    grouped_labels_path : str
        Path to the serialized `grouped_labels`
    images_path : str
        Path to the unzipped ubyte image file
    labels_path : str
        Path to the unzipped ubyte label file
    logger : Logger
        Logger
    """
    
    DATA_FILES = [
        'train-images-idx3-ubyte.gz', 
        'train-labels-idx1-ubyte.gz'
        # 't10k-images-idx3-ubyte.gz',
        # 't10k-labels-idx1-ubyte.gz'
    ]

    def __init__(self):
        self.dataset_path = pkg_resources.resource_filename(__name__, "dataset/")
        self.logger = logging.getLogger(__name__)

        # Filename where the `grouped_labels` list is stored. 
        # The `grouped_labels` list for a given dataset is stored onto 
        # filesystem since `grouped_labels` calculation is an expensive task.
        grouped_labels_fname = "grouped_labels_{}.json".format(self._basename(self.DATA_FILES[1]))
        self.images_path = os.path.join(self.dataset_path, 
                                        self._basename(self.DATA_FILES[0]))
        self.labels_path = os.path.join(self.dataset_path, 
                                        self._basename(self.DATA_FILES[1]))
        self._grouped_labels_path = os.path.join(self.dataset_path, grouped_labels_fname)

        if not self._dataset_exists():
            self.logger.info("MNIST dataset not found...\nDownloading...")
            self._download_mnist()
        else:
            self.logger.info("MNIST dataset exists. Skipping download...")

    def _basename(self, path):
        return os.path.splitext(path)[0]

    def _dataset_exists(self):
        """Check whether the dataset and related files exist
        
        Returns
        -------
        bool
            A bool indicating whether the dataset already exists
        """
        return (os.path.exists(self.dataset_path) and
                os.path.isfile(self.images_path) and
                os.path.isfile(self.labels_path))

    def _download_file(self, url: str, dest_path: str):
        """Download the file given url and save it in the destination path
        
        Parameters
        ----------
        url : str
            File url to download
        dest_path : str
            Destination path - to store the downloaded file
        
        Raises
        ------
        Exception
            Raised when there is an issue with downloading file
            Possible errors can be:
                urllib.request.URLError
                urllib.request.HTTPError
                httplib.HTTPException
                socket.error
        """
        try:
            with urlopen(url) as resp, open(dest_path, 'wb') as out_fp:
                shutil.copyfileobj(resp, out_fp)
        except:
            raise Exception("Error occurred while downloading file: {}".format(dest_path))

    def _unzip_file(self, src_path: str):
        """Method to unzip a file at a given path.
        
        Parameters
        ----------
        src_path : str
            Filepath to the zipped file(gz) source
        
        Raises
        ------
        Exception
            Raised when there is some issue with unzipping
        """
        try:
            # Destination filepath (without gz extension)
            dest_path = self._basename(src_path)
            with gzip.open(src_path, 'rb') as zipped_fp:
                with open(dest_path, 'wb') as unzipped_fp:
                    unzipped_fp.write(zipped_fp.read())
        except:
            raise Exception("Error while unzipping file at path: {}".format(src_path))

    def _download_mnist(self):
        """Download MNIST dataset
        """
        # Remove `grouped_labels.json` file if it already exists
        if os.path.exists(self._grouped_labels_path):
            os.remove(self._grouped_labels_path)

        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        for fname in self.DATA_FILES:
            self.logger.info("Downloading file: {}".format(fname))
            url = 'http://yann.lecun.com/exdb/mnist/{}'.format(fname)
            dest_path = os.path.join(self.dataset_path, fname)

            self._download_file(url, dest_path)
            self._unzip_file(dest_path)
            self.logger.info("Downloaded file {} successfully".format(fname))
        self.logger.info("Dataset downloaded successfully")

    def _load_images(self):
        """Load images from ubyte file
        
        Returns
        -------
        TYPE
            numpy array of shape (num_images, image_height, image_width)
        
        Raises
        ------
        ValueError
            Raised when the magic number did not match
        """
        with open(self.images_path, 'rb') as fp:
            magic = st.unpack('>I', fp.read(4))[0]
            num_images = st.unpack('>I', fp.read(4))[0]
            num_rows = st.unpack('>I', fp.read(4))[0]
            num_cols = st.unpack('>I', fp.read(4))[0]

            if magic != 2051:
                raise ValueError("Magic number for `images` dataset does not match")

            nBytesTotal = num_images * num_rows * num_cols
            images_content = st.unpack('>' + 'B' * nBytesTotal, fp.read(nBytesTotal))
            images = ((255 - np.asarray(images_content, dtype=np.float32)) / 255)
            return images.reshape((num_images, num_rows, num_cols))

    def _load_labels(self):
        """Load labels from ubyte file
        
        Returns
        -------
        np.ndarray
            numpy array of shape(num_labels,)
        
        Raises
        ------
        ValueError
            Raised when the magic number did not match
        """
        with open(self.labels_path, 'rb') as fp:
            magic = st.unpack('>I', fp.read(4))[0]
            if magic != 2049:
                raise ValueError("Magic number for `labels` dataset does not match")

            num_labels = st.unpack('>I', fp.read(4))[0]
            labels_content = st.unpack('>' + 'B' * num_labels, fp.read(num_labels))
            return np.asarray(labels_content)

    def load_dataset(self):
        """Load dataset from filesystem
        
        Returns
        -------
        tuple
            Returns tuple containing images array with shape 
            (num_images, image_height, image_width) and 
            labels arrays with shape (num_labels,)
        
        Raises
        ------
        RuntimeError
            Raised when there was some issue loading dataset from file
        ValueError
            Raised when the `images` and `labels` sizes do not match
        """
        images, labels = self._load_images(), self._load_labels()

        if images is None or labels is None:
            raise RuntimeError("Could not load dataset from file")
        elif not images.shape[0] == labels.shape[0]:
            raise ValueError("Image and Label array sizes do not match")
        return images, labels

    def load_grouped_labels(self, labels: np.ndarray):
        """Load the serialized `grouped_labels` list if it exists.
           Else create the `grouped_labels` list and serialize  it.
           This serialization is useful in that we need not create
           a new `grouped_labels` array everytime the API is called.
        
        Parameters
        ----------
        labels : np.ndarray
            Loaded labels dataset array
        
        Returns
        -------
        list
            A list of lists where every sublist at index i
            contains indices of images having label i
            Example: 
                grouped_labels[0] is a list containing the
                indices of all images having label 0
        """
        if os.path.isfile(self._grouped_labels_path):
            # If there was any issue with reading JSON, 
            # warn and create new serialized JSON for
            # `grouped_labels` list
            try:
                with open(self._grouped_labels_path, "r") as fp:
                    grouped_labels = json.load(fp)
            except json.JSONDecodeError:
                self.logger.error("The `grouped_labels` json file has been tampered. "
                                "Creating new serialization...")
            else:
                return grouped_labels

        # Create sublist for as many unique labels
        grouped_labels = [[] for i in range(len(set(labels)))]
        for idx, label in enumerate(labels):
            grouped_labels[label].append(idx)

        with open(self._grouped_labels_path, 'w') as fp:
            json.dump(grouped_labels, fp)
        return grouped_labels
