"""Sequence generator
"""
import os
import json
import logging
import pkg_resources
from collections import Iterable

import numpy as np
from PIL import Image

from .dataset_handler import DatasetHandler
from .image_handler import ImageHandler
from .spacing_handler import SpacingHandler


class DigitSequence:

    """MNIST Sequence generator
    
    Attributes
    ----------
    config : dict
        The loaded config
    dataset_handler : DatasetHandler
        Object to handle dataset related tasks
    default_height : int
        Default height of the output image (per image)
    default_width : int
        Default width of the output image (per image)
    grouped_labels : list
        A list of lists where every sublist at index i
        contains indices of images having label i
        Example: 
            grouped_labels[0] is a list containing the
            indices of all images having label 0
    img_handler : ImageHandler
        Object to handle digit image extraction, transformation 
        and related tasks
    spacing_handler : SpacingHandler
        Object to handle spacing generation and related tasks
    """
    
    def __init__(self, config_path=None):
        self.default_width = 28
        self.default_height = 28

        # Logging setup
        logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Use the default config.json file incase external config was not provided
        config_path = config_path or pkg_resources.resource_filename(__name__, "config.json")

        # Load and validate the config file
        self.config = self._load_config(config_path)
        self._validate_config(self.config)

        # Load the dataset
        self._mnist = DatasetHandler()
        self.images, self.labels = self._mnist.load_dataset()
        self._grouped_labels = self._mnist.load_grouped_labels(self.labels)

        # Digit generation handler and gap generation handler (gaps in between digits)
        self._img_handler = ImageHandler(self.config['image'])
        self._spacing_handler = SpacingHandler(self.config['spacing'], self.default_width)

    def _load_config(self, config_path: str):
        """Load config from the config path
        
        Parameters
        ----------
        config_path : str
            The given filepath to config file
        
        Returns
        -------
        dict
            Return config (as dict) if it was read successfully
        
        Raises
        ------
        FileNotFoundError
            Raised when the config file was not found in the specified directory
        json.JSONDecodeError
            Raised when the JSON decoding was not successful
        PermissionError
            Raised when there is no permission to read the config file
        """
        config = {}
        if not os.path.isfile(config_path):
            raise FileNotFoundError("The config file at '{}' was not found".format(config_path))
        elif not os.access(config_path, os.R_OK):
            raise PermissionError("Permission denied to read config file")

        try:
            with open(config_path) as fp:
                config = json.loads(fp.read())
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Could not read JSON from file")
        return config

    def _validate_config(self, config: dict):
        """Method to validate the values in the loaded config
        
        Parameters
        ----------
        config : dict
            A dictionary of config values
        
        Raises
        ------
        KeyError
            Raised when the required key is not present in the config file
        ValueError
            Raised when the given value for any key is not supported
        """
        if "image" not in config or "transform" not in config["image"]:
            raise KeyError("Image options is not defined in config file")
        if "spacing" not in config or \
                not all(key in config["spacing"] for key in ["type", "subtype"]):
            raise KeyError("Spacing type or subtype is not defined in config file")
        elif "output_format" not in config:
            raise KeyError("Output image format needs to be specified")
        elif not config["output_format"].lower() in ["png", "jpg", "jpeg"]:
            raise ValueError("Given output format is not supported")

    def _validate_inputs(self, digits: Iterable, spacing_range: tuple, image_width: int):
        """Method to validate the given user input
        
        Parameters
        ----------
        digits : Iterable
            A list-like Iterable containing numerical values of
            the digits in the sequence.
        spacing_range : tuple
            A (minimum, maximum) pair (tuple), representing the min and max spacing
            between digits.
        image_width : int
            the width of the output image (in pixels)
        
        Raises
        ------
        TypeError
            Raised when the given argument does not match the required type
        ValueError
            Raised when the given argument does not contain the required value
        """
        if not isinstance(digits, Iterable):
            raise TypeError("Digits argument must be an Iterable")
        elif not digits:
            raise ValueError("Digits argument must contain atleast one digit")
        elif not all(isinstance(x, int) for x in digits):
            raise TypeError("Digits arguments must have only integers")
        elif not isinstance(spacing_range, tuple):
            raise TypeError("`spacing_range` argument must be a tuple")
        elif not all(isinstance(x, int) for x in spacing_range):
            raise TypeError("`spacing_range` tuple must have only integers")
        elif not isinstance(image_width, int):
            raise TypeError("`image_width` argument must be an integer")

    def generate_numbers_sequence(self, digits: list, spacing_range: tuple, image_width: int):
        """
        Generate an image that contains the sequence of given numbers, spaced
        randomly using an uniform distribution.
        
        Parameters
        ----------
        digits : list
            A list-like containing the numerical values of the digits from which
            the sequence will be generated (for example [3, 5, 0]).
        spacing_range : tuple
            a (minimum, maximum) pair (tuple), representing the min and max spacing
            between digits. Unit should be pixel.
        image_width : int
            the width of the output image (in pixels)
        
        Returns
        -------
        The image containing the sequence of numbers. Images should be represented
        as floating point 32bits numpy arrays with a scale ranging from 0 (black) to
        1 (white), the first dimension corresponding to the height and the second
        dimension to the width.
        """
        self._validate_inputs(digits, spacing_range, image_width)

        # Get the np array of the digit images
        digits_images = self._img_handler.get_digits_images(digits, self._grouped_labels, self.images)

        # Based on the spacing option, calculate spacing between digits
        # spacing_arr is an array of spacing values for every possible space
        spacing_arr = self._spacing_handler.get_spacing(len(digits), spacing_range, image_width)

        image_out = np.zeros((self.default_height, 0), dtype=np.float32)
        for i in range(len(spacing_arr)):
            curr_space_width = np.ones([self.default_height, spacing_arr[i]], dtype=np.float32)
            image_out = np.concatenate((image_out, curr_space_width), axis=1)
            if i < len(digits_images):
                image_out = np.concatenate((image_out, digits_images[i]), axis=1)

        # Resize image if the width is not equal to image_width.
        # This is necessary for `variable` spacing type
        if self._spacing_handler.type == SpacingHandler.Types.VARIABLE:
            return self._img_handler.resize_image(image_out, self.default_height, image_width)
        return image_out


    def save_image(self, image_out: np.ndarray, digits: Iterable, save_dir="./"):
        """Save the generated image
        
        Parameters
        ----------
        image_out : np.ndarray
            ndarray of the output image
        digits : Iterable
            A list-like Iterable of the numerical values of
            the digits in the sequence
        save_dir : str, optional
            The destination directory
        
        Returns
        -------
        str
            The saved filename (saved in current directory)
        """
        filename = "seq_{0}.{1}".format(
                            "".join(map(str, digits)), 
                            self.config['output_format']
                        )
        filepath = os.path.join(save_dir, filename)

        img = Image.fromarray((image_out * 255).astype(np.uint8))
        img.save(filepath)
        self.logger.info("Image saved successfully (filepath: {})".format(filepath))
        return filepath
