import random
from collections import Iterable

import cv2
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


class ImageHandler:

    """Handle generation of digit images, their transformation
       depending on config and related tasks.
    
    Attributes
    ----------
    transform : bool
        A bool indicating whether to transform the images of the digits.
        Transforming an image means applying some random image processing
        technique like blur, multiply etc
    transform_pipeline : <imgaug.augmenters.meta.Sequential>
        The image transformation pipeline. This is valid 
        only if the `transform` config is True.
    """
    def __init__(self, image_config):
        self.transform = image_config['transform']
        if not isinstance(self.transform, bool):
            raise ValueError("`transform` value in config file must be a boolean")

        if self.transform:
            # Transform pipeline containing various image processing techniques.
            # iaa.Sometimes(p, ...) method runs the processing on an image with
            # probability `p`.
            self.transform_pipeline = iaa.SomeOf(10, [
                # iaa.Sometimes(0.2, iaa.Multiply((0.9, 1.2), per_channel=0.2)),
                iaa.Sometimes(0.1, iaa.GaussianBlur(sigma=(0, 0.05))),
                iaa.Sometimes(0.15, iaa.ElasticTransformation(alpha=(0.5, 0.6), sigma=0.15)),
                iaa.Sometimes(0.3, iaa.Dropout(0.1)),
                iaa.Sometimes(0.3, iaa.Crop(percent=(0, 0.25)))
            ], random_order=True)

    def _normalize_image(self, image: np.ndarray):
        """Normalize given image to range 0-1
        """
        return cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)

    def _transform_images(self, digits_images: np.ndarray):
        """Apply image processing techniques to transform the digit images
        
        Parameters
        ----------
        digits_images : np.ndarray
            The original array containing images of digits in the sequence
        
        Returns
        -------
        np.ndarray
            The transformed and normalized `digits_images` array
        """
        transformed_images = self.transform_pipeline(images=digits_images)
        return np.array(list(map(self._normalize_image, transformed_images)))

    def get_digits_images(self, digits: Iterable, grouped_labels: list, images: np.ndarray):
        """Return images for the given digits as a numpy array 
        
        Parameters
        ----------
        digits : Iterable
            A list-like Iterable containing numerical values of
            the digits in the sequence.
        grouped_labels : list
            Loaded grouped labels list
        images : np.ndarray
            Loaded images numpy array
        
        Returns
        -------
        np.array
            Return the numpy array containing the images for the given digit sequence
        """
        # For every digit in the sequence, select a random index containing that digit
        label_indices = [random.choice(grouped_labels[digit]) for digit in digits]

        # Select the values at `label_indices` index from `images` array
        digits_images = np.take(images, label_indices, axis=0)
        return self._transform_images(digits_images) if self.transform else digits_images

    def resize_image(self, image: np.ndarray, target_height: int, target_width: int):
        """Resize the given image to target width and height
        
        Parameters
        ----------
        image : np.ndarray
            The image to resize
        target_height : int
            The target height
        target_width : int
            The target width
        
        Returns
        -------
        np.ndarray
            The resized image
        """
        img = Image.fromarray(image)
        return np.array(img.resize((target_width, target_height)))
