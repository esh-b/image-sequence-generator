import random
from enum import Enum

import numpy as np
from PIL import Image

class SpacingHandler:

    """Handle spacing related tasks like calculating the space_width
       between digits (depending on the given spacing type and subtype).
    
    Attributes
    ----------
    default_width : int
        Default width of the digit images
    Types : Enum
        Enum listing the possible spacing types
            FIXED: width for all possible spacings are same
            VARIABLE: width for all the spacings are randomly
                      sampled from uniform distribution
    Subtypes : Enum
        This is a suboption for a given spacing type. Currently available
        values are:
            BETWEEN: Just consider the spaces between digits
            EDGE: Consider the spaces between digits and also near the edges
                  of the image sequence (before the first digit and after the last digit)
                  #EDGE_spaces = #BETWEEN_spaces + 2 (around the edges)
    type : Types
        Spacing type given for the current API
    subtype : Subtypes
        Spacing subtype given for the current API
    """
    
    Types = Enum('Types', [
                    'FIXED',
                    'VARIABLE'
                ])
    Subtypes = Enum('Subtypes', [
                        'EDGE',
                        'BETWEEN'
                    ])

    def __init__(self, spacing_config: dict, default_width: int):
        self.default_width = default_width
        self.type = self._get_spacing_type(spacing_config)
        self.subtype = self._get_spacing_subtype(spacing_config)

    def _get_spacing_type(self, spacing_config: dict):
        """Infer the given spacing type
        """
        given_type = spacing_config['type'].lower()
        if given_type == "fixed":
            return self.Types.FIXED
        elif given_type == "variable":
            return self.Types.VARIABLE
        else:
            raise TypeError("Given spacing-type is invalid. "
                            "Valid options are {}".format(
                            ", ".join([s.name for s in self.Types])))

    def _get_spacing_subtype(self, spacing_config: dict):
        """Infer the given spacing subtype
        """
        given_subtype = spacing_config['subtype'].lower()
        if given_subtype == "edge":
            return self.Subtypes.EDGE
        elif given_subtype == "between":
            return self.Subtypes.BETWEEN
        else:
            raise TypeError("Given spacing-subtype is invalid. "
                            "Valid options are {}".format(
                            ", ".join([s.name for s in self.Subtypes])))

    def _fixed_spacing(self, num_digits: int, spacing_range: tuple, image_width: int):
        """Return the spacing array assuming the spacing is fixed between digits
           and according to the spacing subtype
        
        Parameters
        ----------
        num_digits : int
            Number of digits in the sequence
        spacing_range : tuple
            Tuple containing the minimum and maximum spacing range
        image_width : int
            the width of the output image (in pixels)
        
        Returns
        -------
        np.ndarray
            A numpy array containing the spacing values for all possible spaces
            given fixed spacing and appropriate subspace type
        
        Raises
        ------
        RuntimeError
            Raised for the edge case when sequence has one digit and spacing type is `between`.
            This is because `between` spacing does not make sense for one digit sequence
        ValueError
            Raised when the calculated space width does not satisfy the constraints
        """

        # Edge case - sequence has one digit and spacing type is `between`
        if num_digits == 1 and self.subtype == self.Subtypes.BETWEEN:
            raise RuntimeError("Not possible to add spacing for one digit sequence "
                               "and `between` spacing type")

        avail_space_pixels = image_width - (self.default_width * num_digits)
        if avail_space_pixels < 0:
            raise ValueError("Given `image_width` is not sufficient to stack digits")

        # Calculate width for each of the possible spaces according to spacing type
        num_spacings = (num_digits-1) if self.subtype == self.Subtypes.BETWEEN \
                                     else (num_digits+1)
        space_width = avail_space_pixels / num_spacings
        if not space_width.is_integer():
            raise ValueError("Cannot create equal INT spacing "
                            "for the given spacing type."
                            "Try changing the spacing type or image_width")
        elif space_width < spacing_range[0]:
            raise ValueError("The calculated spacing seems to be "
                            "less than the minimum spacing value")
        elif space_width > spacing_range[1]:
            raise ValueError("The calculated spacing seems to be "
                            "greater than the maximum spacing value")

        spacing_arr = np.repeat(int(space_width), num_digits+1)
        # Replace edge spacings with 0 if spacing type is `between`
        if self.subtype == self.Subtypes.BETWEEN:
            np.put(spacing_arr, [0, -1], 0)
        return spacing_arr

    def _variable_spacing(self, num_digits: int, spacing_range: tuple, image_width: int):
        """Get the spacing array assuming the spacing is varied between digits
           and according to the spacing subtype
        
        Parameters
        ----------
        num_digits : int
            Number of digits in the sequence
        spacing_range : tuple
            Tuple containing the minimum and maximum spacing range
        image_width : int
            the width of the output image (in pixels)
        
        Returns
        -------
        np.ndarray
            A numpy array containing the spacing values for all possible spaces
            given variable spacing and appropriate subspace type
        
        Raises
        ------
        RuntimeError
            Raised for the edge case when sequence has one digit and 
            spacing type is `between`. This is because `between` spacing 
            does not make sense for one digit sequence
        """
        if num_digits == 1 and self.subtype == self.Subtypes.BETWEEN:
            raise RuntimeError("Not possible to add spacing for one digit sequence "
                               "and `between` spacing type")
        
        # sample from uniform distribution
        spacing_arr = np.random.randint(spacing_range[0], spacing_range[1], num_digits+1)
        if self.subtype == self.Subtypes.BETWEEN:
            np.put(spacing_arr, [0, -1], 0)
        return spacing_arr
    
    def get_spacing(self, num_digits: int, spacing_range: tuple, image_width: int):
        """Method to calculate spacings between each digit and return them as array
        
        Parameters
        ----------
        num_digits : int
            Number of digits in the sequence
        spacing_range : tuple
            Tuple containing the minimum and maximum spacing range
        image_width : int
            the width of the output image (in pixels)
        
        Returns
        -------
        np.ndarray
            A numpy array containing the spacing values for all possible spaces
            Example: Suppose there are 3 digits in the sequence, then the possible
            spaces can be shown by '|':
                | digit_1 | digit_2 | digit_3 |
            So, there will be (num_digits+1) spaces available. The returned numpy array
            contains the widths for each of those spaces.
            A width of 0 means that particular space doesnt exist in the image
        """
        if self.type == self.Types.FIXED:
            return self._fixed_spacing(num_digits, spacing_range, image_width)
        else:
            return self._variable_spacing(num_digits, spacing_range, image_width)
