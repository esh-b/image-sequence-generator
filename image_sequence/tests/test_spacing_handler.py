import unittest

import numpy as np

from image_sequence.spacing_handler import SpacingHandler


class TestSpacingHandler(unittest.TestCase):

    def test_get_spacing__variable_edge(self):
        """Test `variable` type and `edge` subtype
        """
        dataset_config = {'type': 'variable', 'subtype': 'edge'}
        spacing_handler = SpacingHandler(dataset_config, default_width=28)

        digits, spacing_range, image_width = range(10), (1, 20), 390
        num_digits = len(digits)
        spacing_arr = spacing_handler.get_spacing(num_digits, spacing_range, image_width)

        # Assert return type
        self.assertIsInstance(spacing_arr, np.ndarray)

        # Assert the spacing widths are integers
        self.assertIs(spacing_arr.dtype, np.dtype(np.int))

        # Assert the return array length. The spacing array should
        # always return array of size (num_digits+1) which is
        # the number of possible spacings around the digits.
        self.assertEqual(len(spacing_arr), num_digits+1)

        # Assert all spacing values are >= 0.
        # Spacing widths cannot be negative
        self.assertTrue(np.all(spacing_arr >= 0))

        # Assert VARIABLE type (spacing values are different)
        self.assertFalse((spacing_arr == spacing_arr[0]).all())

        # Assert random sampling from UNIFORM distribution
        spacing_range_constraint = ((spacing_arr >= spacing_range[0])
                                   & (spacing_arr <= spacing_range[1]))
        self.assertTrue(spacing_range_constraint.all())

    def test_get_spacing__variable_between(self):
        """Test `variable` type and `between` subtype
        """
        dataset_config = {'type': 'variable', 'subtype': 'between'}
        spacing_handler = SpacingHandler(dataset_config, default_width=28)

        digits, spacing_range, image_width = range(10), (1, 20), 370
        num_digits = len(digits)
        spacing_arr = spacing_handler.get_spacing(num_digits, spacing_range, image_width)

        # Assert return type, space width type and return array length
        self.assertIsInstance(spacing_arr, np.ndarray)
        self.assertIs(spacing_arr.dtype, np.dtype(np.int))
        self.assertEqual(len(spacing_arr), num_digits+1)

        # Assert all spacing values are >= 0.
        self.assertTrue(np.all(spacing_arr >= 0))

        # Assert edge spacings have value `zero`
        self.assertEqual(spacing_arr[0], 0)
        self.assertEqual(spacing_arr[-1], 0)

        # Assert VARIABLE type (spacing values are different)
        self.assertFalse((spacing_arr[1:-1] == spacing_arr[1]).all())

        # Assert random sampling from UNIFORM distribution
        spacing_range_constraint = ((spacing_arr[1:-1] >= spacing_range[0])
                                   & (spacing_arr[1:-1] <= spacing_range[1]))
        self.assertTrue(spacing_range_constraint.all())

        # Assert that for sequence with one digit, `between` spacing type is not valid
        with self.assertRaises(RuntimeError):
            spacing_handler.get_spacing(1, spacing_range, image_width)

    def test_get_digit_spacing__fixed_edge(self):
        """Test `fixed` type and `edge` subtype
        """
        dataset_config = {'type': 'fixed', 'subtype': 'edge'}
        spacing_handler = SpacingHandler(dataset_config, default_width=28)

        digits, spacing_range, image_width = range(10), (1, 20), 390
        num_digits = len(digits)
        spacing_arr = spacing_handler.get_spacing(num_digits, spacing_range, image_width)

        # Assert return type, spacing width type and return array length
        self.assertIsInstance(spacing_arr, np.ndarray)
        self.assertIs(spacing_arr.dtype, np.dtype(np.int))
        self.assertEqual(len(spacing_arr), num_digits+1)

        # Assert all spacing values are >= 0.
        # Spacing widths cannot be negative
        self.assertTrue(np.all(spacing_arr >= 0))

        # Assert FIXED and EDGE type
        self.assertTrue((spacing_arr == spacing_arr[0]).all())

        # Assert spacing width satisfies spacing range
        self.assertTrue(spacing_range[0] <= spacing_arr[0] <= spacing_range[1])

        with self.assertRaises(ValueError):
            # not sufficient width to stack digits
            spacing_handler.get_spacing(num_digits, spacing_range, 270)
            # non integer spacing
            spacing_handler.get_spacing(num_digits, spacing_range, 300)
            # space width less than minimum value
            spacing_handler.get_spacing(num_digits, spacing_range, 280)
            # space width greater than maximum value
            spacing_handler.get_spacing(num_digits, spacing_range, 510)

    def test_get_digit_spacing__fixed_between(self):
        """Test `fixed` type and `between` subtype
        """
        dataset_config = {'type': 'fixed', 'subtype': 'between'}
        spacing_handler = SpacingHandler(dataset_config, default_width=28)

        digits, spacing_range, image_width = range(10), (1, 20), 370
        num_digits = len(digits)
        spacing_arr = spacing_handler.get_spacing(num_digits, spacing_range, image_width)

        # Assert return type, spacing width type and return array length
        self.assertIsInstance(spacing_arr, np.ndarray)
        self.assertIs(spacing_arr.dtype, np.dtype(np.int))
        self.assertEqual(len(spacing_arr), num_digits+1)

        # Assert all spacing values are >= 0.
        # Spacing widths cannot be negative
        self.assertTrue(np.all(spacing_arr >= 0))

        # Assert edge spacings have value `zero`
        self.assertEqual(spacing_arr[0], 0)
        self.assertEqual(spacing_arr[-1], 0)

        # Assert FIXED and BETWEEN type
        self.assertTrue((spacing_arr[1:-1] == spacing_arr[1]).all())

        # Assert spacing width satisfies spacing range
        self.assertTrue(spacing_range[0] <= spacing_arr[1] <= spacing_range[1])

        # Assert that for sequence with one digit, `between` spacing type
        # is not valid
        with self.assertRaises(RuntimeError):
            spacing_handler.get_spacing(1, spacing_range, image_width)

        with self.assertRaises(ValueError):
            # not sufficient width to stack digits
            spacing_handler.get_spacing(num_digits, spacing_range, 270)
            # non integer spacing
            spacing_handler.get_spacing(num_digits, spacing_range, 300)
            # space width less than minimum value
            spacing_handler.get_spacing(num_digits, spacing_range, 280)
            # space width greater than maximum value
            spacing_handler.get_spacing(num_digits, spacing_range, 510)
