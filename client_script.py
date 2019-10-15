import sys
import argparse

from image_sequence.generator import DigitSequence

def main(digits, spacing_range, image_width, config_filepath=None):
    generator = DigitSequence(config_filepath)
    gen_image = generator.generate_numbers_sequence(digits, spacing_range, image_width)
    output_filename = generator.save_image(gen_image, digits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("digits", help="digits separated by commas")
    parser.add_argument("min_spacing", type=int, help="minimum spacing between digits")
    parser.add_argument("max_spacing", type=int, help="maximum spacing between digits")
    parser.add_argument("image_width", type=int, help="the width of the output image (in pixels)")
    parser.add_argument("-c", "--config", help="the path to the config file")
    args = parser.parse_args()
    
    try:
        digits = [int(digit) for digit in args.digits.split(',')]
    except ValueError:
        raise ValueError("All the digits in the list must be an integer")

    main(digits, (args.min_spacing, args.max_spacing), args.image_width, args.config)