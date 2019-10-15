"""Script to show the usage of the package as an API
"""
from image_sequence.generator import DigitSequence

def main():
    digits, spacing_range, image_width = range(10), (0, 20), 379

    # default config
    generator = DigitSequence()

    # custom config
    # generator = DigitSequence("./example_config.json")

    gen_image = generator.generate_numbers_sequence(digits, spacing_range, image_width)
    saved_filepath = generator.save_image(gen_image, digits)

if __name__ == "__main__":
    main()
