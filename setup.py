import setuptools

with open("README.md", "r") as fp:
    long_description = fp.read()

setuptools.setup(
    name="digit-sequence",
    version="0.0.1",
    author="Eshwanth",
    author_email="eshwanth.95@gmail.com",
    description="A package to generate image sequence of MNIST digits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'image_sequence': ['config.json'],
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'Pillow',
        'imgaug'
    ],
    python_requires='>=3.5',
)
