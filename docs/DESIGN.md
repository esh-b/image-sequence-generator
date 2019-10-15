# Algorithm and module design

## Spacing concept in the digit sequence
- Any digit sequence will take the form `|| DIGIT_1 || DIGIT_2 ||` where `||` represent the possible spacing gaps in the image sequence.
- Hence, given `n` digits as input, there can be a maximum of `n+1` spacings (`n-1` spacings inbetween digits and 2 edge spacings) in the final image sequence. 
- All these spacing widths (in pixels) combined with the widths of the digits (in pixels) must be equal to the given `image_width` argument.
- **Please note** that in the rest of the document, the term **spacing** refers to the spacing gap (the location of `||` above) while the term **space width** (or spacing width) refers to the spacing width in pixels of these spacings.

## Algorithm description
The algorithm has the following steps:
1. First, the `generate_numbers_sequence` function takes the digits list and selects an image at random for each digit in the input list.
2. Then, depending on the provided configuration, it calculates the spacing width for all the (n+1) spacings (see the spacing concept section).
3. Then, an image array is created by concatenating the spacings and the digits alternatively.

These steps are explained below in detail.

### Step 1:
- The MNIST train dataset (with 60k images) is used to generate image sequences.
- First, the train dataset is loaded from the package/module's `dataset/` directory. If the dataset doesnt exist, then it is downloaded and saved in that directory.
- Once the MNIST train dataset is loaded, all the indices containing a particular label are grouped together and saved in a list (named `grouped_labels`) data structure. This list is basically a map from labels to the list of indices containing that label i.e. the index `i` in `grouped_labels` contains all the indices with the label `i`.
    - Example: `grouped_labels[0]` is a list containing indices with label `0` in the train dataset.
- Then, on iterating over every digit in the input list, we select a random index from `grouped_labels[digit]`. The value at this index in the image array is chosen as the image for that particular digit in the final image sequence.
- **transform**: 
    - We can optionally transform every selected image using the image processing pipelines (depending on the value of `transform` key in the config file) before adding them in the image sequence. These pipelines can be configured to use various image processors. 
    - These transformations might change the range of the digit images. Hence, these transformed images are then normalized to range 0-1 using the `cv2.normalize` function. **Note** that the final image sequence array may have values slightly less than 0 or slightly more than 1 since the normalizations cannot be perfect.
- At the end of this step, we return an array of size `(len(input_list), 28, 28)` (array of all the digits).
- *Optimization step*: The `grouped_labels` data structure is stored onto the filesystem (in json format) once created so that the next time the program is run, the data structure can be directly loaded instead of creating it again (which is time-consuming operation).

### Step 2:
- As mentioned in the configuration section of the `README.md` in the root directory, there are currently 2 spacing types
and 2 spacing subtypes available.
- If the *variablewidth + edge* option is chosen, then the spacing width for all possible spacings (including the edge spacings) are randomly drawn from an uniform distribution (with args of minimum and maximum value from the `spacing_range` input).
- If the *variablewidth + between* option is chosen, the concept is the same as above except that only the spacings between digits are considered (and not the edge spacings).
- If the *fixedwidth + edge* option is chosen, then all possible spacings (including the edge spacings) are considered and all the spacings have the same spacing width (i.e. the available pixels for spaces are shared equally by all the spacings).
- If the *fixedwidth + between* option is chosen, its the same as above except that only the spacings between digits are considered (and not the edge spacings).
- For consistency, in all the cases, we return an array of size (`n+1`) where `n` is the number of digits in the input sequence. It is just that for `between` types, the first and last elements of the array are `0` (meaning that edge spacings' space width is `0`).

### Step 3
- After step 1 and 2, we get the array of digits(with shape `(n, 28, 28)`) and the array of spacings (with shape `(n+1)`).
- Next, we iterate over all the elements in the spacing array such that at every step we create a gap array (using `np.ones`) of shape `(image height, space width)`, then select a digit from the digit array and concatenate them together.
- At every step, the result from the earlier iteration is concatenated with the result of the current iteration. This way, an image sequence is generated.
- Then, if we have used *variablewidth* spacing type, it means that the final image sequence may not have the same width as `image_width`(input argument) due to sampling random values for space widths from a uniform distribution. Hence, the image sequence is resized to the required shape of (28, `image_width`) and returned as the final output.

## Module design
The `image_sequence` module is written in the object-oriented paradigm and that makes it easier to scale and add features in the future.

The module uses 4 classes:
1. `DigitSequence` class (`generator.py`)
- This is the main class instantiated in the API and in the client script. This class has the method `generate_numbers_sequence` to generate the image sequences.

2. `DatasetHandler` class
- This utility class handles all dataset related tasks like downloading dataset, loading dataset etc.

3. `ImageHandler` class
- This utility class handles all tasks related to processing individual digit images like retrieving the digit images given the label, transforming the digit images etc

4. `SpacingHandler` class
- This utility class handles all spacing related tasks like calculating space widths for all the spacings depending on the provided configuration.

### Algorithm description with respect to these classes
- On initialization, the `DigitSequence` object loads the configuration file, instantiates the utility classes and then loads dataset and the `grouped_labels` data structure. If the config file was not provided, it uses the `config.json` file in the module directory as the default configuration.
- Once the `generate_number_sequence` is called, the input is validated, the digit image array and the spacing array are obtained and the final image sequence is generated by iterating over those arrays. This method returns the image in `ndarray` format having values of type `float32` and values ranging from 0 (black) to 1 (white).
- Then, once the `save_image` method of the object is called with the generated image array (from previous step) as argument, it saves the array in the appropriate format in the current directory. The saved filename follows the pattern *seq_<sequence_values>.[png/jpg/jpeg]* (Example: *seq_012345689.png*).

### Advantages of this design
- Separating classes like this helps to add features easily without too many changes in the other modules. For example, to assign space widths to the spacings based on a new technique, the new algorithm has to be defined in the `SpacingHandler` class and a method has to be written to process which returns the spacing array. 
- Since methods in all the classes perform one task (*single responsibility rule*), it is easier to change or tweak features in the future.
- **For general-purpose image sequence generations:**
    - The module is designed to work for general-purpose image sequence generation tasks.
    - For generic datasets, the code can be slightly tweaked to get the dataset information (path, image width, image height) from the config file and the algorithm could simply use these config to work on any generic dataset.
    - Reading this way from the config file will help to generate the image sequences for any given dataset without many changes in the code.
    - Also, we could easily add other image processing options and spacing options as required.
- **Generating large dataset:**
    - The object-oriented design helps to initialize the `DigitSequence` object once and then call its `generate_number_sequence` method any number of times without loading the data everytime. The details of the performance to get an image for various config options are provided in the [`META.md`](META.md) file.
    - This makes it feasible to generate large dataset with different varieties (spacing types, image transformation etc) at a rapid pace.

