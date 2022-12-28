import numpy as np


def load_labels(filepath: str) -> (int, np.ndarray):
    file = open(filepath, 'rb')
    magic_number = int.from_bytes(file.read(4), 'big')
    if magic_number != 2049:
        print('Wrong file format for ' + filepath + '!')
        return 0, []

    number_of_items = int.from_bytes(file.read(4), 'big')
    res = np.ndarray(shape=(number_of_items, 1))
    for i in range(number_of_items):
        res[i][0] = int.from_bytes(file.read(1), 'big')

    return number_of_items, res


def load_images(filepath: str) -> (int, int, int, np.ndarray):
    file = open(filepath, 'rb')
    magic_number = int.from_bytes(file.read(4), 'big')
    if magic_number != 2051:
        print('Wrong file format for ' + filepath + '!')
        return 0, 0, 0, []

    number_of_items = int.from_bytes(file.read(4), 'big')
    number_of_rows = int.from_bytes(file.read(4), 'big')
    number_of_columns = int.from_bytes(file.read(4), 'big')
    res = np.ndarray(shape=(number_of_items, number_of_rows, number_of_columns))

    for i in range(number_of_items):
        for p in range(number_of_rows * number_of_columns):
            res[i][p // number_of_columns][p % number_of_columns] = int.from_bytes(file.read(1), 'big')
    return number_of_items, number_of_rows, number_of_columns, res


training_label_count, training_labels = 0,None
training_images_count, training_images_rows, training_images_cols, training_images = 0, 0, 0, None
test_labels_count, test_labels = 0, None
test_images_count, test_images_rows, test_images_cols, test_images = 0, 0, 0, None
images_size = 0


def load(str)->None:
    # importing training data

    global training_label_count, training_labels
    training_label_count, training_labels = load_labels(str + 'train-labels.idx1-ubyte')

    global training_images_count, training_images_rows, training_images_cols, training_images
    training_images_count, training_images_rows, training_images_cols, training_images \
        = load_images(str + 'train-images.idx3-ubyte')

    assert training_label_count == training_images_count != 0

    # importing testing data

    global test_labels_count, test_labels
    test_labels_count, test_labels = load_labels(str + 't10k-labels.idx1-ubyte')

    global test_images_count, test_images_rows, test_images_cols, test_images
    test_images_count, test_images_rows, test_images_cols, test_images = load_images(str + 't10k-images.idx3-ubyte')

    assert test_labels_count == test_images_count \
           and training_images_rows == test_images_rows \
           and training_images_cols == test_images_cols

    global images_size
    images_size = training_images_rows * training_images_cols

