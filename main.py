import numpy as np
import MNIST_Dataset.LoadMNIST as mnist

from MyRBM.RMB_Classifier import Classifier


### Loading MNIST dataset

mnist.load('MNIST_Dataset/')

sparse_training_labels = np.zeros((mnist.training_images_count, 10))

for i, label in enumerate(mnist.training_labels):
    sparse_training_labels[i][int(label)] = 1.0

sparse_testing_labels = np.zeros((mnist.test_images_count, 10))

for i, label in enumerate(mnist.test_labels):
    sparse_testing_labels[i][int(label)] = 1.0

train_images = mnist.training_images.reshape((mnist.training_images.shape[0], mnist.images_size)) / 255
test_images = mnist.test_images.reshape((mnist.test_images.shape[0], mnist.images_size)) / 255

def get_accuracy(model: Classifier, data: np.array, labels: np.array):
    successes = 0
    for x, y in zip(data, labels):
        if y == model.class_predict(x):
            successes += 1

    return successes / data.shape[0]


for persistence in [True, False]:
    print('\n\n\nPersistence: ', persistence)

    model = Classifier(mnist.images_size, 10, 100, True)

    # initial accuracy is expected to be around 0.1 for obvious reasons
    print('Initial test accuracy: ', get_accuracy(model, test_images, mnist.test_labels))

    ### Training the model

    # schedule
    s = np.array([1e-1, 5e-2, 1e-2, 5e-3])

    model.fit_classifier(train_images, sparse_training_labels, lr=s[0], persistence=persistence, epochs=5, wd=1e-3, momentum=0)
    print('Test accuracy: ', get_accuracy(model, test_images, mnist.test_labels))

    model.fit_classifier(train_images, sparse_training_labels, lr=s[1], persistence=persistence, epochs=5, wd=1e-3, momentum=0)
    print('Test accuracy: ', get_accuracy(model, test_images, mnist.test_labels))

    model.fit_classifier(train_images, sparse_training_labels, lr=s[2], persistence=persistence, epochs=5, wd=1e-3, momentum=0)
    print('Test accuracy: ', get_accuracy(model, test_images, mnist.test_labels))

    # model.fit_classifier(train_images, sparse_training_labels, lr=s[3], persistence=persistence, epochs=5, wd=1e-3, momentum=0)
    # print('Test accuracy: ', get_accuracy(model, test_images, mnist.test_labels))
