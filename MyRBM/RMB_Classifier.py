from MyRBM.rbm0 import RBM
import numpy as np


def sofmax(x: np.ndarray):
    a = np.exp(x)
    return a / np.sum(a)


class Classifier(RBM):

    def __init__(self, input_size, class_numer, hidden_units, sample_binary):
        """
        :param input_size: the number of inputs
        :param class_numer: the number of classes
        :param hidden_units: the number of hidden units
        :param sample_binary: should be true if the input is real-valued; makes samples binary
        """

        self.inputs = input_size
        self.classes = class_numer
        super().__init__(input_size + class_numer, hidden_units, sample_binary)

    def fit_classifier(self, X: np.ndarray, Y: np.ndarray, epochs=10, batch_size=16, gibb_samples=1, lr=1e-3, wd=1e-6, persistence=False, momentum=0.8):
        new_data = np.concatenate((X, Y), axis=1)
        super().fit(new_data, epochs, batch_size, gibb_samples, lr, wd, persistence, momentum)

    def unnormalized_predict(self, x):
        un_prob = []
        for c in range(self.classes):
            b = np.zeros(self.classes)
            b[c] = 1.0  # the label
            new_x = np.concatenate((x, b))
            un_prob.append(self.un_p(new_x))

        return un_prob

    def predict(self, x):

        un_prob = self.unnormalized_predict(x)

        # normalizing values before applying softmax to avoid numerical issues
        mean = np.mean(un_prob)
        std = np.std(un_prob)
        pseudo_n_prob = (un_prob - mean) / std

        return sofmax(pseudo_n_prob)

    def class_predict(self, x):

        un_prob = self.unnormalized_predict(x)

        return np.argmax(un_prob)

