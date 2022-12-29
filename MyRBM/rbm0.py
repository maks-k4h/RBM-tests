import numpy as np


def softplus(x):
    return np.log(1 + np.exp(x))


def sigm(x):
    return 1 / (1 + np.exp(-x))


def get_stochastic_binary(x):
    return (np.random.rand(*x.shape) < x) + 0.0


class RBM:

    def __init__(self, visible_units, hidden_units, sample_binary):
        """
        :param visible_units: the number of visible units
        :param hidden_units: the number of hidden units
        :param sample_binary: should be true if the input is real-valued and has to be cast to binary
        """

        # M — the number of visible units
        # H — the number of hidden units

        # W — weights matrix
        # Bv — bias of the visible layer
        # Bh — bias of the hidden layer

        self.V = visible_units
        self.H = hidden_units

        self.W = np.random.normal(0, 1e-2, size=(self.H, self.V))
        self.Bv = np.random.normal(0, 1e-2, size=self.V)
        self.Bh = np.random.normal(0, 1e-2, size=self.H)


        self.sample_binary = sample_binary

    def F(self, x):
        """
        Computes free energy F(x).

        :param x: visible unit state
        :return: free energy of the model given x
        """

        spa = softplus(self.Bh + self.W @ x)
        a1 = np.inner(self.Bv, x)
        a2 = np.sum(spa)
        return - a1 - a2

    def un_p(self, x: np.ndarray):
        """
        Computes not normalized probability p(x).

        :param x: visible unit state
        :return: s, where p(x) = s/Z = exp(-F(x))/Z
        """

        if self.sample_binary:
            x = x.round()
        return np.exp(-self.F(x))

    def sample_h(self, x):
        """
        Samples hidden units given visible units.

        :param x: visible units
        :return: hidden units
        """

        # sampling hidden units' states from uniform distribution with given
        # conditional probabilities
        return get_stochastic_binary(self.p_given_x(x))

    def sample_x(self, h):
        """
        Samples visible units given hidden units.

        :param h: hidden units
        :return: visible units
        """

        # sampling visible units' states from uniform distribution with given
        # conditional probabilities
        return get_stochastic_binary(self.p_given_h(h))

    def p_given_x(self, x):
        """
        For every hidden unit calculates its probability of being equal to 1
        given visible units.

        :param x: visible units
        :return: conditional probabilities
        """

        a1 = self.Bh
        a2 = self.W @ x

        return sigm(a1 + a2)

    def p_given_h(self, h):
        """
        For every visible unit calculates its probability of being equal to 1
        given hidden units.

        :param h: hidden units
        :return: conditional probabilities
        """

        a1 = self.Bv
        a2 = h @ self.W
        return sigm(a1 + a2)

    def gibb_sample(self, x, iterations=1):
        """
        Sequentially performs Gibb sampling of hidden and visible units
        given the initial state of visible units.

        :param x: initial visible units
        :param iterations: number of Gibb sampling iterations
        :return: a Gibb sample of visible units on the last iteration
        """

        x_tilda = x
        for i in range(iterations):
            x_tilda = self.sample_x(self.sample_h(x_tilda))

        return x_tilda

    def fit(self, X: np.ndarray, epochs=10, batch_size=16, gibb_samples=1, lr=1e-3, wd=1e-6, persistence=True,
            momentum=0.8):
        """
        Makes the model to give high probabilities to samples from the distribution
        of train data and low to those that are not.

        Warning: X is shuffled

        :param X: train data,
        :param epochs: the number of epochs
        :param batch_size: the size of batches
        :param gibb_samples: the number of Gibb samples per training sample. Ignored if PCD is used.
        :param lr: learning rate
        :param wd: weight decay
        :param persistence: use Persistence Contrastive Divergence
        :param momentum: momentum coefficient
        :return:
        """

        print('Training parameters:')
        print('Persistence: ', 'on' if persistence else 'off')
        print('Sample binary: ', 'on' if self.sample_binary else 'off')
        print('Learning rate: ', lr)
        print('Weight decay: ', wd)
        print('Gibb samples: ', gibb_samples)
        print('Batch size: ', batch_size)
        print('Momentum: ', momentum)
        print('Epochs: ', epochs)
        print()

        N = X.shape[0]
        batch_size = min(N, batch_size)

        # moments
        Bv_m = np.zeros_like(self.Bv)
        Bh_m = np.zeros_like(self.Bh)
        W_m = np.zeros_like(self.W)

        np.random.shuffle(X)

        persistence_batch = None
        if persistence:
            persistence_batch = X[0:batch_size]
            # making binary anyway
            persistence_batch = get_stochastic_binary(persistence_batch)

        for epoch in range(epochs):
            print("EPOCH {}...".format(epoch + 1))

            for bn in range(N // batch_size):
                batch = X[bn * batch_size: (bn + 1) * batch_size]

                # the gradient of positive log likelihood (so we perform ascending)
                grad_Bv = np.zeros_like(self.Bv)
                grad_Bh = np.zeros_like(self.Bh)
                grad_W = np.zeros_like(self.W)

                for i, x_t in enumerate(batch):
                    if self.sample_binary:
                        x_t = get_stochastic_binary(x_t)

                    # performing gibb sampling
                    if persistence:
                        x_tilda = self.gibb_sample(persistence_batch[i], 1)
                        # updating memo
                        persistence_batch[i] = x_tilda
                    else:
                        x_tilda = self.gibb_sample(x_t, gibb_samples)

                    h_t = self.sample_h(x_t)

                    # as is recommended, use pure probabilities
                    # when computing the last state of hidden units
                    # in the chain
                    h_tilda = self.p_given_x(x_tilda)

                    # accumulating the gradients
                    grad_Bv += x_t - x_tilda
                    grad_Bh += h_t - h_tilda
                    grad_W += np.outer(h_t, x_t) - np.outer(h_tilda, x_tilda)

                # averaging the gradients
                grad_Bv /= batch_size
                grad_Bh /= batch_size
                grad_W /= batch_size

                # moments
                Bv_m = grad_Bv + momentum * Bv_m
                Bh_m = grad_Bh + momentum * Bh_m
                W_m = grad_W + momentum * W_m

                # updating weights
                self.Bv += lr * Bv_m - wd * self.Bv
                self.Bh += lr * Bh_m - wd * self.Bh
                self.W += lr * W_m - wd * self.W

        print('Done!')
