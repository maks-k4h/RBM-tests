import numpy as np


def softplus(x):
    return np.log(1 + np.exp(x))


def sigm(x):
    return 1 / (1 + np.exp(-x))


class RBM:

    def __init__(self, visible_units, hidden_units):

        # M — the number of visible units
        # H — the number of hidden units

        # W — weights matrix
        # Bv — bias of the visible layer
        # Bh — bias of the hidden layer

        self.V = visible_units
        self.H = hidden_units

        self.W = np.random.rand(self.H, self.V)
        self.Bv = np.random.rand(self.V)
        self.Bh = np.random.rand(self.H)

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

    def un_p(self, x):
        """
        Computes not normalized probability p(x).

        :param x: visible unit state
        :return: s, where p(x) = s/Z = exp(-F(x))/Z
        """
        return np.exp(-self.F(x))

    def sample_h(self, x):
        """
        Samples hidden units given visible units.

        :param x: visible units
        :return: hidden units
        """

        # sampling hidden units' states from uniform distribution with given
        # conditional probabilities
        return (np.random.rand(self.H) < self.p_given_x(x)) + 0.0

    def sample_x(self, h):
        """
        Samples visible units given hidden units.

        :param h: hidden units
        :return: visible units
        """

        # sampling visible units' states from uniform distribution with given
        # conditional probabilities
        return (np.random.rand(self.V) < self.p_given_h(h)) + 0.0

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

    def fit(self, X: np.ndarray, epochs=10, batch_size=16, gibb_samples=1, lr=1e-3, wd=1e-5, persistence=False, momentum=0.8):
        """
        Makes the model to give high probabilities to samples from the distribution
        of train data and low to those that are not.

        Warning: X is shuffled, if persistence is set — changed.

        :param X: train data,
        :param epochs: the number of epochs
        :param batch_size: the size of batches
        :param gibb_samples: the number of Gibb samples per training sample
        :param lr: learning rate
        :param wd: weight decay
        :param persistence: use Persistence Contrastive Divergence
        :param momentum: momentum coefficient
        :return:
        """

        N = X.shape[0]
        batch_size = min(N, batch_size)

        # moments
        Bv_m = np.zeros_like(self.Bv)
        Bh_m = np.zeros_like(self.Bh)
        W_m = np.zeros_like(self.W)

        for epoch in range(epochs):
            np.random.shuffle(X)

            for bn in range(N // batch_size):
                batch = X[bn * batch_size: (bn + 1) * batch_size]

                # the gradient of positive log likelihood (so we perform ascending)
                grad_Bv = np.zeros_like(self.Bv)
                grad_Bh = np.zeros_like(self.Bh)
                grad_W = np.zeros_like(self.W)

                for i, x in enumerate(batch):

                    # performing gibb sampling
                    x_tilda = self.gibb_sample(x, gibb_samples)

                    # accumulating the gradients
                    grad_Bv += x - x_tilda
                    grad_Bh += self.sample_h(x) - self.sample_h(x_tilda)
                    grad_W += np.outer(self.sample_h(x), x) - np.outer(self.sample_h(x_tilda), x_tilda)

                    if persistence:
                        X[bn*batch_size + i] = x_tilda

                # averaging the gradients
                grad_Bv /= batch_size
                grad_Bh /= batch_size
                grad_W /= batch_size

                # moments
                Bv_m = (1-momentum) * grad_Bv + momentum * Bv_m
                Bh_m = (1-momentum) * grad_Bh + momentum * Bh_m
                W_m = (1-momentum) * grad_W + momentum * W_m

                # updating weights
                self.Bv += lr * Bv_m - wd * self.Bv
                self.Bh += lr * Bh_m - wd * self.Bh
                self.W += lr * grad_W - wd * self.W








