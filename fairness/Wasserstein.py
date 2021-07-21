import numpy as np

class FairRegressionTransform():

    def __init__(self, base_method, split=True, sigma=1e-6, alpha=0, sens_index=-1):
        """Summary

        Parameters
        ----------
        base_method : TYPE
            Description
        split : bool, optional
            Description
        sigma : float, optional
            Description
        alpha : int, optional
            Description
        sens_index : TYPE, optional
            Description
        """
        self.base_method = base_method
        self.split = split
        self.sigma = sigma
        self.alpha = alpha
        self.sens_index = sens_index  # ndarray

    def fit(self, X_unlab, weights):

        """
        Parameters
        ----------
        X_unlab : TYPE
            Description
        weights : TYPE
            Description
        """

        self.weights = weights
        self.sensitives = np.unique(X_unlab[:, self.sens_index])
        K = len(self.sensitives)
        assert K != 1, 'Only one sensitive attribute provided'
        assert len(self.weights) == K, 'There are {} sensitive features, but {} weights'.format(K, len(self.weights))
        N, _ = X_unlab.shape
        permutation = np.random.permutation(N)
        if self.split:
            tmp = int(N / 2)
            X1 = X_unlab[permutation[:tmp]]
            X2 = X_unlab[permutation[tmp:]]
        else:
            X1 = X_unlab
            X2 = X_unlab
        pred1 = self.base_method.predict(X1) + np.random.uniform(-self.sigma, self.sigma, tmp)
        pred2 = self.base_method.predict(X2) + np.random.uniform(-self.sigma, self.sigma, N - tmp)
        self.quantile = {}
        self.cdf = {}
        for s in self.sensitives:
            self.quantile[s] = pred1[X1[:, self.sens_index] == s]
            self.cdf[s] = pred2[X2[:, self.sens_index] == s]

    def predict(self, X):
        """Summary

        Parameters
        ----------
        X : TYPE
            Description

        Returns
        -------
        TYPE
            Description
        """
        n_test, _ = X.shape
        unfair = self.base_method.predict(X) + np.random.uniform(-self.sigma, self.sigma, n_test)
        y_pred = np.zeros(n_test)
        for ind, pred in enumerate(unfair):
            q = np.sum(self.cdf[X[ind, self.sens_index]] <= pred) / len(self.cdf[X[ind, self.sens_index]])
            for s in self.sensitives:
                y_pred[ind] += self.weights[s] * np.quantile(self.quantile[s], q, interpolation='lower')
        return np.sqrt(self.alpha) * y_pred + (1 - np.sqrt(self.alpha)) * self.base_method.predict(X)
