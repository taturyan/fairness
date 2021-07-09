import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class FairRegressionTransform():
    def __init__(self, base_method, split=True, sigma=1e-6):
        self.base_method = base_method
        self.split = split
        self.sigma = sigma
    def fit(self, X_unlab, weights):
        self.weights = weights
        self.sensitives = np.unique(X_unlab[:, -1])
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
            self.quantile[s] = pred1[X1[:,-1] == s]
            self.cdf[s] = pred2[X2[:,-1] == s]
    def predict(self, X):
        n_test, _ = X.shape
        unfair = self.base_method.predict(X) + np.random.uniform(-self.sigma, self.sigma, n_test)
        y_pred = np.zeros(n_test)
        for ind, pred in enumerate(unfair):
            q = np.sum(self.cdf[X[ind,-1]] <= pred) / len(self.cdf[X[ind,-1]])
            for s in self.sensitives:
                y_pred[ind] += self.weights[s] * np.quantile(self.quantile[s], q, interpolation='lower')
        return y_pred