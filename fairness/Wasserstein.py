import numpy as np

class FairRegressionTransform():

    def __init__(self, base_method, split=True, sigma=1e-6, alpha=0, sens_index=-1):
        """Summary

        Parameters
        ----------
        base_method :
            A regression model already trained on labeled data.
            It must have a "predict" method, like in sklearn models.
        split : bool, optional
            If True, the unlabeled data must be splitted in the algorithm.
             It is set to True by default.
        sigma : float, optional
            The interval of the noise added to prediction.
        alpha : int, optional
            Relative Improvement constraint from [0,1] range.
            It is set to 0 by default.
        sens_index : int, optional
            The index of the sensitive attribute in the data.
            It is set to -1 by default.
        """
        self.base_method = base_method
        self.split = split
        self.sigma = sigma
        self.alpha = alpha
        self.sens_index = sens_index

    def fit(self, X_unlab, weights = {-1 : 0.5, 1 : 0.5}):

        """
        Parameters
        ----------
        X_unlab : Array
            An unlabeled dataset with sensitive attribute.
        weights : dict
             Weights of the sensitive attribute.
             It is set to {-1 : 0.5, 1 : 0.5} by default.
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
            (Test) Data to predict on;
             with sensitive attribute.

        Returns
        -------
        TYPE
            Prediction on X with fairness-adjusted regression model.
        """
        n_test, _ = X.shape
        unfair = self.base_method.predict(X) + np.random.uniform(-self.sigma, self.sigma, n_test)
        y_pred = np.zeros(n_test)
        for ind, pred in enumerate(unfair):
            q = np.sum(self.cdf[X[ind, self.sens_index]] <= pred) / len(self.cdf[X[ind, self.sens_index]])
            for s in self.sensitives:
                y_pred[ind] += self.weights[s] * np.quantile(self.quantile[s], q, interpolation='lower')
        return np.sqrt(self.alpha) * self.base_method.predict(X) + (1 - np.sqrt(self.alpha)) *  y_pred

