import numpy as np
from scipy.special import softmax
import collections


class RegSensIn():

    def __init__(self, base_method, beta=1, L=20, num_iter=1000, M=10, weights=[.5, .5], sens_index=-1):

        """
        Parameters
        ----------
        base_method :
            A regression model already trained on labeled data.
            It must have a "predict" method, like in sklearn models.
        beta : int, optional
            The temperature parameter used in Nesterov's algorithm.
            It is set to 1 by default.
        L : int, optional
            We use 2L+1 points, to construct a uniform grid on the interval [-M, M].
        num_iter : int, optional
            Number of iterations in the Nesterov's algorithm
        M : int, optional
            Bound on the target
        weights : list, optional
            Proportions of each sensitive attribute in the data, respectively to -1 and 1.
        sens_index : int, optional
            The index of the sensitive attribute in the data.
            It is set to -1 by default.
        """

        self.base_method = base_method
        self.beta = beta
        self.L = L
        self.num_iter = num_iter
        self.M = M
        self.weights = weights
        self.sens_index = sens_index

    def fit(self, X_unlab):
        """Summary

        Parameters
        ----------
        X_unlab : Array
            An unlabeled dataset with sensitive attribute.
        """
        coef = np.zeros(2 * self.L + 1)
        moment = np.zeros(2 * self.L + 1)
        y_pred0 = self.base_method.predict(X_unlab[X_unlab[:,self.sens_index] == -1])
        y_pred1 = self.base_method.predict(X_unlab[X_unlab[:,self.sens_index] == 1])
        discr = np.arange(-self.L, self.L + 1) * self.M / self.L
        z0 = self.weights[0] * np.square(y_pred0[:, np.newaxis] - discr)
        z1 = self.weights[1] * np.square(y_pred1[:, np.newaxis] - discr)
        tau = 0
        for t in range(self.num_iter):
            tmp = (1 + np.sqrt(1 + 4 * tau ** 2)) / 2
            gamma = (1 - tau) / tmp
            tau = tmp
            coef_prev = coef
            coef = (moment - (self.beta / 2) * (np.mean(softmax((moment - z1) / self.beta, axis=1), axis=0) -
                                                np.mean(softmax((-moment - z0) / self.beta, axis=1), axis=0)))
            moment = (1 - gamma) * coef + gamma * coef_prev
        self.coef_ = coef
        self.discr_ = discr

    def predict(self, X):
        """Summary

        Parameters
        ----------
        X : Array
            (Test) Data to predict on;
             with sensitive attribute.

        Returns
        -------
        Array
            Prediction on X with fairness-adjusted regression model.
        """
        n_samples, _ = X.shape
        s = np.zeros(n_samples)
        s[X[:,self.sens_index] == -1] = -1
        s[X[:,self.sens_index] == 1] = 1
        z = np.square(self.base_method.predict(X)[:, np.newaxis] - self.discr_)
        z[X[:,self.sens_index] == -1, :] *= self.weights[0]
        z[X[:,self.sens_index] == 1, :] *= self.weights[1]

        return (np.argmin(-s[:,np.newaxis] * self.coef_ + z, axis=1) - self.L) * self.M / self.L




class RegSensOut():

    def __init__(self, base_method, classifier, beta=1, L=20, num_iter=1000, M=10, weights=[.5, .5]):

        """
        Parameters
        ----------
        base_method :
            A regression model already trained on labeled data.
            It must have a "predict" method, like in sklearn models.
        classifier : TYPE
            A classification model already trained on labeled data.
            It must have "predict_proba" and "classes_" methods, like in sklearn models.
        beta : int, optional
            The temperature parameter used in Nesterov's algorithm.
            It is set to 1 by default.
        L : int, optional
            We use 2L+1 points, to construct a uniform grid on the interval [-M, M].
        num_iter : int, optional
            Number of iterations in the Nesterov's algorithm
        M : int, optional
            Bound on the target
        weights : list, optional
            Proportions of each sensitive attribute in the data, respectively to -1 and 1.
        """

        self.base_method = base_method
        self.classifier = classifier
        self.beta = beta
        self.L = L
        self.num_iter = num_iter
        self.M = M
        self.weights = weights

    def fit(self, X_unlab):
        """Summary

        Parameters
        ----------
        X_unlab : Array
            An unlabeled dataset with sensitive attribute.
        """
        coef = np.zeros(2 * self.L + 1)
        moment = np.zeros(2 * self.L + 1)

        reg_pred = self.base_method.predict(X_unlab)
        clf_prob = self.classifier.predict_proba(X_unlab)
        tau_X = clf_prob[:, list(self.classifier.classes_).index(1)].reshape([-1,1])
        #p1 = collections.Counter(S)[ind]/len(S)
        p1 = self.weights[1]

        discr = np.arange(-self.L, self.L + 1) * self.M / self.L
        z = np.square(reg_pred[:, np.newaxis] - discr)

        tau = 0
        for t in range(self.num_iter):
            tmp = (1 + np.sqrt(1 + 4 * tau ** 2)) / 2
            gamma = (1 - tau) / tmp
            tau = tmp
            coef_prev = coef
            coef = moment - (self.beta) * (p1**2) * np.mean((1-tau_X/p1) * softmax(((1-tau_X/p1)*moment - z) /
                        self.beta, axis=1), axis=0)
            moment = (1 - gamma) * coef + gamma * coef_prev
        self.coef_ = coef
        self.discr_ = discr
        self.p1_ = p1

    def predict(self, X):
        """Summary

        Parameters
        ----------
        X : Array
            (Test) Data to predict on;
             with sensitive attribute.

        Returns
        -------
        Array
            Prediction on X with fairness-adjusted regression model.
        """
        n_samples, _ = X.shape

        z = np.square(self.base_method.predict(X)[:, np.newaxis] - self.discr_)
        clf_prob = self.classifier.predict_proba(X)
        tau_X = clf_prob[:, list(self.classifier.classes_).index(1)].reshape([-1,1])

        return (np.argmin(z + self.coef_ * (tau_X/self.p1_ - 1), axis=1) - self.L) * self.M / self.L