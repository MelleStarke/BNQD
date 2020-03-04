import gpflow as gpf
import numpy as np
from optimizers import *
import tensorflow as tf
import plotter as plt


# Superclass only used as abstract class
class GPRegressionModel:

    def __init__self(self, x, y, kernel, lik=gpf.likelihoods.Gaussian):
        raise NotImplementedError

    def train(self, optimizer=gpf.optimizers.Scipy, verbose=False):
        raise NotImplementedError

    def predict(self, x_test):
        raise NotImplementedError

    def log_marginal_likelihood(self, mode='BIC'):
        raise NotImplementedError

    def plot(self, x_test, axis=None, color=None):
        raise NotImplementedError


class ContinuousModel(GPRegressionModel):
    isOptimized = False

    def __init__(self, x, y, kernel, lik=gpf.likelihoods.Gaussian):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.kernel = kernel
        # Manual construction sometimes adds Gaussian white noise,
        # sometimes does not???
        #        self.m = GPy.core.GP(X = x, Y = y, kernel = self.kernel, likelihood = lik)
        self.m = gpf.models.GPR((x, y), self.kernel)
        self.ndim = np.ndim(x)
        self.BICscore = None

    #
    def train(self, optimizer=None, verbose=False):
        """Train the continuous model
        """
        if optimizer is None:
            optimizer = gpf.optimizers.Scipy()
        optimizer.minimize(lambda: - self.m.log_marginal_likelihood(), variables=self.m.trainable_variables)
        self.isOptimized = True

    #
    def predict(self, x_test):
        if len(x_test.shape) == 1:
            x_test = np.atleast_2d(x_test).T
        return self.m.predict_y(x_test)

    #
    def log_marginal_likelihood(self, mode='BIC'):
        """Computes the log marginal likelihood for the continuous model. Since
        this is intractable, we instead approximate it.

        :param mode: Selects how to approximate the evidence. Currently, only
        BIC is implemented, which is a crude approximation, but works well in
        our examples and simulations.

        :return: Returns log p(D|M).
        """
        if mode == 'BIC':
            if not self.isOptimized:
                print('Parameters have not been optimized; training now')
                self.train()

            if self.BICscore is None:
                k = self.m.num_params
                L = self.m.log_likelihood()
                BIC = L - k / 2 * np.log(self.n)
                self.BICscore = BIC
            return self.BICscore
        elif mode in ['laplace', 'Laplace']:
            raise NotImplementedError('Laplace approximation is not yet implemented')
        elif mode == 'AIS':
            raise NotImplementedError('Annealed importance sampling is not yet implemented')
        else:
            raise NotImplementedError('Unrecognized marginal likelihood approximation {:s}'.format(mode))

    #
    def plot(self, x_test, axis=None, plotOptions=dict(), scaleFunction=None,
             scaleData=None):
        if axis is None:
            axis = plt.gca()

        color = plotOptions.get('color', 'darkgreen')
        alpha = plotOptions.get('alpha', 0.3)
        linestyle = plotOptions.get('linestyle', 'solid')
        label = plotOptions.get('label', 'Optimized prediction')

        mu, Sigma2 = self.predict(x_test)
        Sigma = np.sqrt(Sigma2)

        lower = np.squeeze(mu - 0.5 * Sigma)
        upper = np.squeeze(mu + 0.5 * Sigma)

        if scaleFunction is not None:
            mu = scaleFunction(mu, scaleData)
            lower = scaleFunction(lower, scaleData)
            upper = scaleFunction(upper, scaleData)

        if self.kernel.input_dim == 1:
            axis.plot(x_test, mu, label=label, color=color,
                      linestyle=linestyle)
            axis.fill_between(x_test, lower, upper, alpha=alpha, color=color,
                              edgecolor='white')
        elif self.kernel.input_dim == 2:
            p = int(np.sqrt(x_test.shape[0]))
            x0 = np.reshape(x_test[:, 0], newshape=(p, p))
            x1 = np.reshape(x_test[:, 1], newshape=(p, p))
            mu_res = np.reshape(mu, newshape=(p, p))
            axis.plot_surface(X=x0, Y=x1, Z=mu_res, color=color,
                              antialiased=True, alpha=0.5, linewidth=0)

            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
        else:
            raise ('Dimensionality not implemented')
    #