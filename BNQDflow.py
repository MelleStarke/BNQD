import gpflow
from gpflow.optimizers import Scipy
from gpflow.likelihoods import Gaussian
import numpy as np
import tensorflow as tf
import BNQD
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os
import warnings
import importlib
import scipy.stats as stats
import bisect
import abc

warnings.simplefilter('ignore')
importlib.reload(BNQD)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# plt.style.use('seaborn-dark-palette') # plt.style.use('ggplot')
np.random.seed(1)

print("GPFlow version:      {}".format(gpflow.__version__))
print("BNQD version:     {}".format(BNQD.__version__))


class GPRegressionModel():
    """
    Abstract class for continuous and discontinuous models
    """

    def __init__(self, X, Y, kernel, D, likelihood=Gaussian(), mean_function=None):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.kernel = kernel
        self.likelihood = likelihood
        self.mean_function = mean_function
        self.ndim = D
        self.BICscore = None
        self.isOptimized = False

    def train(self, optim, max_iter, verbose=False):
        raise NotImplementedError

    def predict(self, X_test, n_samples):
        raise NotImplementedError

    def get_log_marginal_likelihood(self):
        raise NotImplementedError

    def plot(self, X_test, mean, var, samples, b = None):
        raise NotImplementedError



class ContinuousModel(GPRegressionModel):

    def __init__(self, X, Y, kernel, D, likelihood= Gaussian(), mean_function=None, noise_variance=1.0):
        super().__init__(X, Y, kernel, D, likelihood, mean_function)
        self.m = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=self.mean_function, noise_variance = noise_variance)


    def train(self, optim = Scipy(), max_iter=1000, verbose=True):
        # Minimization
        def objective_closure():
            return - self.m.log_marginal_likelihood()

        opt_logs = optim.minimize(objective_closure,
                                  self.m.trainable_variables,
                                  options=dict(maxiter=max_iter))
        if verbose:
            gpflow.utilities.print_summary(self.m)
        self.isOptimized = True


    def predict(self, X_test, n_samples = 5):
        # predict mean and variance of latent GP at test points
        if len(X_test.shape)==1:
            X_test = np.expand_dims(X_test, axis = 1)
        mean, var = self.m.predict_f(X_test)

        # generate n samples from posterior
        samples = self.m.predict_f_samples(X_test, n_samples)
        return mean, var, samples


    def get_log_marginal_likelihood(self):
        """Computes the log marginal likelihood for the dichotomous model.
        Since this is intractable, we instead approximate it.

        :param mode: Selects how to approximate the evidence. Currently, only
        BIC is implemented, which is a crude approximation, but works well in
        our examples and simulations.

        :return: Returns log p(D|M).
        """
        if not self.isOptimized:
            print('Parameters have not been optimized; training now')
            self.train()

        if self.BICscore is None:
            k = len(self.m.parameters)  # parameters are represented as tuple (documentation is found in gpflow.Module)
            L = self.m.log_likelihood()
            BIC = L - k / 2 * np.log(self.n)
            self.BICscore = BIC
        return self.BICscore

    def example_gpflow_plot(self, X_test, mean, var, samples, true_func = None, b = None):
        plt.figure(figsize=(12,8))
        plt.plot(self.X, self.Y, 'kx', mew=2)
        plt.plot(X_test, mean, 'C0', lw=2)
        plt.fill_between(X_test[:, 0],
                        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
                        color='C0', alpha=0.2)

        plt.plot(X_test, samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
        if mu is not None:
            plt.plot(X_test, true_func, label='True function', linewidth=2.0, color='black')
        if b is not None:
            plt.axvline(x=b, color='black', linestyle=':')
        plt.set_xlabel = "x"
        plt.set_ylabel = 'y'
        name = self.kernel.__class__.__name__
        if name == 'Sum':
            name = self.kernel.kernels[0].name + ' and ' + self.kernel.kernels[1].name
        plt.title("Posterior prediction using the " + name + " kernel")


    def plot(self, x_test, axis=None, plotOptions=dict(), scaleFunction=None,
             scaleData=None, conf_interval = True, n_samples = 5):

        """
        - Currently plots n_samples as well unless stated otherwise
        - plots the 95% confidence interval (1.96 should be changed to 0.5 to reverse).

        """
        if axis is None:
            axis = plt.gca()

        color = plotOptions.get('color', 'darkgreen')
        alpha = plotOptions.get('alpha', 0.2)
        linestyle = plotOptions.get('linestyle', 'solid')
        label = plotOptions.get('label', 'Optimized prediction')

        mu, var, samples = self.predict(x_test, n_samples)
        mu = np.squeeze(mu)
        var = np.squeeze(var)
        sigma = np.sqrt(var)

        lower = np.squeeze(mu - 1.96 * sigma) # np.squeeze(mu - 0.5 * Sigma)
        upper = np.squeeze(mu + 1.96 * sigma) # np.squeeze(mu + 0.5 * Sigma)
        x_test = np.squeeze(x_test)

        # ToDo: scale function not tested yet
        if scaleFunction is not None:
            mu = scaleFunction(mu, scaleData)
            lower = scaleFunction(lower, scaleData)
            upper = scaleFunction(upper, scaleData)

        if self.ndim == 1:
            axis.plot(x_test, mu, label=label, color=color,
                      linestyle=linestyle)

            axis.fill_between(x_test, lower, upper, alpha=alpha, color=color, edgecolor='white')
            # Plot samples drawn from posterior
            if n_samples > 0:
                axis.plot(x_test, samples[:, :, 0].numpy().T, color=color,
                      alpha = alpha*2, linewidth=.5)

        # ToDo: 2-dimensional plot not tested yet
        elif self.ndim == 2:
            print('Made it to 2d :D')
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



class DiscontinuousModel(GPRegressionModel):

    def __init__(self, X, Y, kernel, D, label_func=None,labelLUT=None,
                 likelihood= Gaussian(), mean_function=None, noise_variance=1.0):
        super().__init__(X, Y, kernel, D, likelihood, mean_function)

        self.label_func = label_func
        self.labelLUT = labelLUT
        if self.label_func is None:
            lab1 = self.labelLUT == 0
        else:
            lab1 = label_func(X)
        lab2 = np.logical_not(lab1)

        # ugly Numpy behaviour 2.0
        x1, x2 = X[lab1,], X[lab2,]
        y1, y2 = Y[lab1,], Y[lab2,]

        # ToDo: suggested shorter notation.
        # data = [x1, x2, y1, y2]
        # for d in data:
        #     if len(d.shape)==1:
        #         d = np.expand_dims(d, axis=1)
        #         #d = d.reshape(d.shape[0], self.ndim)
        # x1, x2, y1, y2 = data
        if len(x1.shape) == 1:
            x1 = np.expand_dims(x1, axis=1)
        if len(x2.shape) == 1:
            x2 = np.expand_dims(x2, axis=1)
        if len(y1.shape) == 1:
            y1 = np.expand_dims(y1, axis=1)
        if len(y2.shape) == 1:
            y2 = np.expand_dims(y2, axis=1)

        # Create two continuous models
        m1 = ContinuousModel(x1, y1, kernel, D, likelihood, mean_function, noise_variance)
        m2 = ContinuousModel(x2, y2, kernel, D, likelihood, mean_function, noise_variance)
        self.submodels = [m1, m2]

    def train(self, optim = Scipy(), max_iter=1000, verbose=True, share_hyp = False):
        param_list = list()
        for submodel in self.submodels:
            submodel.train(optim, max_iter, verbose)
            param_list.append(submodel.m.trainable_parameters)

        if share_hyp:
            raise NotImplementedError
        self.isOptimized = True

    def predict(self, x_test, n_samples = 5, mask=True):
        """Predict the values for the test predictors for each of the two sub-
        models.

        :param mask: If true, only the predictions for the corresponding model
        are shown; otherwise the predictions for the full range of x_test are
        given.

        :return: GP predictions (means, variances, samples) per model.
        """
        if np.isscalar(x_test):
            return (self.submodels[0].predict(np.atleast_2d(np.array(x_test)), n_samples=n_samples),
                    self.submodels[1].predict(np.atleast_2d(np.array(x_test)), n_samples=n_samples))
        else:
            if mask:
                # Label test data
                lab1 = self.label_func(x_test)
                lab2 = np.logical_not(lab1)
                x1 = x_test[lab1,]
                x2 = x_test[lab2,]
                # Change shape of data from (N,) to (N,1)
                if len(x1.shape) == 1:
                    x1 = np.expand_dims(x1, axis=1)
                if len(x2.shape) == 1:
                    x2 = np.expand_dims(x2, axis=1)

                return (self.submodels[0].predict(x1, n_samples=n_samples),
                        self.submodels[1].predict(x2, n_samples=n_samples))
            else:
                return (self.submodels[0].predict(x_test, n_samples=n_samples),
                        self.submodels[1].predict(x_test, n_samples=n_samples))

    def get_log_marginal_likelihood(self, mode='BIC'):
        """Computes the log marginal likelihood for the dichotomous model.
        Since this is intractable, we instead approximate it.

        :param mode: Selects how to approximate the evidence. Currently, only
        BIC is implemented, which is a crude approximation, but works well in
        our examples and simulations.

        :return: Returns log p(D|M).
        """
        if not self.isOptimized:
            print('Parameters have not been optimized; training now')
            self.train()
        if self.BICscore is None:
            BIC = 0
            for model in self.submodels:
                k = len(model.m.parameters)  # parameters are represented as tuple (documentation is found in gpflow.Module)
                L = model.m.log_likelihood()
                BIC = L - k / 2 * np.log(self.n)
            self.BICscore = BIC
        return self.BICscore


    def plot(self, x_test, axis=None, plotOptions=None, b=0.0,
             plotEffectSize=False, scaleFunction=None, scaleData=None,
             plotFullRange=False, n_samples = 5):
        if axis is None:
            axis = plt.gca()

        def add_boundary(x, b):
            if not np.isin(b, x):
                tmp = list(x)
                bisect.insort(tmp, b)
                return np.array(tmp)
            else:
                return x

        if plotOptions is None:
            plotOptions = [dict(), dict()]
        ms1 = plotOptions[0].get('markersize', 10)
        ms2 = plotOptions[1].get('markersize', 10)

        if not plotFullRange:
            if self.label_func is None:
                lab1 = self.labelLUT == 0
            else:
                lab1 = np.array([self.label_func(i) for i in x_test])
            #            lab1 = x_test < b
            lab2 = np.logical_not(lab1)
            x1, x2 = x_test[lab1,], x_test[lab2,]

        else:
            x1, x2 = x_test, x_test

        # for printing purposes mainly
        x1 = add_boundary(x1, b)
        x2 = add_boundary(x2, b)

        if self.ndim == 1:
            # Plot the fit of the two submodels
            x1 = np.expand_dims(x1, axis=1)
            x2 = np.expand_dims(x2, axis=1)
            self.submodels[0].plot(x1, axis=axis, plotOptions=plotOptions[0],
                                scaleFunction=scaleFunction, scaleData=scaleData, n_samples=n_samples)
            self.submodels[1].plot(x2, axis=axis, plotOptions=plotOptions[1],
                                scaleFunction=scaleFunction, scaleData=scaleData, n_samples=n_samples)

            # Predict at the intervention threshold
            m0b, v0b, _ = self.submodels[0].predict(np.array([b]), n_samples)

            m1b, v1b, _ = self.submodels[1].predict(np.array([b]), n_samples)

            if scaleFunction is not None:
                m0b = scaleFunction(m0b, scaleData)
                v0b = scaleFunction(v0b, scaleData)
                m1b = scaleFunction(m1b, scaleData)
                v1b = scaleFunction(v1b, scaleData)

            if plotEffectSize:
                axis.plot([b, b], [np.squeeze(m0b), np.squeeze(m1b)], c='k',
                          linestyle='-', marker=None, linewidth=3.0, zorder=10)
                axis.plot(b, m0b, c='k', marker='o', markeredgecolor='k',
                          markerfacecolor='lightgrey', ms=ms1, zorder=10)
                axis.plot(b, m1b, c='k', marker='o', markeredgecolor='k',
                          markerfacecolor='lightgrey', ms=ms2, zorder=10)

            return (m0b, v0b), (m1b, v1b)

        # ToDo: 2d not tested
        elif self.ndim == 2:

            mu1, _, _ = self.submodels[0].predict(x1, n_samples)
            mu2, _, _ = self.submodels[1].predict(x2, n_samples)

            p = int(np.sqrt(x_test.shape[0]))
            mu1_aug = np.zeros((p * p, 1))
            mu1_aug.fill(np.nan)
            mu1_aug[lab1,] = mu1
            mu1_aug = np.reshape(mu1_aug, newshape=(p, p))

            mu2_aug = np.zeros((p * p, 1))
            mu2_aug.fill(np.nan)
            mu2_aug[lab2,] = mu2
            mu2_aug = np.reshape(mu2_aug, newshape=(p, p))

            x0 = np.reshape(x_test[:, 0], newshape=(p, p))
            x1 = np.reshape(x_test[:, 1], newshape=(p, p))
            axis.plot_surface(X=x0, Y=x1, Z=mu1_aug,
                              color=plotOptions[0]['color'], antialiased=True,
                              alpha=0.5, linewidth=0)
            axis.plot_surface(X=x0, Y=x1, Z=mu2_aug,
                              color=plotOptions[1]['color'], antialiased=True,
                              alpha=0.5, linewidth=0)

            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
            plt.show()
        else:
            raise ('Dimensionality not implemented')


class BnpQedModel():
    """The analysis object for one single kernel. Includes a continuous and a
    discontinuous model.
    """

    isOptimized = False
    log_BF_10 = None
    summary_object = None
    BFmode = ''

    def __init__(self, x, y, kernel, D, label_func=None, labelLUT=None, likelihood = Gaussian(),
                 mean_function = None, noise_variance = 1.0,  mode='BIC', design='Generic'):

        self.x = x
        self.y = y
        self.ndim = D
        self.label_func = label_func
        self.labelLUT = labelLUT

        if np.ndim(x) == 1 and len(x.shape) == 1:
            x = np.atleast_2d(x).T

        if len(y.shape) == 1:
            y = np.atleast_2d(y).T
        # Todo: Kernel in gpflow does not work with the copy function. Passing the original kernel might give wrong results.
        self.CModel = ContinuousModel(x, y, kernel, D, likelihood, mean_function, noise_variance)
        self.DModel = DiscontinuousModel(x, y, kernel, D, label_func, labelLUT, likelihood,
                                         mean_function, noise_variance)
        self.BFmode = mode
        self.design = design

    def train(self, optim = Scipy(), max_iter=1000, b=0.0, verbose = True):
        """Train both the continuous and the discontinuous model using GPy.

        We use the default implementation of GPy, which is L-BFGS.
        Multiple restarts are used to avoid settling for a local optimum.

        :param num_restarts: scalar
        :return:    The BNP-QED summary object containing statistics of the
                    model comparison.
        """

        self.CModel.train(optim, max_iter, verbose = verbose)
        self.DModel.train(optim, max_iter, verbose = verbose, share_hyp = False)
        self.isOptimized = True
        #return self.summary(mode=self.BFmode, b=b)

    def predict(self, x_test, n_samples = 5):
        """ Predicting the responses of either model for unseen X.

        :param x_test: Range of predictor values, used for either interpolating
                       or extrapolating.
        :return:    The posterior predictive mean and variance for every point
                    in x_test, for
                        1) the continuous model and
                        2) BOTH discontinuous models.
        """

        return self.CModel.predict(x_test, n_samples), self.DModel.predict(x_test, n_samples)


    def get_log_Bayes_factor(self, mode='BIC'):
        """ The Bayes factor given the specified data and kernel.

        The Bayes factor is defined as
            BF_DC = p(D | M_D) / p(D | M_C),
        where M_D, M_C represent the discontinuous and continuous models,
        respectively.

        For numerical stability, we compute the logarithm of the Bayes factor.

        :param mode: The approximation strategy for computing model evidence.
        :return: The log Bayes factor
        """

        if not self.isOptimized:
            self.train()
        if self.log_BF_10 is None:
            self.log_BF_10 = self.DModel.get_log_marginal_likelihood() \
                             - self.CModel.get_log_marginal_likelihood()
        return self.log_BF_10

    def discEstimate(self, b=0.0):
        """The predictions of the discontinuous model at the boundary value b.

        :param b: The boundary value. Can be called repeatedly for multi-
        dimensional discontinuity estimates.
        :return: Returns a tuple of means and a typle of variances for the two
        Gaussian distributions that correspond to the predictions by the
        discontinuous model at b.

        """
        m0b, v0b, _ = self.DModel.submodels[0].predict(np.array([b]), n_samples = 0)
        m1b, v1b, _ = self.DModel.submodels[1].predict(np.array([b]), n_samples = 0)
        return (m0b, m1b), (v0b, v1b)

    def get_posterior_model_probabilities(self, mode='BIC'):
        """Computes the posterior model probabilities p(M|D)

        We have:
            p(M_D|D) / P(M_C|D) = p(D|M_D) / p(D|M_C) p(M_D) / M(M_C)
        and
            p(M_D|D) + p(M_C|D) = 1
        so if we assume the prior model probabilities are equal, we have:
            p(M_D|D) / (1 - p(M_D|D)) = p(D|M_D) / p(D|M_C)
            and
            BF_DC = p(D|M_D) / p(D|M_C)
            -> p(M_D|D) = BF_DC / (1 + BF_DC)

        :param mode: The approximation method for the marginal likelihood.
        :return: A dict with of posterior model probability per model.
        """
        # Note: assumes uniform prior!
        bf = np.exp(self.get_log_Bayes_factor(mode))
        if np.isinf(bf):
            return {'pmc': 0.0, 'pmd': 1.0}
        else:
            pmd = bf / (1 + bf)
            pmc = 1 - pmd
            return {'pmc': pmc, 'pmd': pmd}

    def summary(self, mode='BIC', b=0.0):
        """A function aggregating all derived statistics after the models have
        been trained.

        :param mode: The approximation method for the marginal likelihood.
        :param b: The boundary value; assumes a threshold label function. Will
                    be made generic later.
        :return: A dictionary containing:
            - logbayesfactor
            - evidence (marginal likelihoods)
            - pmp (posterior model probabilities)
            And for ndim == 1:
            - es_BMA (effect size across M_D and M_C)
            - es_Disc (effect size for M_D)
            - pval (p-value of discontinuity; uses a z-test)
            - es_range (the range over which we estimate a density of the
                           effect size)
            - f(b) (the predictions at b for the two models in M_D)
            - es_transform (the transformation from standardized units to the
                            actual effect size)
        """
        if self.summary_object is None:
            if mode is None:
                mode = self.BFmode
            summ = dict()
            summ['logbayesfactor'] = self.get_log_Bayes_factor(mode)
            summ['evidence'] = \
                {'mc': self.CModel.get_log_marginal_likelihood(),
                 'md': self.DModel.get_log_marginal_likelihood()}
            summ['pmp'] = self.get_posterior_model_probabilities(mode)

            # ToDo: effect size not implemented yet.
            # if self.ndim == 1 and self.design != 'DiD':
            #     # compute effect size
            #
            #     es = self.get_effect_size(summ, b)
            #     for k, v in es.items():
            #         summ[k] = v
            #
            # elif self.ndim == 2 and self.design != 'DiD':
            #     warnings.warn('Computing 2D effect size with Monte Carlo may take a while.')
            #
            #     m = len(b)
            #     es = {i: self.get_effect_size(summ, b[i]) for i in range(m)}
            #
            #     for k, v in es[0].items():
            #         summ[k] = [es[i][k] for i in range(m)]
            #
            # else:
            #     warnings.warn('Effect size analysis for D = {:d} not implemented.'.format(self.ndim))
            self.summary_object = summ
        return self.summary_object

    def plot(self, x_test, axis=None, b=0.0, plotEffectSize=False, mode='BIC'):
        summary = self.summary(mode=mode, b=b)
        pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
        LBF = summary['logbayesfactor']

        if self.ndim == 1:
            if plotEffectSize:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                                    figsize=(16, 6))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                               figsize=(16, 6),
                                               sharex=True, sharey=True)
            if self.label_func is None:
                lab1 = self.labelLUT == 0
            else:
                lab1 = self.label_func(self.x)
            lab2 = np.logical_not(lab1)

            x1 = np.expand_dims(self.x[lab1,], axis=1)
            x2 = np.expand_dims(self.x[lab2,], axis=1)
            y1 = np.expand_dims(self.y[lab1,], axis=1)
            y2 = np.expand_dims(self.y[lab2,], axis=1)

            ax1.plot(x1, y1, linestyle='', marker='o',
                     color='k')
            ax1.plot(x2, y2, linestyle='', marker='x',
                     color='k')
            self.CModel.plot(x_test, ax1)
            ax1.axvline(x=b, color='black', linestyle=':')
            ax1.set_title(r'Continuous model, $p(M_C \mid x)$ = {:0.2f}'.format(pmc))
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_xlim([np.min(self.x),np.max(self.x)])

            ax2.plot(x1, y1, linestyle='', marker='o',
                     color='k')
            ax2.plot(x2, y2, linestyle='', marker='x',
                     color='k')
            ax2.axvline(x=b, color='black', linestyle=':')
            m0stats, m1stats = self.DModel.plot(x_test,
                                                ax2,
                                                [{'colors': ('firebrick', 'firebrick')},
                                                 {'colors': ('firebrick', 'firebrick')}],
                                                b=b,
                                                plotEffectSize=plotEffectSize)
            ax2.set_title(r'Discontinuous model, $p(M_D \mid x)$ = {:0.2f}'.format(pmd))
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            # ToDo: get rid of hardcoding xlim
            ax2.set_xlim([np.min(self.x), np.max(self.x)])

            # ToDo: Effect size not implemented yet.
            if plotEffectSize:
                # create ES plot
                xmin, xmax = summary['es_interval']
                n = 100
                xrange = np.linspace(xmin, xmax, n)
                y = summary['es_Disc']
                pval = summary['pval']
                d_bma = summary['es_BMA']
                ax3.plot(xrange, y, c='firebrick', label=r'$M_D$',
                         linewidth=2.0, linestyle='--')
                ax3.fill_between(xrange, y, np.zeros((n)), alpha=0.1,
                                 color='firebrick')
                ax3.axvline(x=0, linewidth=2.0, label=r'$M_C$',
                            color='darkgreen', linestyle='--')
                ax3.plot(xrange, d_bma, c='k', label=r'BMA', linewidth=2.0)
                ax3.fill_between(xrange, d_bma, np.zeros((n)), alpha=0.1,
                                 color='k')
                ax3.legend(loc='best')
                ax3.set_xlabel(r'$\delta$')
                ax3.set_ylabel('Density')
                ax3.set_title(r'Size of discontinuity ($p$ = {:0.3f})'.format(pval))
                ax3.set_ylim(bottom=0)
                ax3.set_xlim([xmin, xmax])

            fig.suptitle(r'GP RDD analysis, log BF10 = {:0.4f}'.format(LBF))
            if plotEffectSize:
                return fig, (ax1, ax2, ax3)
            else:
                return fig, (ax1, ax2)
        elif self.ndim == 2:
            fig = plt.figure(figsize=(14, 6))

            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            if self.label_func is None:
                lab1 = self.labelLUT == 0
            else:
                lab1 = self.label_func(self.x)
            lab2 = np.logical_not(lab1)
            ax1.scatter(self.x[lab1, 0], self.x[lab1, 1], self.y[lab1,],
                        marker='o', c='black')
            ax1.scatter(self.x[lab2, 0], self.x[lab2, 1], self.y[lab2,],
                        marker='x', c='black')
            self.CModel.plot(x_test, ax1)
            ax1.set_title('Continuous model, p(M|x) = {:0.2f}'.format(pmc))
            ax1.set_xlabel(r'$x_1$')
            ax1.set_ylabel(r'$x_2$')
            ax1.set_zlabel('y')

            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.scatter(self.x[lab1, 0], self.x[lab1, 1], self.y[lab1,],
                        marker='o', c='black')
            ax2.scatter(self.x[lab2, 0], self.x[lab2, 1], self.y[lab2,],
                        marker='x', c='black')
            ax2.set_xlabel(r'$x_1$')
            ax2.set_ylabel(r'$x_2$')
            ax2.set_zlabel('y')
            self.DModel.plot(x_test, ax2, colors=('firebrick', 'coral'))
            ax2.set_title('Continuous model, p(M|x) = {:0.2f}'.format(pmd))
            fig.suptitle('GP RDD analysis, log BF10 = {:0.4f}'.format(LBF))
            return fig, (ax1, ax2)
        else:
            raise NotImplementedError('Dimensionality not implemented')


class BnpQedAnalysis():

    def __init__(self, x, y, kernel_dict, D, label_func=None, labelLUT=None, b=0.0, opts=dict()):
        self.x = x
        self.y = y
        self.kernel_dict = kernel_dict
        self.K = len(kernel_dict)
        self.ndim = D
        self.label_func = label_func  # function to label any point
        self.labelLUT = labelLUT  # look-up-table for provided points

        assert label_func is not None or labelLUT is not None, 'Provide either a label function or look-up-table'
        self.b = b
        self.max_iter = opts.get('max_iter', 1000)
        self.optim = opts.get('optim', Scipy())
        self.mode = opts.get('mode', 'BIC')
        self.likelihood = opts.get('likelihood', Gaussian())
        self.verbose = opts.get('verbose', True)
        self.noise_variance = opts.get('noise_variance', 1.0)
        self.mean_function = opts.get('mean_function', None)
        self.trained = False
        self.results = dict()
        self.total_disc_es = None
        self.total_disc_pdf = None
        self.total_bma_es = None
        self.total_bma_pdf = None
        self.rdd_p_values = None
        self.design = opts.get('Design', 'Generic')

    def train(self):
        """Trains the different models.

        Trains the continuous and discontinuous models for each of the provided
        kernels.
        """

        for kernel_name, kernel in self.kernel_dict.items():
            if self.verbose: print('Training with {:s} kernel'.format(kernel_name))
            model = BnpQedModel(self.x, self.y, kernel, self.D, self.label_func,
                                self.labelLUT, self.likelihood, self.noise_variance,
                                self.mode, self.design)
            model.train(self.optim, self.max_iter, b=self.b, verbose = self.verbose)
            if self.verbose:
                print('Log Bayes factor in favor of discontinuity = {:0.2f}'.format(
                    model.summary(b=self.b)['logbayesfactor']))
                print('Evidence: M_C = {:0.3f}, M_D = {:0.3f}'.format(model.summary(b=self.b)['evidence']['mc'],
                                                                      model.summary(b=self.b)['evidence']['md']))
                print('Posterior model probabilities: p(M_C|D) = {:0.3f}, p(M_D|D) = {:0.3f}'.format(
                    model.summary(b=self.b)['pmp']['pmc'],
                    model.summary(b=self.b)['pmp']['pmd']))
                print('')
            self.results[kernel_name] = model
        self.trained = True
        return self.results

    def plot_model_fits(self, x_test, plot_opts=dict()):
        """Basic plotting functionality.

        Details of the plots, such as ticklabels, can be added to the figures
        later via the fig and axes objects. Left column shows all continuous
        models, the right column shows all discontinuous models.

        :param x_test: The range over which the models are interpolated for
                       visualization.
        :param plot_opts: Visualization options.
        :return: The figure and axes handles.
        """

        cmodel_color = plot_opts.get('cmodel_color', 'black')
        dmodel_pre_color = plot_opts.get('dmodel_pre_color', '#cc7d21')
        dmodel_post_color = plot_opts.get('dmodel_post_color', 'darkgreen') # '#0e2b4d'
        color_data = plot_opts.get('color_data', '#334431')
        marker_pre = plot_opts.get('marker_pre', 'x')
        marker_post = plot_opts.get('marker_post', 'o')
        marker_size = plot_opts.get('marker_size', 5)
        marker_alpha = plot_opts.get('marker_alpha', 1.0)
        plot_effect_size = plot_opts.get('plot_effect_size', True)
        plot_title = plot_opts.get('plot_title', 'Model fits')
        plot_samewindow = plot_opts.get('plot_same_window', False)
        axes = plot_opts.get('axes', None)
        plot_full_range = plot_opts.get('plot_full_range',
                                        self.label_func is None)
        plot_xlim = plot_opts.get('plot_xlim',
                                  [np.min(self.x), np.max(self.x)])
        plot_ylim = plot_opts.get('plot_ylim',
                                  [np.min(self.y), np.max(self.y)])

        if not plot_samewindow:
            if axes is None:
                fig, axes = plt.subplots(nrows=self.K, ncols=2, sharex=True,
                                         sharey=True, figsize=(12, 6 * self.K))
            else:
                fig = plt.gcf()

            for i, kernel_name in enumerate(self.kernel_dict.keys()):
                self.results[kernel_name].CModel.plot(x_test, axes[i, 0],
                                                      plotOptions={'color': cmodel_color})
                self.results[kernel_name].DModel.plot(x_test, axes[i, 1],
                                                      b=self.b,
                                                      plotOptions=({'color': dmodel_pre_color},
                                                                   {'color': dmodel_post_color}),
                                                      plotEffectSize=plot_effect_size,
                                                      plotFullRange=plot_full_range)
                axes[i, 0].set_ylabel(kernel_name)
                summary = self.results[kernel_name].summary(b=self.b)
                pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
                axes[i, 0].set_title('p(M_C | x, y) = {:0.3f}'.format(pmc))
                axes[i, 1].set_title('p(M_D | x, y) = {:0.3f}'.format(pmd))
        else:
            if axes is None:
                fig, axes = plt.subplots(nrows=self.K, ncols=1, sharex=True,
                                         sharey=True, figsize=(6, 6 * self.K))
            else:
                fig = plt.gcf()

            # ToDo: names in title not working yet.
            for i, kernel_name in enumerate(self.kernel_dict.keys()):
                self.results[kernel_name].CModel.plot(x_test, axes[i],
                                                      plotOptions={'color': cmodel_color})
                self.results[kernel_name].DModel.plot(x_test, axes[i],
                                                      b=self.b,
                                                      plotOptions=({'color': dmodel_pre_color},
                                                                   {'color': dmodel_post_color}),
                                                      plotEffectSize=plot_effect_size,
                                                      plotFullRange=plot_full_range)
                axes[i].set_ylabel(kernel_name)
                summary = self.results[kernel_name].summary(b=self.b)
                pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
                axes[i].set_title('p(M_C | x, y) = {:0.3f}, p(M_D | x, y) = {:0.3f}'.format(pmc, pmd))

        for ax in axes.flatten():
            ax.axvline(x=self.b, color='black', linestyle='--')
            if self.label_func is None:
                lab1 = self.labelLUT == 0
            else:
                lab1 = self.label_func(self.x)
            lab2 = np.logical_not(lab1)

            x1 = np.expand_dims(self.x[lab1,], axis=1)
            x2 = np.expand_dims(self.x[lab2,], axis=1)
            y1 = np.expand_dims(self.y[lab1,], axis=1)
            y2 = np.expand_dims(self.y[lab2,], axis=1)

            ax.plot(x1, y1, linestyle='None',
                    marker=marker_pre, color=color_data, alpha=marker_alpha,
                    ms=marker_size)
            ax.plot(x2, y2, linestyle='None',
                    marker=marker_post, color=color_data, alpha=marker_alpha,
                    ms=marker_size)
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
        plt.suptitle(plot_title)
        return fig, axes

        # Testing samples
#------------------------------
# Create data
N = 100
D = 1
b = 0.4
n_samples_prior = 4
X = np.random.rand(N,D)
label_func = lambda x: x < b

def mean_function(x):
    return np.sin(12*x) + 0.66*np.cos(25*x) + 3
Y = mean_function(X) + np.random.randn(N,D)*0.1

# test points for prediction
xx = np.linspace(-100, 100, 100).reshape(100, 1)
mu = mean_function(xx)

# Kernels
linear = gpflow.kernels.Linear()
rbf = gpflow.kernels.RBF()
periodic = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
kernels = [linear, rbf, periodic]
kernel_names = ['Linear', 'rbf', 'periodic']
kernel_dict = dict(zip(kernel_names, kernels))

# Optimizer for minimization
optim = Scipy()
max_iter = 1000

# Initialize model comparison for one kernel
bnpqed = BnpQedModel(X, Y, rbf, D, label_func)
bnpqed.train()
bnpqed.predict(xx, n_samples = 4)
disc = bnpqed.discEstimate(b = b)
log_bayes = bnpqed.get_log_Bayes_factor()
posterior_probs = bnpqed.get_posterior_model_probabilities()
bnpqed.plot(xx, b=b)
plt.show()


# opts = dict()
# opts['max_iter'] = 1000  # restarts for parameter optimization
# opts['mode'] = 'BIC'  # method for approximating evidence
# opts['verbose'] = True  # switch for more output
# bnqd = BnpQedAnalysis(X, Y, kernel_dict, D, label_func, None, b, opts)
# bnqd.plot_model_fits(xx)




