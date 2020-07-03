import tensorflow as tf
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import Tuple

from abc import abstractmethod, ABC

from bnqdflow.analyses import Analysis, SimpleAnalysis, PlaceholderAnalysis, MultiKernelAnalysis
from bnqdflow.util import visitor


def normalize(array):
    return np.array(array) / np.sum(array)


class EffectSizeMeasure(ABC):
    """
    Abstract effect size measure class.
    Allows for the use of different effect size measures, while still being able to share methods and fields.
    """
    def __init__(self):
        self.effect_size = None

    @abstractmethod
    def calculate_effect_size(self, analysis) -> None:
        """
        Abstract effect size measure calculation method.
        This should be used as part of a visitor pattern by all implementations of the EffectSizeMeasure class.
        Implementation of the visitor pattern is done via the bnqdflow.util.visitor decorator.

        :param analysis: Analysis object used as a source for all information required to calculate the effect size.
                         Also used for the visitor patterns implementation.
        :return:
        """
        raise NotImplementedError


class Sharp(EffectSizeMeasure):
    """
    Sharp effect size measure object.
    Calculates the effect size while assuming there is a discrete separation between the data used by the sub-models of
    the discontinuous model.
    """
    def __init__(self,
                 n_samples: int = 300,
                 n_mc_samples: int = 500,
                 x_range: Tuple[float, float] = None,
                 lml_method = None):
        """
        :param n_samples: Number of x-samples used for the effect size distribution.
        :param n_mc_samples: Number of Monte Carlo samples for the BMA density estimate.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_mc_samples = n_mc_samples
        self.x_range = x_range
        self.lml_method = lml_method

    @visitor(Analysis)
    def calculate_effect_size(self, analysis):
        raise NotImplementedError("There doesn't exist an implementation for the sharp effect size measure for this "
                                  "analysis: {}".format(analysis.__class__.__name__))

    @visitor(SimpleAnalysis)
    def calculate_effect_size(self, analysis: SimpleAnalysis) -> None:
        """Computes the effect size at the boundary b. The BMA of the effect
        size is approximated using Monte Carlo sampling and Gaussian kernel
        density estimation.

        Note that this measure of effect size applies only to zeroth-order
        discontinuities, i.e. regression discontinuity.

        :return: Returns a dictionary containing
        - the effect size estimate by the discontinuous model as a pdf
        - the effect size estimate by the discontinuous model as summary
        statistics
        - The BMA effect size estimate.
        - The two-step p-value (i.e. frequentist p-value given the effect size
        distribution by the discontinuous model).
        - The range over which the effect size distribution is given, used for
        plotting.
        - The mean predictions at b.
        - The normalization from standardized effect size to the scale of the
        data.
        """
        ip = analysis.intervention_point

        # Means and variances of the two sub-models of the discontinuous model.
        (m0b, v0b), (m1b, v1b) = analysis.disc_m.predict_y([ip, ip])

        # Mean and standard dev differences. Used to calculate the discontinuous model's effect size estimate.
        disc_mean_diff = np.squeeze(m1b - m0b)  # TODO: why was this swapped around?
        disc_std_diff = tf.sqrt(tf.squeeze(v0b + v1b))

        if disc_mean_diff < 0:
            pval = 1 - stats.norm.cdf(x=0, loc=disc_mean_diff, scale=disc_std_diff)
        else:
            pval = stats.norm.cdf(x=0, loc=disc_mean_diff, scale=disc_std_diff)

        if self.x_range is None:
            print("x range is none")
            xmin, xmax = (np.min([disc_mean_diff - 4 * disc_std_diff, -0.1 * disc_std_diff]),
                          np.max([disc_mean_diff + 4 * disc_std_diff, 0.1 * disc_std_diff]))
        else:
            xmin, xmax = self.x_range

        xrange = np.linspace(xmin, xmax, self.n_samples)

        # Effect size estimate by the discontinuous model as a pdf
        y = stats.norm.pdf(xrange, disc_mean_diff, disc_std_diff)

        samples = np.zeros((self.n_mc_samples))
        nspike = int(np.round(analysis.posterior_model_probabilities()[0] * self.n_mc_samples))
        samples[nspike:] = np.random.normal(loc=disc_mean_diff,
                                            scale=disc_std_diff,
                                            size=(self.n_mc_samples - nspike))

        if not np.isscalar(ip):
            d_bma = None
        else:

            if nspike == self.n_mc_samples:
                # BMA dominated by continuous model
                # Put all mass at xrange closest to b
                d_bma = np.zeros((self.n_samples))
                xdelta = xrange[1] - xrange[0]
                ix = np.argmin((xrange - ip) ** 2)
                d_bma[ix] = 1.0 / xdelta
            elif nspike == 0:
                # BMA dominated by discontinuous model
                d_bma = y
            else:
                # BMA is a mixture
                kde_fit = stats.gaussian_kde(samples,
                                             bw_method='silverman')
                d_bma = kde_fit(xrange)

        self.effect_size = {
            # Estimated Bayesian model average
            'es_bma': d_bma,

            # Estimated effect size by the discontinuous model
            'es_disc': y,

            # Difference in mean and standard deviation of the two sub-models of the discontinuous model.
            'es_disc_stats': (disc_mean_diff, disc_std_diff),

            # Two-step p-value for the discontinuous model.
            'pval': pval,

            # Range over which the effect size distribution is given. Used for plotting
            'es_range': xrange,

            # Mean predictions at the intervention point by the sub-models of the discontinuous model.
            'f(b)': (m0b, m1b),

            # Normalization from the standard effect size to the scale of the data.
            'es_transform': lambda z: z * disc_std_diff + disc_mean_diff
        }

    @visitor(SimpleAnalysis)
    def calculate_reduced_effect_size(self, analysis):
        if self.effect_size is None:
            self.calculate_effect_size(analysis)

        xrange = self.effect_size['es_range']
        es_disc = self.effect_size['es_disc']
        es_bma = self.effect_size['es_bma']

        es_disc = np.sum(xrange * normalize(es_disc))
        es_bma = np.sum(xrange * normalize(es_bma))

        return {
            # Expected effect size value under the discontinuous model
            'es_disc': es_disc,
            # Expected effect size value according to the BMA
            'es_bma': es_bma
        }

    @visitor(MultiKernelAnalysis)
    def calculate_effect_size(self, analysis):
        es_list = analysis.get_effect_sizes(measure=None, force_recalc=True)

        assert np.all([es_list[0]['es_range'] == es['es_range'] for es in es_list]), \
            "All effect sizes must be over the same x range in order to obtain a distribution over " \
            "the total effect size"

        model_probs = np.array(analysis.map(lambda a: list(a.posterior_model_probabilities())))
        model_evidences = np.array([[np.exp(c.log_posterior_density(method=self.lml_method)) for c in containers]
                                    for containers in [a.containers for a in analysis.analyses]])
        kernel_evidences = normalize(np.sum(model_evidences, axis=1))

        print(f"kernel evidences shape: {np.shape(kernel_evidences)}")

        es_bma = np.sum([es['es_bma'] for es in es_list] * kernel_evidences[:, None], axis=0)

        print(f"expected bma es: {np.sum(normalize(es_bma) * es_list[0]['es_range'])}\n(shape={np.shape(es_bma)})")

        disc_model_evidences = normalize(model_evidences[:, 1])

        es_disc = np.sum([es['es_disc'] for es in es_list] * disc_model_evidences[:, None], axis=0)

        plt.title('differences')
        plt.plot(es_list[0]['es_range'], es_bma - es_disc)
        plt.show()

        print(f"expected disc es: {np.sum(normalize(es_bma) * es_list[0]['es_range'])}\n(shape={np.shape(es_disc)})")

        self.effect_size = {
            'k_probs': kernel_evidences,
            'm_evds': model_evidences,
            'm_probs': model_probs,
            'es_bma': es_bma,
            'es_disc': es_disc,
            'es_range': es_list[0]['es_range']
        }

    @visitor(MultiKernelAnalysis)
    def calculate_reduced_effect_size(self, analysis):
        if self.effect_size is not None:
            es_bma = self.effect_size['es_bma']
            es_disc = self.effect_size['es_disc']
            es_range = self.effect_size['es_range']

            return {
                'es_bma': np.sum(normalize(es_bma) * es_range),
                'es_disc': np.sum(normalize(es_disc) * es_range)
            }

        else:
            es_list = analysis.get_effect_sizes(measure=None, force_recalc=True)
            model_probs = np.array(analysis.map(lambda a: list(a.posterior_model_probabilities())))
            model_evidences = np.array([[np.exp(m.log_posterior_density(method=self.lml_method)) for m in c.models]
                                        for c in analysis.containers])
            kernel_evidences = np.sum(model_evidences * model_probs, axis=1)
            Z = np.sum(kernel_evidences)
            kernel_evidences = kernel_evidences / Z

            es_bma_list = np.array([es['es_bma'] for es in es_list])
            es_disc_list = np.array([es['es_disc'] for es in es_list])
            es_range_list = np.array([es['es_range'] for es in es_list])

            es_bma = np.sum([prob * es * evd for prob, es, evd
                             in zip(es_bma_list, es_range_list, kernel_evidences)])
            es_disc = np.sum([prob * es * evd for prob, es, evd
                              in zip(es_disc_list, es_range_list, model_evidences[:, 1])])

            return {
                'es_bma': es_bma,
                'es_disc': es_disc
            }

    @visitor(PlaceholderAnalysis)
    def calculate_effect_size(self, analysis: PlaceholderAnalysis) -> None:
        print("There doesn't exist an implementation for Sharp.calculate_effect_size() for {}"
              .format(analysis.__class__.__name__))
        return None

    def plot(self):
        plt.title("Effect size")
        x_range = self.effect_size['es_range']
        plt.fill_between(x_range, self.effect_size['es_bma'], label='BMA ES', alpha=0.5, color="#05668d")
        plt.fill_between(x_range, self.effect_size['es_disc'], label='Disc ES', alpha=0.5, color="#02c39a")



class FuzzyEffectSize(EffectSizeMeasure):

    def __init__(self):
        super().__init__()

    def calculate_effect_size(self, analysis) -> None:
        pass
