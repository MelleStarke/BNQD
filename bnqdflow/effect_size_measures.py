import tensorflow as tf
import numpy as np
import scipy.stats as stats

from abc import abstractmethod, ABC

from bnqdflow.analyses import Analysis, SimpleAnalysis, PlaceholderAnalysis
from bnqdflow.util import visitor


class EffectSizeMeasure(ABC):

    def __init__(self):
        self.effect_size = None

    @abstractmethod
    def calculate_effect_size(self, analysis) -> None:
        raise NotImplementedError


class Sharp(EffectSizeMeasure):

    def __init__(self, n_samples: int = 100, n_mc_samples: int = 500):
        super().__init__()
        self.n_samples = n_samples
        self.n_mc_samples = n_mc_samples

    @visitor(Analysis)
    def calculate_effect_size(self, analysis):
        raise NotImplementedError("There doesn't exist an implementation for the sharp effect size measure for this "
                                  "analysis: {}".format(analysis.__class__.__name__))

    @visitor(SimpleAnalysis)
    def calculate_effect_size(self, analysis: SimpleAnalysis) -> None:
        ip = analysis.intervention_point
        (m0b, v0b), (m1b, v1b) = analysis.discontinuous_model.predict_y([ip, ip])

        dict_mean_diff = np.squeeze(m1b - m0b)  # TODO: why was this swapped around?
        disc_std_diff = tf.sqrt(tf.squeeze(v0b + v1b))

        if dict_mean_diff < 0:
            pval = 1 - stats.norm.cdf(x=0, loc=dict_mean_diff, scale=disc_std_diff)
        else:
            pval = stats.norm.cdf(x=0, loc=dict_mean_diff, scale=disc_std_diff)

        xmin, xmax = (np.min([dict_mean_diff - 4 * disc_std_diff, -0.1 * disc_std_diff]),
                      np.max([dict_mean_diff + 4 * disc_std_diff, 0.1 * disc_std_diff]))

        n = 300
        xrange = np.linspace(xmin, xmax, n)
        y = stats.norm.pdf(xrange, dict_mean_diff, disc_std_diff)

        samples = np.zeros((self.n_mc_samples))
        nspike = int(np.round(analysis.posterior_model_probabilities()[0] * self.n_mc_samples))
        samples[nspike:] = np.random.normal(loc=dict_mean_diff,
                                            scale=disc_std_diff,
                                            size=(self.n_mc_samples - nspike))

        if not np.isscalar(ip):
            d_bma = None
        else:

            if nspike == self.n_mc_samples:
                # BMA dominated by continuous model
                # Put all mass at xrange closest to b
                d_bma = np.zeros((n))
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

        self.effect_size = {'es_BMA': d_bma,
                            'es_Disc': y,
                            'es_disc_stats': (dict_mean_diff, disc_std_diff),
                            'pval': pval,
                            'es_range': xrange,
                            'f(b)': (m0b, m1b),
                            'es_transform': lambda z: z * disc_std_diff + dict_mean_diff}

    @visitor(PlaceholderAnalysis)
    def calculate_effect_size(self, analysis: PlaceholderAnalysis) -> None:
        print("There doesn't exist an implementation for Sharp.calculate_effect_size() for {}"
              .format(analysis.__class__.__name__))
        return None


class FuzzyEffectSize(EffectSizeMeasure):

    def __init__(self):
        super().__init__()

    def calculate_effect_size(self, analysis) -> None:
        pass
