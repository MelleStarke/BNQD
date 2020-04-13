import bnqdflow as bf
import gpflow as gf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce

from gpflow.kernels import Constant, Linear, Cosine, Periodic, RBF, Exponential, Matern32

from bnqdflow.analyses import BnpQedAnalysis_different_kernel

from scipy.stats import norm

ip = 0.  # Intervention point
dc = 1.  # Discontinuity
sigma = 0.5  # Standard deviation
sigma_d = 0.0  # Value added to the standard deviation after the intervention point
n = 200  # Number of data points

kernels = [
    Constant(),
    Linear() + Constant(),
    RBF(),
    Matern32()
]

x = np.linspace(-3, 3, n)  # Evenly distributed x values

# Latent function options
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function

res = {
    'constant': list(),
    'linear': list(),
    'rbf': list(),
    'matern32': list()
}

SHOW_PLOTS = 0
epochs = 1

for e in range(epochs):

    print(f'epoch {e}')
    np.random.seed(e)
    y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise

    # Data used by the control model (pre-intervention)
    xc = x[x <= ip]
    yc = y[x <= ip]

    # Data used by the (post-)intervention model
    xi = np.array([])
    yi = np.array([])

    con = list()
    lin = list()
    rbf = list()
    mat = list()

    for x_extra, y_extra in zip(x[x > ip], y[x > ip]):
        xi = np.append(xi, x_extra)
        yi = np.append(yi, y_extra)

        for name, k in zip(['constant', 'linear', 'rbf', 'matern32'], kernels):

            n_data = len(xc) + len(xi)

            cm = bf.models.ContinuousModel(gf.utilities.deepcopy(k), (x[:n_data], y[:n_data]))
            dm = bf.models.DiscontinuousModel(gf.utilities.deepcopy(k), [(xc, yc), (xi, yi)], ip)

            a = bf.analyses.SimpleAnalysis([(xc, yc), (xi, yi)], (cm, dm), ip)

            a.train(optimizer=gf.optimizers.XiNat(), verbose=False)

            if SHOW_PLOTS:
                a.plot_regressions()
                plt.show()

            c_lml = a.continuous_model.log_posterior_density(method='bic')
            d_lml = a.discontinuous_model.log_posterior_density(method='bic')

            res['c_lml'].append(c_lml)
            res['d_lml'].append(d_lml)
            res['log_bf'].append(a.log_bayes_factor(method='bic'))
            res['es'].append(a.get_effect_size(bf.effect_size_measures.Sharp(n_samples=200, x_range=x_range)))

    res['constant'].append(con)
    res['linear'].append(lin)
    res['rbf'].append(rbf)
    res['matern32'].append(mat)

for k, v in res.items():
    if k == 'es':
        minx, maxx = x_range
        x_range = np.linspace(minx, maxx, 200)
        mean_bma = np.mean(list(map(lambda x: x['es_BMA'], v)), 0)
        std_bma = np.std(list(map(lambda x: x['es_BMA'], v)), 0)
        plt.figure(dpi=300)
        bf.util.plot_regression(x_range, mean_bma, std_bma, col='orange', alpha=0.4)
        for bma in map(lambda x: x['es_BMA'], v):
            plt.plot(x_range, bma, alpha=10/epochs, c='cyan')
        plt.show()
    else:
        print(f'{k}: mean={np.mean(v)} std={np.std(v)}')

        v = sorted(v)
        fit = norm.pdf(v, np.mean(v), np.std(v))
        plt.title(k)
        plt.plot(v, fit, marker='o')
        plt.hist(v, bins=20, density=1)
        plt.show()
