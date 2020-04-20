import bnqdflow as bf
import gpflow as gf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import reduce
from tensorflow_probability import distributions as dist

from gpflow.kernels import Constant, Linear, Cosine, Periodic, RBF, Exponential, Matern32

from scipy.stats import norm

bf.SET_USE_CUSTOM_KERNEL_COPY_FUNCTION(True)

ip = 0.  # Intervention point
dc = 1.  # Discontinuity
sigma = 0.5  # Standard deviation
sigma_d = 0.0  # Value added to the standard deviation after the intervention point

def linear_data_same_slope(n=100, slope=0.3, bias=0.0, ip=0.0, disc=4.0, noise_sd=1.0):
    x = np.linspace(-3, 3, n)
    f = bias + slope * x + disc * (x > ip)
    y = np.random.normal(loc=f, scale=noise_sd)
    return x, f, y

class Container(dict, tf.Module):
    def __init__(self):
        super().__init__()

res = {
    'ckv': list(),
    'ckl': list(),
    'clv': list(),
    'dkv': list(),
    'dkl': list(),
    'dlv': list()
}

kernels = [
    Linear() + Constant(),
    RBF(),
    Matern32(),
    Exponential()
]

SHOW_PLOTS = 1
epochs = 5

container = Container()

for e in range(epochs):

    print(f'epoch {e}')
    np.random.seed(e)
    '''
    n = 100  # Number of data points

    x = np.linspace(-3, 3, n)  # Evenly distributed x values

    # Latent function options
    f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function

    y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise
    '''

    x, f, y = linear_data_same_slope(n=100, slope = 3)

    # Data used by the control model (pre-intervention)
    xc = x[x <= ip]
    yc = y[x <= ip]

    # Data used by the (post-)intervention model
    xi = np.array([])
    yi = np.array([])

    for n_post_ip, (x_extra, y_extra) in enumerate(zip(x[x > ip], y[x > ip])):

        xi = np.append(xi, [x_extra])
        yi = np.append(yi, [y_extra])

        for k in kernels:

            a = bf.analyses.SimpleAnalysis([(xc, yc), (xi, yi)], k, ip)

            a.train(verbose=False)

            gf.utilities.print_summary(a)

            if SHOW_PLOTS:
                a.plot_regressions()
                plt.show()

            kn = k.kernels[0].__class__.__name__ if isinstance(k, gf.kernels.Combination) else k.__class__.__name__
            container.update({f"{kn}; N post-ip{n_post_ip}; ep={e}": a})


gf.utilities.print_summary(container)

'''
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
'''
