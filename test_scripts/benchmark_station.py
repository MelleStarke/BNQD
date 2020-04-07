import bnqdflow as bf
import gpflow as gf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import norm

ip = 0.  # Intervention point
dc = 1.0  # Discontinuity
sigma = 0.5  # Standard deviation
sigma_d = 0.5  # Value added to the standard deviation after the intervention point
n = 100  # Number of data points

x = np.linspace(-3, 3, n)  # Evenly distributed x values

# Latent function options
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function

res = {
    'd_c_kv': list(),
    'd_c_kl': list(),
    'd_c_lv': list(),
    'd_c_ll': list(),
    'd_c_lml': list(),
    'd_dc_kv': list(),
    'd_dc_kl': list(),
    'd_dc_lv': list(),
    'd_dc_ll': list(),
    'd_dc_lml': list(),
    'd_dc_lpd': list()
}
epochs = 10

for e in range(epochs):
    print(f'epoch {e}')
    np.random.seed(e)
    y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise

    # Data used by the control model (pre-intervention)
    xc = x[x <= ip]
    yc = y[x <= ip]

    # Data used by the (post-)intervention model
    xi = x[x > ip]
    yi = y[x > ip]

    cm1 = bf.models.ContinuousModel(gf.kernels.RBF(), (x, y))
    cm2 = bf.models.GPMContainer(gf.kernels.RBF(), [(x, y)])
    dm1 = bf.models.DiscontinuousModel([(xc, yc), (xi, yi)], gf.kernels.RBF(), ip)
    dm2 = bf.models.GPMContainer(gf.kernels.RBF(), [(xc, yc), (xi, yi)], [ip])

    models = [cm1, cm2, dm1, dm2]
    for i, m in enumerate(models):
        m.train(verbose=False)

    cm1.plot_regression(predict_y=True)
    cm2.plot_regression(predict_y=True)
    plt.show()
    dm1.plot_regression(predict_y=True)
    dm2.plot_regression(predict_y=True)
    plt.show()