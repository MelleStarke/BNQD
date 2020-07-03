import numpy as np
import matplotlib.pyplot as plt
import bnqdflow as bf
from gpflow.kernels import RBF

ip = 0.0  # single intervention point
dc = 0.5
sigma = 0.2  # Standard deviation
n = 50  # Number of data points

x = np.linspace(-3, 3, n)  # Evenly distributed x values

f = lambda x: 0.8*np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 1) + dc * (x > ip)  # Underlying function
#f = lambda x: (0.8)*x + dc * (x > ip)
y = np.random.normal(f(x), sigma, size=n)  # y values as the underlying function + noise

# Data used by the control model (pre-intervention)
x1 = x[x <= ip]
y1 = y[x <= ip]

# Data used by the (post-)intervention model
x2 = x[x > ip]
y2 = y[x > ip]

data = [(x1, y1), (x2, y2)]


m = bf.analyses.SimpleAnalysis(RBF(), data, [ip])
m.train()
m.plot_regressions(separate=True, plot_data=False, num_f_samples=0)