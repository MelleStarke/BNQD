import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1984)


def linear_dummy_data(N=50, range=(0., 10.), a=3, b=0.7, noise=2, ip=5, dc=1.5):
    xs = np.random.uniform(range[0], range[1], size=N)
    ys = np.array([np.random.normal(a * x + b + (ip if x > ip else 0), noise) for x in xs])
    return xs, ys

''' if optimizer is None:
            optimizer = gpf.optimizers.Scipy()
        optimizer.minimize(lambda: - self.m.log_marginal_likelihood(), variables=self.m.trainable_variables)'''