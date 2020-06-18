"""
Script for messing around with the implementation, 3rd party libraries, and Python in general.
Not meant for exhaustive testing of the implementation.
"""

from __future__ import generators
# import BNQD
import gpflow as gf
import numpy as np
import tensorflow as tf
import time
import abc
import gpflow
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm
import random
import bnqdflow as bf

from typing import List, Tuple

from copy import copy, deepcopy

from bnqdflow.util import visitor, Step, copy_kernel, ensure_tensor
from bnqdflow.models import GPMContainer, ContinuousModel, DiscontinuousModel, MiniBatchIterator

from tensorflow_probability import distributions as dist

from tensorflow.keras.optimizers import Adam

import gpflow.mean_functions as mf
from gpflow.kernels import *
from gpflow.models import GPR

import random

from bnqdflow.effect_size_measures import Sharp
from bnqdflow.analyses import Analysis


class A:
    def __init__(self):
        self.type = 'A'

    def accept(self, b):
        b.visit(self)


class A1(A):
    pass


class A2(A):
    pass


class B(abc.ABC):
    def __init__(self):
        self.type = 'B'

    @visitor(A1)
    def visit(self, a):
        print("{} visited a1".format(self.__class__.__name__))

    @visitor(A2)
    def visit(self, a):
        print("{} visited a2".format(self.__class__.__name__))


class B1(B):

    @visitor(A1)
    def visit(self, a):
        print("{} visited a1".format(self.__class__.__name__))

    @visitor(A2)
    def visit(self, a):
        print("{} visited a2".format(self.__class__.__name__))


class B2(B):
    pass




#####################################
###### Test Dataset Parameters ######
#####################################

ip = 0  # Intervention point
ip2 = 2.3
dc = .5  # Discontinuity
sigma = .2  # Standard deviation
sigma_d = 0.  # Value added to the standard deviation after the intervention point
n = 100  # Number of data points


############################
###### Kernel Options ######
############################

#k = Constant() + Linear()  # "Linear" kernel
#k = Exponential()
k = SquaredExponential()
#k = Periodic(SquaredExponential())
#k = Cosine() + Constant()


###########################################
###### Generation of Test Dataset ######
###########################################

x = np.linspace(-1, 1, n)  # Evenly distributed x values

# Latent function options
#f = 0.8*np.sin(x) + 0.2*x**2 + 0.2 * np.cos(x / 1) + dc * (x > ip)  # Underlying function
#f = (0.8 - 1.6 * (x > ip))*x + dc * (x > ip)
f = np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14) + dc * (x > ip)
y = np.random.normal(f, sigma + sigma_d * (x > ip), size=n)  # y values as the underlying function + noise

# Uncomment to flip the data along the y-axis
#y = np.flip(y)

# Data used by the control model (pre-intervention)
xc = x[x <= ip]
yc = y[x <= ip]

xd = x[x > ip]
yd = y[x > ip]

# Data used by the (post-)intervention model
xi = xd[xd <= ip2]
yi = yd[xd <= ip2]

xe = xd[xd > ip2]
ye = yd[xd > ip2]


def plot_dist(dist, range = (-2, 2), n = 100):
    xs = np.linspace(*range, n)
    plt.plot(xs, np.exp(dist.log_prob(xs)))

mc = GPMContainer(k, [(xc, yc), (xd, yd)], intervention_points=[0], gpm_type='hmc')
mc.kernel.variance.prior = dist.Normal(tf.constant(0.5, dtype=tf.float64), tf.constant(3, dtype=tf.float64))
mc.kernel.lengthscales.prior = dist.Normal(tf.constant(1, dtype=tf.float64), tf.constant(3, dtype=tf.float64))
mc.likelihood.variance.prior = dist.Normal(tf.constant(0.5, dtype=tf.float64), tf.constant(3, dtype=tf.float64))


def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

from gpflow.ci_utils import ci_niter

"""rng = np.random.RandomState(123)
tf.random.set_seed(42)

n = 100

x = np.array(range(n))
y = x ** 2.

x1, y1 = x[x < int(n/2)], y[x < int(n/2)]
x2, y2 = x[x >= int(n/2)], y[x >= int(n/2)]

d1, d2 = ((x1), (y1)), ((x2), (y2))

data = [d1, d2]

print(f"data:\n{np.array(data)}")

it = MiniBatchIterator(data, 0.1)

for i, batch in enumerate(it):
    values = (list(map(lambda data: tuple(map(lambda v: v.numpy(), data)), batch)))
    print(f"\n\nbatch {i}:\n{values}")"""


losses = mc.train(optimizer=Adam(0.1), loss_variance_goal=0.1)
plt.plot(losses)
plt.title("MAP training losses")
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.show()
gf.utilities.print_summary(mc)
mc.sample_posterior_params()
mc.plot_posterior_param_samples('marginal')
plt.show()
mc.plot_posterior_param_samples('sequence')
plt.show()

plt.plot(mc.posterior_sampling_results[1])
plt.title("Likelihood values during HMC sampling")
plt.xlabel("Iteration")
plt.ylabel(r"$p(D|\theta)p(\theta)$")
plt.show()

print(f"Marginal likelihood estimates:\n\tnative: {mc.log_posterior_density('nat')}"
      f"\n\tBIC: {mc.log_posterior_density('bic')}\n\tHMC: ")

mc.plot_regression()
plt.show()



sys.exit()
mc.train()
mc.plot_regression()
plt.show()

mc.sample_posterior_params(n_samples=200, n_burnin_steps=100)
mc.plot_posterior_param_samples(mode='marginal')
plt.show()
#plt.plot(tf.squeeze(like_samples), label="log likelihood")
plt.show()
mc.plot_regression()
plt.show()
mc.plot_regression(predict_y=True)
plt.show()

"""
for i, m in enumerate(mc.models):
    for p in m.trainable_parameters:
        if p.prior is not None:
            plt.figure()
            plt.title(f'model {i}; {p.name}')
            plot_dist(p.prior)
            plt.show()"""

gf.utilities.print_summary(mc)

print("log marginal likelihoods:\n\tBIC: {}\n\tnative: {}\n\tHMC: {}"
      .format(*map(lambda m: mc.log_posterior_density(m), ['bic', 'nat', 'hmc'])))

"""ks[1].variance.prior = dist.Gamma(np.float64(20), np.float64(4.35))
m1 = None
for k in ks:
    m1 = GPMContainer(gf.utilities.deepcopy(k), [(x, y)], [])
    m2 = GPMContainer(gf.utilities.deepcopy(k), [(xc, yc), (xd, yd)], [ip])
    for name, m in zip(['c', 'd'], [m1, m2]):
        m.train()
        m.plot_regression()
        plt.show()
        print(f"{name} l: {m.log_posterior_density()}")

print(f"trainable parameters: {m1.trainable_parameters}")
print(f"log prior density: {m1.kernel.variance.log_prior_density()}")"""
