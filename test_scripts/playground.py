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
f = (0.8 - 1.6 * (x > ip))*x + dc * (x > ip)
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

import numpy as np
import matplotlib.pyplot as plt
import gpflow as gf
import bnqdflow as bf
import sys
import tensorflow as tf

from gpflow import kernels as ks

from bnqdflow.analyses import MultiKernelAnalysis

summ = gf.utilities.print_summary

# %% md

## Dataset Generation

# %%

ip = 0.0  # single intervention point
dc = 0.5
sigma = 0.2  # Standard deviation
n = 100  # Number of data points

x = np.linspace(-3, 3, n)  # Evenly distributed x values

f = lambda x: 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 1) + dc * (x > ip)  # Underlying function
#f = lambda x: (0.8 - 1.6 * (x > ip))*x + dc * (x > ip)
y = np.random.normal(f(x), sigma, size=n)  # y values as the underlying function + noise

# Data used by the control model (pre-intervention)
x1 = x[x <= ip]
y1 = y[x <= ip]

# Data used by the (post-)intervention model
x2 = x[x > ip]
y2 = y[x > ip]

data = [(x1, y1), (x2, y2)]

plt.plot(x1, f(x1), c='blue')
plt.plot(x2, f(x2), c='orange')
plt.scatter(x1, y1, marker='+', c='blue', alpha=0.5)
plt.scatter(x2, y2, marker='x', c='orange', alpha=0.5)
plt.show()

# %% md

## Model Specification

# %%

kernels = [
    ks.Constant(),
    ks.Linear() + ks.Constant(),
    ks.Exponential(),
    ks.RBF(),
    ks.Periodic(ks.RationalQuadratic())
]


class SupTest(tf.Module):
    def __init__(self, data):
        self.cont_data = None
        self.cont_data = data


class Test(SupTest):
    def __init__(self, data):
        self.cont_m = None
        super().__init__(data)

    @property
    def cont_data(self):
        if self.cont_m is None:
            return self.__cont_data
        else:
            self.__cont_data = None
            return int(self.cont_m)

    @cont_data.setter
    def cont_data(self, data):
        if self.cont_m is None:
            self.__cont_data = data


test = Test(5)

print(test.cont_data)
test.cont_data = 7
print(test.cont_data)
test.cont_m = 5
print(test.cont_data)
test.cont_data = 7
print(test.cont_data)

a = MultiKernelAnalysis(
    kernels,
    data=data,
    intervention_point=ip,
    effect_size_measure=bf.effect_size_measures.Sharp(x_range=(-1, 3.5))
)

a.init_models()

# summ(a)

# %% md

## Model Training

# %%

a.train()
a.map(lambda a: a.plot_regressions(separate=True))
# %% md

## Effect Sizes

# %%

ess = a.get_effect_sizes()

fucking_kernel_names_jezus_fucking_christ_just_give_me_a_goddamned_list_of_kernel_names = deepcopy(a.kernel_names)

for sa, name in zip(a.analyses, fucking_kernel_names_jezus_fucking_christ_just_give_me_a_goddamned_list_of_kernel_names):
    sa._effect_size_measure.plot()
    plt.title(name)
    plt.legend()
    plt.show()

a.total_effect_size(bf.effect_size_measures.Sharp())
a.effect_size_measure.plot()
plt.title('total effect size')
plt.legend()
plt.show()

# %% md

## Metrics

# %%

for i, k in enumerate(fucking_kernel_names_jezus_fucking_christ_just_give_me_a_goddamned_list_of_kernel_names):
    print(f"kernel: {k}\n"
          f"\tlog bayes factor: {a.log_bayes_factors()[i]}\n"
          f"\texpected effect size: {a.get_reduced_effect_sizes()[i]}\n"
          f"\tnormalized kernel evidence: {a.effect_size_measure.effect_size['k_evds'][i]}\n"
          f"\tmodel evidences: {a.effect_size_measure.effect_size['m_evds'][i]}\n"
          f"\tindividual posterior model probs: {a.effect_size_measure.effect_size['m_probs'][i]}\n\n")

print(f"total log BF:\n{a.total_log_bayes_factor()}\n"
      f"total expected BMA effect size: {a.total_reduced_effect_size()}")



