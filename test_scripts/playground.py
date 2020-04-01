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

from bnqdflow.util import visitor, Step

import gpflow.mean_functions as mf
from gpflow.kernels import RBF, Linear, Constant, Static, White
from gpflow.models import GPR

import bnqdflow

import random

from bnqdflow.effect_size_measures import Sharp
from bnqdflow.analyses import Analysis

'''
xs, ys = util.linear_dummy_data()
cm = BNQD.ContinuousModel(xs, ys, gpf.kernels.Linear())
opt = gpf.optimizers.Scipy()
cm.train()
#print(cm.predict(np.array([[1.]])))
print(type(np.random.uniform(0, 10, size=(50,))))
print(type(np.random.uniform(0, 10, size=50)))
'''


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




b = 0.
ip = 0.0
n = 300
dc = 0.0
x = np.linspace(-5, 5, n)
f = (b + 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc)
f2 = 0
sigma = 1
y = np.random.normal(f2, sigma, size=n) + b

xc = tf.Variable([1, 2, 3])
xi = tf.Variable([4, 5, 6])
yc = tf.Variable([10, 20, 30])
yi = tf.Variable([40, 50, 60])
dc = (xc, yc)
di = (xi, yi)

k1 = RBF()# + gpflow.kernels.Constant()
k2 = RBF()#Linear() + Constant()
m1 = GPR((x[:, None], y[:, None]), k1)
m2 = GPR((x[:, None], y[:, None]), k2)


class Container:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

def cl():
    return -m1.log_likelihood()


opt = gpflow.optimizers.Scipy()

opt.minimize(cl, m1.trainable_variables)

gf.utilities.print_summary(m1)

print(tf.reshape(gpflow.base.Parameter([0]), (1, -1)))

xs = np.linspace(-7, 7, 200)[:, None]

plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')

ms, vs = m1.predict_f(xs)

bnqdflow.util.plot_regression(xs[:, 0], ms[:, 0], vs[:, 0])

plt.show()

plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')

ms, vs = m1.predict_y(xs)

bnqdflow.util.plot_regression(xs[:, 0], ms[:, 0], vs[:, 0])

plt.show()
"""
Options this whole effect size distribution thingy:
Option 1:
    Just have different static functions for different effect size measures
    + No need for inheritance
    - """