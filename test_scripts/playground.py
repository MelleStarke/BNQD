from __future__ import generators
# import BNQD
# import gpflow as gpf
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
b = 10.
ip = 0.0
n = 100
dc = 8.0
x = np.linspace(-3, 3, n)
f = b + 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)
sigma = np.sqrt(1)
y = np.random.normal(f, sigma, size=n)

xc = tf.Variable([1, 2, 3])
xi = tf.Variable([4, 5, 6])
yc = tf.Variable([10, 20, 30])
yi = tf.Variable([40, 50, 60])
dc = (xc, yc)
di = (xi, yi)


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






k = RBF()# + gpflow.kernels.Constant()
mean = Step(mf.Linear(), mf.Linear())
gpr = gpflow.models.GPR((x[:, None], y[:, None]), k, mean_function=mean)
opt = gpflow.optimizers.Scipy()

def cl():
    return - gpr.log_likelihood()

print(tf.reshape(gpflow.base.Parameter([0]), (1, -1)))

gpflow.utilities.print_summary(gpr)

opt.minimize(cl, gpr.trainable_variables)

gpflow.utilities.print_summary(gpr)

xs = np.linspace(-12, 12, 200)[:, None]
ms, vs = gpr.predict_y(xs)

plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')

plt.plot(xs[:, 0], ms[:, 0], c='blue', label='$control_model$')
plt.fill_between(xs[:, 0], ms[:, 0] - 1.96 * np.sqrt(vs[:, 0]),
                     ms[:, 0] + 1.96 * np.sqrt(vs[:, 0]), color='blue', alpha=0.2)
plt.show()


"""
Options this whole effect size distribution thingy:
Option 1:
    Just have different static functions for different effect size measures
    + No need for inheritance
    - """