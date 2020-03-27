from __future__ import generators
# import BNQD
# import gpflow as gpf
import numpy as np
import tensorflow as tf
import time
import abc
import gpflow

from bnqdflow.util import visitor

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
b = 0.0
n = 100
x = np.linspace(-3, 3, n)
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + 1.0 * (x > b)
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


class B:
    def __init__(self):
        self.type = 'B'

    @abc.abstractmethod
    def visit(self, a):
        raise NotImplementedError


class B1(B):

    @visitor(A1)
    def visit(self, a):
        print("{} visited a1".format(self.__class__.__name__))

    @visitor(A2)
    def visit(self, a):
        print("{} visited a2".format(self.__class__.__name__))



a = A()
a1 = A1()
a2 = A2()
b = B1()

a1.accept(b)
a2.accept(b)
time.sleep(1)
a.accept(b)



'''
gpr = gpflow.models.GPR((x[:, None], y[:, None]), gpflow.kernels.SquaredExponential())
opt = gpflow.optimizers.Scipy()

def cl():
    return - gpr.log_likelihood()
    
opt.minimize(cl, gpr.trainable_variables)

print(gpr.predict_y(x_test))
'''

"""
Options this whole effect size distribution thingy:
Option 1:
    Just have different static functions for different effect size measures
    + No need for inheritance
    - """