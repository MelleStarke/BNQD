#import BNQD
#import gpflow as gpf
import numpy as np
import util
import matplotlib.pyplot as plt
#import tensorflow as tf
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
f = 0.8*np.sin(x) + 0.2*x**2 + 0.2*np.cos(x/4) + 1.0*(x>b)
sigma = np.sqrt(1)
y = np.random.normal(f, sigma, size=n)

print("x: {}, shape: {}\nx[:,None]: {}, shape: {}".format(x, x.shape, x[:,None], x[:,None].shape))


