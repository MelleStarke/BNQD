#import BNQD
#import gpflow as gpf
import numpy as np
import tensorflow as tf
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

xc = tf.Variable([1, 2, 3])
xi = tf.Variable([4, 5, 6])
yc = tf.Variable([10, 20, 30])
yi = tf.Variable([40, 50, 60])
dc = (xc, yc)
di = (xi, yi)

res = tf.Variable([[], []])
print(res)
for d in [dc, di]:
    res = tf.concat([res, d], 1)

sum = tf.reduce_sum([dc, di], 0)

print(res)
print(sum)


