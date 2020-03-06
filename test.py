import BNQD
import gpflow as gpf
import numpy as np
import util
import matplotlib.pyplot as plt

xs, ys = util.linear_dummy_data()
cm = BNQD.ContinuousModel(xs, ys, gpf.kernels.Linear())
opt = gpf.optimizers.Scipy()
cm.train()
#print(cm.predict(np.array([[1.]])))
print(type(np.random.uniform(0, 10, size=(50,))))
print(type(np.random.uniform(0, 10, size=50)))