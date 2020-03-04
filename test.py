import BNQD
import gpflow as gpf
import util

xs, ys = util.linear_dummy_data()
cm = BNQD.ContinuousModel(xs, ys, gpf.kernels.Linear())
opt = gpf.optimizers.Scipy()
cm.train()
print(cm.predict(1))
