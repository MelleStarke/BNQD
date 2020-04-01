# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:05:11 2019

@author: u341138
"""

import numpy as np   
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
from scipy.linalg import block_diag
#from scipy.stats import multivariate_normal

from typing import Optional, Tuple

from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.kernels import Kernel
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction

Data = Tuple[tf.Tensor, tf.Tensor]


print("GPflow version:      {}".format(gpflow.__version__))
print("TensorFlow version:  {}".format(tf.__version__))

class BNQD():
    
    pass



class DGPR(GPModel):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this models is sometimes referred to as the 'marginal log likelihood',
    and is given by

    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N\left(\mathbf y\,|\, 0, \mathbf K + \sigma_n \mathbf I\right)
    """

    def __init__(self, data: Data, kernel: Kernel, mean_function: Optional[MeanFunction] = None,
                 noise_variance: float = 1.0):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        #_, y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=1)
        self.data = data

    def log_likelihood(self):
        r"""
        todo

        """
        
        log_prob = 0
        for dataset in self.data:
            x, y = dataset
            K = self.kernel(x)            
            num_data = x.shape[0]
            k_diag = tf.linalg.diag_part(K)
            s_diag = tf.fill([num_data], self.likelihood.variance)
            ks = tf.linalg.set_diag(K, k_diag + s_diag)
            L = tf.linalg.cholesky(ks)
            m = self.mean_function(x)

            # [R,] log-likelihoods for each independent dimension of Y
            log_prob += multivariate_normal(y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(self, predict_at: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        todo
        
        """
        
        res = list()
        
        for i, dataset in enumerate(self.data):
        
            x_data, y_data = dataset
            err = y_data - self.mean_function(x_data)
    
            kmm = self.kernel(x_data)
            knn = self.kernel(predict_at[i], full_cov=full_cov)
            kmn = self.kernel(x_data, predict_at[i])
    
            num_data = x_data.shape[0]
            s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))
    
            conditional = gpflow.conditionals.base_conditional
            f_mean_zero, f_var = conditional(kmn, kmm + s, knn, err, full_cov=full_cov,
                                             white=False)  # [N, P], [N, P] or [P, N, N]
            f_mean = f_mean_zero + self.mean_function(predict_at[i])
            res.append((f_mean, f_var))
        return res

""" End of attempt
"""


#gpflow.reset_default_graph_and_session()

# see https://github.com/GPflow/GPflow/issues/600 for model silimar to BNQD!

# after training, anchor model to session! see https://gpflow.readthedocs.io/en/develop/notebooks/tips_and_tricks.html trick 5

# simulate data

# np.random.seed(1)

b = 0.0
n = 100
x = np.linspace(-3, 3, n)
f = 0.8*np.sin(x) + 0.2*x**2 + 0.2*np.cos(x/4) + 1.0*(x>b)
sigma = 0.3
y = np.random.normal(f, sigma, size=n)

plt.figure()

plt.plot(x[x<=b], f[x<=b], label='True f', c='k')
plt.plot(x[x>=b], f[x>=b], c='k')
plt.axvline(x=b, linestyle='--', c='k')
plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')

# k = gpflow.kernels.Matern52()
#k = gpflow.kernels.Exponential()
#k = gpflow.kernels.SquaredExponential()
k = gpflow.kernels.Linear() + gpflow.kernels.Constant()
#k = gpflow.kernels.Cosine() + gpflow.kernels.Constant()
Gaussian = gpflow.likelihoods.Gaussian()

m0 = gpflow.models.GPR(data=(x[:,None], y[:,None]), kernel=gpflow.utilities.deepcopy(k))

opt = gpflow.optimizers.Scipy()

def opt_fun_m0():
    return -m0.log_marginal_likelihood()

opt_m0_logs = opt.minimize(opt_fun_m0, m0.trainable_variables, options=dict(maxiter=100))

gpflow.utilities.print_summary(m0)

xx = np.linspace(-3, 3, 200)

m0_mean, m0_var = m0.predict_f(xx[:,None])

plt.plot(xx, m0_mean, c='green', label='$M_0$')
plt.fill_between(xx, m0_mean[:,0] - 1.96*np.sqrt(m0_var[:,0]), m0_mean[:,0] + 1.96*np.sqrt(m0_var[:,0]), color='green', alpha=0.2)

m1 = DGPR(data=((x[x<=b,None], y[x<=b,None]), (x[x>b, None], y[x>b,None])), kernel=k)


def opt_fun_m1():
    return -m1.log_likelihood()


opt_m1_logs = opt.minimize(opt_fun_m1, m1.trainable_variables, options=dict(maxiter=100))

gpflow.utilities.print_summary(m1)

xx_control = np.linspace(-3, 0, 100)
xx_intv = np.linspace(0, 3, 100)            

m1_c_pred, m1_i_pred = m1.predict_f((xx_control[:, None], xx_intv[:, None]))

    
for i, cond in enumerate([(xx_control, m1_c_pred), (xx_intv, m1_i_pred)]):    
    x_pred, pred = cond    
    mean, var = pred
    if i==0:
        plt.plot(x_pred, mean, color='b', label='$M_1$')
    else:
        plt.plot(x_pred, mean, color='b')
    plt.fill_between(x_pred, mean[:,0] + 1.96*np.sqrt(var[:,0]), mean[:,0] - 1.96*np.sqrt(var[:,0]), color='b', alpha=0.2)

plt.xlim([-3, 3])
plt.legend()


m0_log_marginal = m0.log_marginal_likelihood().numpy()
m1_log_marginal = m1.log_marginal_likelihood().numpy()


bf10 = m1_log_marginal - m0_log_marginal
plt.title('log BF = {:0.4f}'.format(bf10))


K0 = m0.kernel(x[:,None])
K1 = block_diag(m1.kernel(x[x<=b,None]), m1.kernel(x[x>b,None]))

f_K, axes_K = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
axes_K[0].imshow(K0)
axes_K[1].imshow(K1)
plt.show()
