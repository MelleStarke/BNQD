from bnqdflow import *
import numpy as np
import matplotlib.pyplot as plt

from gpflow.kernels import Kernel, SquaredExponential, Constant, Linear, Periodic, Cosine

np.random.seed(1984)


#############################
###### Testing Options ######
#############################

SHOW_TRAINING_DATA = 1
SHOW_UNDERLYING_FUNCTION = 0

# Whether or not the sub-models of the discontinuous model use the same hyper parameters
SHARE_PARAMS = 1

# Tests for the continuous or discontinuous model individually
TEST_INDIVIDUAL_MODELS = 0
# Whether to use the continuous or discontinuous model for the individual test
TEST_INDIVIDUAL_CONTINUOUS_MODEL = 0

# Test for the full BNQDAnalysis object
TEST_ANALYSIS = 1

# Function that worked in a previous version of GPflow to reset the cached TensorFlow graph
#gf.reset_default_graph_and_session()


#####################################
###### Test Dataset Parameters ######
#####################################

ip = 0.  # Intervention point
dc = 1.  # Discontinuity
sigma = 0.1  # Standard deviation
n = 10  # Number of data points


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

x = np.linspace(-3, 3, n)  # Evenly distributed x values
f = 0.8 * np.sin(x) + 0.2 * x ** 2 + 0.2 * np.cos(x / 4) + dc * (x > ip)  # Underlying function
y = np.random.normal(f, sigma, size=n)  # y values as the underlying function + noise

# Data used by the control model (pre-intervention)
x_c = x[x <= ip]
y_c = y[x <= ip]

# Data used by the (post-)intervention model
x_i = x[x > ip]
y_i = y[x > ip]


plt.figure()
if SHOW_UNDERLYING_FUNCTION:
    plt.plot(x[x <= ip], f[x <= ip], label='True f', c='k')
    plt.plot(x[x >= ip], f[x >= ip], c='k')
plt.axvline(x=ip, linestyle='--', c='k')  # Vertical intervention point line
if SHOW_TRAINING_DATA:
    plt.plot(x, y, linestyle='none', marker='x', color='k', label='obs')


###########################
###### Testing Stuff ######
###########################

if TEST_INDIVIDUAL_MODELS:

    # Checks if the kernel is a combination of multiple kernels by trying to access the list of kernels
    # Doesn't really do anything, but may be helpful in the future
    try:
        k.kernels
        print("combo kernel")
    except AttributeError:
        print("regular kernel")

    m = None
    if TEST_INDIVIDUAL_CONTINUOUS_MODEL:
        m = models.ContinuousModel((x, y), k)
    else:
        m = models.DiscontinuousModel([(x_c, y_c), (x_i, y_i)], k, ip, share_params=bool(SHARE_PARAMS))

    m.train()
    # Plot the mean and variance of the model (default = 100 x-value samples)
    m.plot(100)

    if TEST_INDIVIDUAL_CONTINUOUS_MODEL:
        print("\ncontinuous model:\n\tBIC score: {}\n\tnative log marginal likelihood: {}"
              .format(m.log_marginal_likelihood("bic"), m.log_marginal_likelihood("native")))

    else:
        print("\ndiscontinuous model:\n\tBIC score: {}\n\tnative log marginal likelihood: {}"
              .format(m.log_marginal_likelihood("bic"), m.log_marginal_likelihood("native")))


if TEST_ANALYSIS:

    # Data used by the control model and the intervention model
    d_c, d_i = (x_c, y_c), (x_i, y_i)

    # Full data
    d = [d_c, d_i]
    a = analyses.SimpleAnalysis(d, k, ip, share_params=bool(SHARE_PARAMS), marginal_likelihood_method='BIC')

    a.train()
    a.plot()
    bf = a.log_bayes_factor(method='bic', verbose=True)
    e = a.get_effect_size(effect_size_measures.Sharp())
    plt.plot(e['es_BMA'])
    plt.plot(e['es_Disc'])
    plt.show()

plt.show()
