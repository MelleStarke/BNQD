
###########################
###### Linear analysis ######
###########################
"""
Script used for testing the implementation.
Features some options to quickly manipulate the script.
"""
from bnqdflow import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow as gf
#plt.rcParams['figure.constrained_layout.use'] = True
from gpflow.kernels import Kernel, SquaredExponential, Constant, Linear, Periodic, Cosine, Exponential, Matern52,Matern32
SET_USE_CUSTOM_KERNEL_COPY_FUNCTION(True)
np.random.seed(20)

#############################
###### Testing Options ######
#############################

SHOW_TRAINING_DATA = 1
SHOW_UNDERLYING_FUNCTION = 0

# Whether or not the sub-models of the discontinuous model use the same hyper parameters
SHARE_PARAMS = 1

# Test for the full BNQDAnalysis object
TEST_LINEAR_ANALYSIS = 0
TEST_NONLINEAR_ANALYSIS = 1

# Method used for estimation of the marginal likelihood
MAR_LIK_METHOD = "nat"

# Function that worked in a previous version of GPflow to reset the cached TensorFlow graph
# gf.reset_default_graph_and_session()


#####################################
###### Test Dataset Parameters ######
#####################################

ip = 0.  # Intervention point
dc = 1.0  # Discontinuity
sigma = 0.5 # Standard deviation
sigma_d = 0.  # Value added to the standard deviation after the intervention point
n = 20  # Number of data points

############################
###### Kernel Options ######
############################

Matern = Matern32()
linear_kernel =  Linear() + Constant() # "Linear" kernel
exp_kernel = Exponential()
RBF_kernel = SquaredExponential()

kernel_names = ['Linear','Exponential','Gaussian',  'Matern','BMA']
kernels = [linear_kernel,exp_kernel,RBF_kernel,Matern]
# make a dictionary that zips the kernel names with the corresponding kernel
kernel_dict = dict(zip(kernel_names, kernels))  # making a dictionary of kernels and their corresponding names

###########################################
###### Generation of Test Dataset ######
###########################################
def get_predictors(n):
    #    return random_predictors(n)
    return even_predictors(n)

def even_predictors(n):
    return np.linspace(-3, 3, n)

def random_predictors(n, xmin=-3, xmax=3):
    return np.sort(np.random.uniform(low=xmin, high=xmax, size=n))

def linear_data_same_slope(n=100, slope=0.3, bias=0.0, ip=0.0, disc=4.0, noise_sd=1.0):
    x = get_predictors(n)
    f = bias + slope * x + disc * (x > ip)
    y = np.random.normal(loc=f, scale=noise_sd)
    return x, f, y

def periodic_data_disc(n=100, bias=1.2, period=6, ip=0.0, amp=2.0, disc=-1.0, noise_sd=0.5):
    x = get_predictors(n)
    f = bias + amp*np.sin((2*np.pi/period)*(x)) + disc*(x>ip)
    y = np.random.normal(loc=f, scale=noise_sd)
    return x,f, y

def split_data(x, x0):
    x_before_intervention = x[x < x0, None]
    x_after_intervention = x[x >= x0, None]
    return x_before_intervention, x_after_intervention


###################################
###### Pointer Testing Stuff ######
###################################

class Container(dict, tf.Module):
    def __init__(self):
        super().__init__()


###########################
###### Non linear analysis ######
###########################

num_sim = 1
# list of true discontinuity sizes
d_true_gaps = [0.5,1.0,2.0,4.0]

bf = np.zeros((int(n / 2), len(kernels) + 1, num_sim))
# bf = np.zeros((50, len(kernels) , num_sim))
lengthscaleM0 = np.zeros((int(n / 2), len(kernels) , num_sim))
lengthscaleM1 = np.zeros((int(n / 2), len(kernels) , num_sim))

data = linear_data_same_slope

container = Container()

if True:
    plt.figure(figsize = (20,10))
    fig, axs = plt.subplots(3, 4, figsize=(20, 10),gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    fig.suptitle('Bayesian decision metric with nonlinear data for different discontinuity and noise level',
                 fontsize = 'xx-large')
    for g, gap in enumerate(d_true_gaps):
        x, f,y = data(n=n, ip=0.0, disc=gap, noise_sd=sigma)
        axs[0,g].set_title("True function with gap: {} and noise {}".format(gap, sigma),fontsize = 'large')
        axs[0,g].plot(x, f, color='blue', linewidth=2, label='True function')
        axs[0,g].plot(x, y, 'kx', mew=1, label='Observations')
        axs[0,g].set_xlabel("x",fontsize = "large")
        axs[0,g].set_ylabel("y",fontsize = "large")
        axs[0,g].axvline(x=ip, linestyle='--', linewidth=1, color='black', label='Intervention point')



    for g, gap in enumerate(d_true_gaps): #different discontinuity sizes
        print("D-gap: ", gap)
        for j in range(0, num_sim): #for each discontinuity I want to average multiple simulations
            np.random.seed(j)
            x,f, y = data(n=n, ip=0.0, disc=gap, noise_sd=sigma)
            x_before_intervention, x_after_intervention = split_data(x, ip)  # split data


            print("Start of simulation: {}".format(j))
            for i in range(0, len(x_after_intervention)): #sequentially updating one datapoint
                print(f"N post-ip: {i}")

                # always increment x_before intervention by one element of x_after_intervention
                x_before_intervention = np.append(x_before_intervention, x_after_intervention[i])
                new_y = y[:len(x_before_intervention)]

                x_bi = x_before_intervention[x_before_intervention <= ip]
                y_bi = new_y[x_before_intervention <= ip]
                x_ai = x_before_intervention[x_before_intervention > ip]
                y_ai = new_y[x_before_intervention > ip]

                # make dataset
                d_c, d_i = (x_bi, y_bi), (x_ai, y_ai)
                d = [d_c, d_i]

                # make a model
                bnqd = analyses.BnpQedAnalysis_different_kernel(d, kernel_dict, ip, share_params=bool(SHARE_PARAMS),
                                                                marginal_likelihood_method='BIC')
                # train the model
                bnqd.train()

                for k, v in bnqd.results.items():
                    #v.plot_regressions()
                    pass

               # results = bnqd.pretty_print()
                results,results_lengthscales = bnqd.pretty_print()
                #print(results)
                #print(results_lengthscales)
                results_lengthscale_m0 = results_lengthscales['LengthscaleM0']
                results_lengthscale_m1 = results_lengthscales['LengthscaleM1']

                results = results['Log BF']

                # model comparison plot
                for k in range(len(kernels)):
                    bf[i][k][j] = results.iloc[k]
                    bf[i][k + 1][j] = bnqd.get_total_log_Bayes_factor()


                for k in range(len(kernels)):
                    lengthscaleM0[i][k][j] = results_lengthscale_m0.iloc[k]
                    lengthscaleM1[i][k][j] = results_lengthscale_m1.iloc[k]

                container.update({f"gap={gap}; sim={j}; N post-ip={i}": bnqd})

        print(f"n dict items: {len(container)}")
        gf.utilities.print_summary(container)

        colors = ['#44C5CB', '#FCE315', '#F53D52', '#FF9200', 'black']
        #colors = ['#FCE315', '#F53D52', '#FF9200', 'black']
        markers = ['o', 's', '^', 'x', 'v']

        lw = 2
        cs = 6

        for k in range(len(kernel_names)):

            axs[1,g].plot(np.arange(len(x_after_intervention)),
                               np.mean(bf[:, k, :], axis=1),
                               color=colors[k],
                               label=kernel_names[k])
            axs[1,g].fill_between(np.arange(len(x_after_intervention)),
                        np.mean(bf[:, k, :], axis=1)+0.5*np.std(bf[:, k, :], axis=1),
                        np.mean(bf[:, k, :], axis=1) - 0.5*np.std(bf[:, k, :], axis=1),
                        color=colors[k],
                        label=kernel_names[k], alpha = 0.2)

        axs[1,g].set_xlabel("Timepoint after intervention",fontsize = "large")
        axs[1,g].set_ylabel("Log Bayes Factor", fontsize = "large")
        axs[1, g].set_title("True discontinuity: {} and noise: {} ".format(gap, sigma),fontsize = 'large')

        handles, labels = axs[1,g].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=10)

        colors = ['#44C5CB', '#FCE315', '#F53D52', '#FF9200', 'black']
        #colors = [ '#FCE315', '#F53D52', '#FF9200', 'black']
        markers = ['o', 's', '^', 'x', 'v']

        lw = 2
        cs = 6
        #kernel_names = ['Exponential','Gaussian','Matern']

        for k in range(len(kernels)):
            #if k==1 or k == 2 or k == 3:
            if k>0:
                #axs[2, g].plot(np.arange(len(x_after_intervention)),
                #               np.mean(lengthscaleM0[:, k, :], axis=1),
                #               color=colors[k],
                #               label=kernel_names[k]+ str(' M0'))
                axs[2, g].plot(np.arange(len(x_after_intervention)),
                               np.mean(lengthscaleM1[:, k, :], axis=1),
                               color=colors[k],
                               label=kernel_names[k]+str(' M1'))

            axs[2, g].set_xlabel("Timepoint after intervention", fontsize='large')
            axs[2, g].set_ylabel("Lengthscale" , fontsize='large')
            #axs[2, g].set_title("True discontinuity: {} and noise: {} ".format(gap, sigma), fontsize='large')

        handles, labels = axs[1, g].get_legend_handles_labels()
        plt.figlegend(handles, labels, loc='lower center', ncol=10)

    plt.show()



