import numpy as np
import matplotlib.pyplot as plt
import gpflow as gf
import warnings
import bnqdflow.util as util
import tensorflow as tf
import math
import sys

from typing import Optional, Tuple, Union, List, Any, Callable

from numpy import ndarray

from tensorflow import Tensor

from tensorflow_probability import mcmc

from itertools import cycle, islice

from copy import copy, deepcopy

from gpflow import optimizers, Module, ci_utils
from gpflow.kernels import Kernel
from gpflow.models import (GPModel,
                           GPR,
                           BayesianModel,
                           GPMC,
                           SVGP,
                           SGPMC,
                           ExternalDataTrainingLossMixin,
                           InternalDataTrainingLossMixin
                           )
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.likelihoods import Gaussian, Likelihood

from tensorflow.python.data.ops.iterator_ops import OwnedIterator as DatasetOwnedIterator

##############################
###### Global Constants ######
##############################

N_MODELS = 2  # Nr. of sub-models in the discontinuous model
N_LOSS_SAMPLES = 20

class MiniBatchIterator:

    def __init__(self, data_list, batch_size: float):

        self.max_iter = math.ceil(1 / batch_size)
        self.n_iter = 0

        def make_iterator(data):
            N = len(data[0])
            size = int(batch_size * N)
            training_data = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(N)
            return iter(training_data.batch(size))
        self.iterators = list(map(make_iterator, data_list))

    def __iter__(self):
        return self

    def __next__(self):

        if self.n_iter >= self.max_iter:
            raise StopIteration

        def get_batch(iterator):
            next_batch = next(iterator)
            return next_batch

        self.n_iter += 1
        return list(map(get_batch, self.iterators))


class GPMContainer(BayesianModel):
    """
    Generic container for GPModel objects.

    Computes all statistics related to the data accordingly.
    (e.g. log likelihood is computed as the sum of the likelihood of each contained GPModel over its respective data)

    Every functioality related to prediction uses lists, where the index of each element corresponds to the index of
    the GPModel that is used.

    Due to the generic nature of this class, it can be adapted to support more than two GPModels and multiple
    intervention points.

    More user-friendly classes are defined below this class (i.e. ContinuousModel and DiscontinuousModel)
    """

    def __init__(
            self,
            regression_source: Union[Kernel, GPModel, List[GPModel]],
            data_list: Optional[List[RegressionData]] = None,
            likelihood: Optional[Likelihood] = Gaussian(),
            intervention_points: Optional[List[Tensor]] = None,
            share_params: Optional[bool] = True,
            gpm_type: Optional[str] = 'gpr',
            inducing_var_ratio: Optional[Union[Tuple[float], float]] = 0.01
    ):
        super().__init__()

        assert isinstance(regression_source, (Kernel, GPModel)) \
               or (isinstance(regression_source, list) and isinstance(regression_source[0], GPModel)), \
            "The regression_source objects needs to be an instance of either a Kernel or a GPModel, " \
            "or a list of GPModel objects."

        # Initializes the list of models
        if isinstance(regression_source, list):
            # Sets the list of models as the regression_source if it's a list of GPModel objects
            self.models = regression_source
        else:
            # Generates a list of GPModel objects if otherwise
            assert len(data_list) > 0, \
                "The list of RegressionData cannot be empty."

            self.models = self.generate_gp_models(regression_source, data_list=data_list, gpm_type=gpm_type,
                                                  likelihood=likelihood, inducing_var_ratio=inducing_var_ratio)

        # Sets the list of intervention points
        self.intervention_points = [] if intervention_points is None else intervention_points
        assert self.n_models == (len(self.intervention_points) + 1), \
            f"The number of GPModel objects contained in GPMContainer ({self.n_models}) should be one higher " \
            f"than the number of intervention points ({len(self.intervention_points)})."

        # Only applicable to GPMContainers with more than one model (i.e. discontinuous)
        if self.n_models > 1:
            # The names of the parameters that should be shared or kept separated.
            # These can be different depending on what GPModel implementation is used.
            # The following parameters are always used in GPModel implementations.
            applicable_params = ['kernel', 'likelihood', 'mean_function']

            if share_params:
                # Ensures all models use the same hyper parameter objects
                self._ensure_same_params(applicable_params)
            else:
                # Ensures all models use different hyper parameter objects
                self._ensure_different_params(applicable_params)

        else:
            # If the GPMContainer only contains one model, intialize the list of intervention points as empty.
            # It's better to do this here than by using a default value, because that would make the parameter mutable.
            self.intervention_points = []

        self.optimizer = None
        self.is_trained = False
        self.posterior_sampling_results = None
        self.posterior_gp_regression = None


    @staticmethod
    def generate_gp_models(
            model_or_kernel: Union[GPModel, Kernel],
            data_list: List[RegressionData],
            likelihood: Optional[Likelihood] = Gaussian(),
            gpm_type: Optional[str] = 'gpr',
            inducing_var_ratio: Optional[Union[Tuple[float], float]] = 0.01
    ):
        """
        Generates a list of GPModel objects with the same length as data_list.

        If a GPModel object was passed, the list will consist of deep copies of the GPModel, with the data reassigned.
        If a Kernel was passed, the list will consist of GPR (all containing the Kernel) instead.

        :param inducing_var_ratio: Ratio of inducing points to data points
        :param gpm_type: GPModel implementetion to be generated
        :param likelihood: Likelihood for the GPModels
        :param model_or_kernel: GPModel or Kernel object used to generate the list of models
        :param data_list: List of RegressionData. Each model will get one element.
        :return:
        """
        assert isinstance(model_or_kernel, (Kernel, GPModel)), \
            "The regression_source object needs to be an instance of either a Kernel or a GPModel, "
        assert all(map(lambda data: type(data) is tuple and len(data) is 2, data_list)), \
            "data_list should be a list of tuples of length 2 (i.e. a list of RegressionData)"

        is_kernel = isinstance(model_or_kernel, Kernel)
        # Ensures induving_vars_ratio is a tuple with the same length as the number of data partitions
        inducing_var_ratio = inducing_var_ratio if type(inducing_var_ratio) is tuple \
            else tuple(map(lambda _: inducing_var_ratio, range(len(data_list))))

        models = list()
        for i, data in enumerate(data_list):
            # Ensures both the InputData and OutputData are in a format usable by tensorflow
            data = tuple(map(util.ensure_tf_matrix, data))
            N = len(data[0])

            if is_kernel:
                if gpm_type.lower() in ['hmc', 'gpmc', 'mc', 'mcmc']:
                    models.append(GPMC(data,
                                       kernel=model_or_kernel,
                                       likelihood=likelihood))

                elif gpm_type.lower() in ['gpr']:
                    models.append(GPR(data,
                                      kernel=model_or_kernel))

                elif gpm_type.lower() in ['sparse', 'svgp']:
                    # Nr. of inducing variables
                    M = inducing_var_ratio[i] if type(inducing_var_ratio[i]) is int else int(inducing_var_ratio[i] * N)
                    # Randomly selected X data of length M
                    inducing_vars = data[0][np.random.choice(N, size=M, replace=False), :].copy()
                    models.append(SVGP(kernel=model_or_kernel,
                                       likelihood=likelihood,
                                       inducing_variable=inducing_vars,
                                       num_data=N))
                    # Ensures self.data_list works for GPMContainer. Lazy fix though
                    models[i].data = data
                    # Sets all parameters in the inducing variables as non-trainable. Important for training
                    gf.utilities.set_trainable(models[i].inducing_variable, False)

                elif gpm_type.lower() in ['sgpmc', 'sparse mc', 'sparse hmc']:
                    # Nr. of inducing variables
                    M = inducing_var_ratio[i] if type(inducing_var_ratio[i]) is int else int(inducing_var_ratio[i] * N)
                    # Randomly selected X data of length M
                    inducing_vars = data[0][np.random.choice(N, size=M, replace=False), :].copy()
                    models.append(SGPMC(data,
                                        kernel=model_or_kernel,
                                        likelihood=likelihood,
                                        inducing_variable=inducing_vars))
                    # Sets all parameters in the inducing variables as non-trainable. Important for training
                    gf.utilities.set_trainable(models[i].inducing_variable, False)

                else:
                    raise ValueError("{} is not a valid GPModel specification. See gpflow.models for the available "
                                     "options".format(gpm_type))

            else:
                # Appends a deepcopy of the passed GPModel to the list of models
                # Haven't tested this and might mess up the structure so prioritize the other options
                model = gf.utilities.deepcopy(model_or_kernel)
                model.data = data
                models.append(model)

        return models

    @property
    def n_models(self) -> int:
        """
        The number of models contained in the class.
        """
        return len(self.models)

    @property
    def is_continuous(self) -> bool:
        """
        Whether or not the GPMContainer only contains one GPModel. Which would make it continuous
        """
        return len(self.models) == 1

    @property
    def shared_params(self):
        """
        Whether or not the models share hyper parameters.
        """
        if self.n_models < 2:
            warnings.warn("The GPMContainer contains less then two models. Therefore, parameters cannot be shared "
                          "between models by definition. share_params will return True by default in this case.")
            return True

        for i in range(1, self.n_models):
            if self.models[i].kernel is not self.models[0].kernel \
                    or self.models[i].likelihood is not self.models[0].likelihood:
                return False

        return True

    @property
    def data_list(self) -> List[RegressionData]:
        """
        Collects all data objects of the contained models.
        """
        return list(map(lambda m: m.data, self.models))

    @property
    def N(self) -> int:
        """
        Number of total data points
        """
        return sum(map(lambda data: len(data[0]), self.data_list))

    @property
    def kernel(self):
        """
        Kernel object shared by all contained models
        """
        if self.n_models < 2 or self.shared_params:
            return self.models[0].kernel

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single kernel "
                      "object cannot be returned. You can call the kernel of s specific model via e.g. "
                      "container.models[0].kernel")
        return None

    @property
    def likelihood(self):
        """
        Likelihood object shared by all contained models
        """
        if self.n_models < 2 or self.shared_params:
            return self.models[0].likelihood

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single likelihood "
                      "object cannot be returned. You can call the likelihood of s specific model via e.g. "
                      "container.models[0].likelihood")
        return None

    @property
    def mean_function(self):
        """
        MeanFunction object shared by all contained models
        """
        if self.n_models < 2 or self.shared_params:
            return self.models[0].mean_function

        warnings.warn("The models contained in the GPMContainer don't share parameters. Therefore, a single "
                      "mean_function object cannot be returned. You can call the mean function of s specific model via e.g. "
                      "container.models[0].mean_function")
        return None
    
    @property
    def is_sparse(self):
        """
        Whether or not sparse models are contained
        """
        return all(map(lambda m: m.inducing_variable is not None, self.models))

    @property
    def inducing_variables(self):
        """
        List of all inducing variables
        """
        if all(map(lambda m: m.inducing_variable is not None, self.models)):
            return list(map(lambda m: m.inducing_variable, self.models))

        warnings.warn("The models contained in the GPMContainer aren't sparse variational models. Therefore, inducing "
                      "variables cannot be obtained")

        return None

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> Tensor:
        """
        Combined log likelihood of the contained models over their respective data.

        :return: Total log likelihood of the GPMContainer.
        """
        if isinstance(self.models[0], ExternalDataTrainingLossMixin):
            return tf.math.reduce_sum([m.maximum_log_likelihood_objective(*((data,) + args), **kwargs)
                                       for m, data in zip(self.models, self.data_list)])

        return tf.reduce_sum(list(map(lambda m: m.maximum_log_likelihood_objective(*args, **kwargs), self.models)), 0)

    def log_posterior_density(self, method=None, *args, **kwargs) -> Tensor:
        """
        Combined log marginal likelihood of the contained models over their respective data.
        This is done via one of two methods: using the BIC score, or with GPflow's native implementation.

        :param method: Method used for estimation of the log marginal likelihood.
        :return: Total log marginal likelihood of GPMContainer.
        """
        if method is None:
            if self.posterior_sampling_results is not None:
                method = 'hmc'

            elif self.is_sparse:
                method = 'elbo'

            else:
                method = 'bic'

        method = method.lower()

        if method in ["bic", "bic score", "bic_score"]:
            k = len(self.trainable_parameters)
            L = self.log_posterior_density(method='native', *args, **kwargs)
            BIC = L - k / 2 * np.log(self.N)
            return BIC

        elif method in ["native", "nat", "gpflow"]:
            return self.maximum_log_likelihood_objective(*args, **kwargs) + self.log_prior_density()

        elif method in ['mcmc', 'hmc', 'sampled']:
            return tf.math.log(tf.math.reduce_mean(np.exp(self.posterior_sampling_results[1])))

        elif method in ['svgp', 'sparse', 'elbo']:
            return self.mean_elbo(*args, **kwargs)

        else:
            raise ValueError(f"Incorrect method for log marginal likelihood calculation: {method}. "
                             "Please use either 'bic' or 'native' (i.e. gpflow method)")

    def mean_elbo(self, minibatch_ratio: Optional[float] = 0.02, plot_estimations=False):
        """
        Mean ELBO score over several mini batches
        :param minibatch_ratio: Ratio of mini match size to the number of data points
        :param plot_estimations: Whether or not to plot a histogram of all ELBO scores
        """
        assert all(map(lambda iv: iv is not None, self.inducing_variables)), \
            "The GP container doesn't contain sparse models, therefore the elbo cannot be calculated"

        elbo_fns = list(map(tf.function, map(lambda m: m.elbo, self.models)))
        elbo_means = list()

        for model, data, elbo_fn in zip(self.models, self.data_list, elbo_fns):
            batch_size = minibatch_ratio if type(minibatch_ratio) is int else int(minibatch_ratio * len(data[0]))
            train_data = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(len(data[0]))
            train_iter = iter(train_data.batch(batch_size))

            evals = [elbo_fn(batch) for batch in islice(train_iter, batch_size)]
            elbo_mean = tf.math.reduce_mean(evals)
            elbo_means.append(elbo_mean)

            if plot_estimations:
                ground_truth = elbo_fn(data).numpy()
                plt.figure()
                plt.hist(evals, label="Minibatch estimations")
                plt.axvline(ground_truth, c="k", label="Ground truth")
                plt.axvline(np.mean(evals), c="g", ls="--", label="Minibatch mean")
                plt.legend()
                plt.title(f"Histogram of ELBO evaluations using minibatches\nDiscrepancy: {ground_truth - elbo_mean}")
                plt.show()

        return tf.math.reduce_sum(elbo_means)

    def train(
            self,
            optimizer: object = None,
            scipy_method: Optional[str] = None,
            max_iter: Optional[int] = 1000,
            minibatch_ratio: Optional[float] = 0.02,
            loss_variance_goal: Optional[float] = None,
            verbose=True
    ) -> Optional[List[Tensor]]:
        """
        Trains all contained models.

        This is done by optimizing all trainable variables found in the GPMContainer, according to the combined
        training loss of all contained models.

        :param scipy_method: What method to use for the Scipy optimizer
        :param max_iter: Maximum nr. of training iterations
        :param minibatch_ratio: Ratio of mini batch size to the nr. of data points
        :param loss_variance_goal: Minimum variance over the past few loss values for the training to stop
        :param optimizer: Optimizer used for estimation of the optimal hyper parameters.
        :param verbose: Prints the model's summary if true.
        """
        is_sparse = self.is_sparse

        if optimizer is None:
            optimizer = tf.optimizers.Adam() if is_sparse else optimizers.Scipy()

        self.optimizer = optimizer
        losses = list()

        # Uses the optimizer to minimize the _training_loss function, by adjusting the trainable variables.
        if isinstance(optimizer, optimizers.Scipy):
            assert not is_sparse, "The SciPy optimizer cannot be used for sparse GPModels."

            kwargs = dict() if scipy_method is None else {'method': scipy_method}
            optimizer.minimize(self._training_loss, self.trainable_variables, **kwargs)

        elif isinstance(optimizer, tf.optimizers.Optimizer):
            args = tuple()

            for i in range(max_iter):
                optimizer.minimize(self._training_loss, self.trainable_variables)

                if not is_sparse or i % 10 == 0:
                    losses.append(self._training_loss())

                progress = math.ceil(i / max_iter * 100)

                if loss_variance_goal is not None and len(losses) >= N_LOSS_SAMPLES:
                    variance = tf.math.reduce_variance(losses[-N_LOSS_SAMPLES - 1:-1])
                    sys.stdout.write("\rprogress: %d%% | loss variance: %.4f" % (progress, variance))
                    
                    if variance <= loss_variance_goal:
                        break
                   
                else:
                    sys.stdout.write("\rprogress: %d%%" % progress)
                sys.stdout.flush()

        print()
        self.is_trained = True
        return losses

    def sample_posterior_params(self, n_samples=500, n_burnin_steps=300, n_leapfrog_steps=10, step_size=0.01,
                                n_adapt_step=10, accept_prob=0.75, adaptation_rate=0.1, trace_fn=None):
        """
        Samples the posterior distribution of the hyper-parameters via Hamiltonian Monte Carlo

        :param n_samples: Nr. of HMC samples
        :param n_burnin_steps: Nr. of burn-in steps
        :param n_leapfrog_steps: Nr. of leapfrog steps
        :param step_size: Step size of the HMC sampler
        :param n_adapt_step: Nr. of adaptation steps
        :param accept_prob: Acceptance probability
        :param adaptation_rate: Adaptation rate
        :param trace_fn: Function used to trace intermediate values. Results are returned at the end.
                         The arguments of the function are (1) the samples and (2) the previous kernel results
        :return: List of intermediate values of the trace function
        """
        incompatible_params = [name for name, item in gf.utilities.parameter_dict(self).items()
                               if item.trainable and item.prior is None]
        assert len(incompatible_params) is 0, \
            f"All trainable parameters must contain a prior in order to sample the posterior hyper-parameter " \
            f"distribution. These parameters don't have a prior defined: {incompatible_params}"

        if trace_fn is None:
            trace_fn = lambda *args: ()

        hmc_helper = optimizers.SamplingHelper(
            lambda: self.log_posterior_density('native'), self.trainable_parameters
        )

        hmc = mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=n_leapfrog_steps, step_size=step_size
        )
        adaptive_hmc = mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=n_adapt_step, target_accept_prob=gf.utilities.to_default_float(accept_prob),
            adaptation_rate=adaptation_rate
        )

        @tf.function
        def run_chain_fn():
            return mcmc.sample_chain(
                num_results=n_samples,
                num_burnin_steps=n_burnin_steps,
                current_state=hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=(lambda states, pkr:
                          (pkr.inner_results.proposed_results.target_log_prob, trace_fn(states, pkr))),
            )

        samples, (log_likelihoods, traces) = run_chain_fn()
        parameter_samples = hmc_helper.convert_to_constrained_values(samples)
        self.posterior_sampling_results = parameter_samples, log_likelihoods
        return traces

    def hmc_gp_regression(self, x_samples_list, mode='f'):
        """
        Returns the means and variances of the posterior GP regression at the data points in x_samples_list.
        Can either predict the latent function (mode='f') or the distribution over the data (mode='y')
        :param x_samples_list:
        :param mode:
        :return:
        """
        assert self.posterior_sampling_results is not None, \
            "Posterior hyper-parameter samples must be available to calculate the posterior gp regression"

        # Stores the optimized point estimates so they can be reapplied later
        old_param_vals = gf.utilities.parameter_dict(self)
        f_samples = list()
        # HMC samples of the posterior hyper-parameter distribution
        samples, _ = self.posterior_sampling_results

        for i in range(np.shape(samples)[1]):  # Loops over each HMC iteration
            for var, var_samples in zip(self.trainable_parameters, samples):
                # Sets the value of each parameter to its value at HMC iteration i
                var.assign(var_samples[i])

            # Gets some number of f samples according to the current parameter values (default 3)
            f = self.predict_f_samples(x_samples_list, 3)
            f_samples.append(f)

        # Removes singleton iterables
        f_samples = np.squeeze(f_samples)
        # Mean f values over all samples
        means = np.mean(f_samples, axis=(0, 2))
        # Variance of f over all samples
        vars = np.var(f_samples, axis=(0, 2))
        means_and_vars = list(zip(means, vars))

        # Predicts the means and variances over the outcome variable if predict_y is true
        if mode == 'y':
            means_and_vars = [m.likelihood.predict_mean_and_var(*mvs)
                              for mvs, m in zip(means_and_vars, self.models)]

        # Reassigns the optimized point estimates of the hyper-parameters
        gf.utilities.multiple_assign(self, old_param_vals)

        return means_and_vars

    def predict_f(self, Xnew_list: List[Union[InputData, ndarray]], full_cov=False, full_output_cov=False) \
            -> List[MeanAndVariance]:
        """
        Computes the means and variances of the posterior latent functions of the contained models at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce the predicted mean and variance of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the posterior latent functions.
        """
        assert len(Xnew_list) is len(self.models), \
            f"The number of elements in Xnew_list ({len(Xnew_list)}) should be the same as the number of sub-models " \
            f"({self.n_models}). Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_f(util.ensure_tf_matrix(section), full_cov, full_output_cov))

        return res

    def predict_f_samples(self, Xnew_list: List[Union[InputData, ndarray]], num_samples: Optional[int] = None,
                          full_cov: bool = True, full_output_cov: bool = False) -> List[Tensor]:
        """
        Produce a list of samples from the posterior latent function(s) at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce samples of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to draw samples.
        :param num_samples: Number of samples to draw.
        :param full_cov: If True, draw correlated samples over the inputs. If False, draw samples that are uncorrelated
                         over the inputs.
        :param full_output_cov: If True, draw correlated samples over the outputs. If False, draw samples that are
                                uncorrelated over the outputs.
        :return: List of samples.
        """
        assert len(Xnew_list) is len(self.models), \
            "The number of elements in Xnew_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_f_samples(util.ensure_tf_matrix(section), num_samples, full_cov,
                                                   full_output_cov))

        return res

    def predict_y(self, Xnew_list: List[Union[InputData, ndarray]], full_cov: bool = False,
                  full_output_cov: bool = False) -> List[MeanAndVariance]:
        """
        Compute the mean and variance of the held-out data at the input points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        Xnew_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, Xnew_list should look something like: [[], [0.2, 0.3, 0.4]].
        This will produce the predicted mean and variance of the 2nd model at input points 0.2, 0.3, and 0.4.

        :param Xnew_list: List of input locations at which to compute the means and variances.
        :param full_cov: whether or not to return the full covariance matrices of the latent functions.
        :param full_output_cov:
        :return: List of means and variances of the held-out data points.
        """
        assert len(Xnew_list) is len(self.models), \
            "The number of elements in Xnew_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."

        res = list()
        for model, section in zip(self.models, Xnew_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                res.append(model.predict_y(util.ensure_tf_matrix(section), full_cov, full_output_cov))

        return res

    def predict_log_density(self, data_list: List[RegressionData], full_cov: bool = False,
                            full_output_cov: bool = False):
        """
        Compute the log densities of the data at the new data points.

        Each element in the list of input points will be given to the corresponding GPModel. Therefore, the length of
        data_list should be the same as the number of contained models. If you wish to, for example, only use the 2nd
        model, and the GPMContainer contains two models, data_list should look something like:
        [[], ([0.2, 0.3, 0.4], [2.0, 2.0, 2.0])]. This will produce the predicted log density of the 2nd model at
        coordinates (0.2, 2.0), (0.3, 2.0), and (0.4, 2.0).

        :param data_list: List of RegressionData (i.e. tuples of shape (x, y)) for which to compute the log densities.
        :param full_cov:
        :param full_output_cov:
        :return: List of predicted log densities.
        """
        assert len(data_list) is len(self.models), \
            "The number of elements in data_list should be the same as the number of sub-models. " \
            "Each element in the list of input data is predicted by one model each."
        assert all(map(lambda data: len(data) is 0 or (len(data) is 2 and len(data[0]) is len(data[1])), data_list)), \
            "The list of data should consist of either empty lists (where you don't want predictions to be made), " \
            "or tuples of size 2, where both elements have the same length."

        res = list()
        for model, section in zip(self.models, data_list):
            if ((-1,) + np.shape(section))[-1] == 0:
                # Adds an empty element to the result list if the section is empty.
                # Necessary since gpflow doesn't allow prediction of empty InputData.
                res.append([])
            else:
                section = tuple(map(util.ensure_tf_matrix, section))
                res.append(model.predict_log_density(section, full_cov, full_output_cov))

        return res

    def plot_regression(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5, plot_data=True,
                        predict_y=False):
        """
        Plots the regressions of the models.

        :param padding: Proportion of the x-range that is added to the sides of the plot.
                        Can also be a tuple to allow for different paddings on the left and right.
        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        colours = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        col = colours[(self.n_models - 1) % len(colours)]
        markers = ['x', '+', '.', '*', 'd', 'v', 's', 'p', 'X', 'P', 'h']

        param_posterior_available = self.posterior_sampling_results is not None

        # Formats the margin as a tuple with duplicate elements.
        # Ensures the code works for a single float as well as a tuple of two floats.
        if type(padding) is not tuple:
            padding = (padding, padding)

        # Plots the vertical intervention point line(s).
        for ip in self.intervention_points:
            plt.axvline(ip, linestyle='--', c='k')

        if plot_data:
            # Plots the training data with different markers per section.
            for i, ((x, y), m) in enumerate(zip(self.data_list, cycle(markers))):
                plt.plot(x[:, 0], y[:, 0], linestyle='none', marker=m, color='k', label=f'$obs_{i}$')

        # Finds minimum and maximum x values.
        x_vals = tf.concat(list(map(lambda section: section[0], self.data_list)), 0)
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))
        x_range = max_x - min_x
        min_x, max_x = (min_x - (x_range * padding[0]), max_x + (x_range * padding[1]))
        x_range = max_x - min_x

        # List of intervention points plus the min and max x-value. Used to calculate the x-samples list.
        separations = [min_x] + self.intervention_points + [max_x]
        # List of x-samples used for plotting. Each x-sample is plotted using its respective model.
        x_samples_list = list()

        for i in range(len(separations) - 1):
            # Left-most x-value of the section
            left_bound = separations[i]
            # Right-most x-value of the section
            right_bound = separations[i + 1]

            section_ratio = (right_bound - left_bound) / x_range
            # Number of x-samples in the section.
            # Incremented by one to account for integer conversion and the overlapping of bounds between sections.
            section_samples = int(n_samples * section_ratio + 1)
            x_samples_list.append(np.linspace(left_bound, right_bound, section_samples))

        means_and_vars = None

        # Plots the regression according to the posterior distribution of the hyper-parameters
        if param_posterior_available:
            means_and_vars = self.hmc_gp_regression(x_samples_list, mode=('y' if predict_y else 'f'))

        # Plots the regression according to the optimized point estimates of the hyper-parameters
        else:
            # Which prediction function to use. Depends on the value of predict_y.
            predict = self.predict_y if predict_y else self.predict_f

            # Predicts the means and variances for both x_samples
            means_and_vars = np.squeeze(predict(x_samples_list))

        # Ensures only a single label occurs in the pyplot legend
        labeled = False

        for x_samples, (mean, var) in zip(x_samples_list, means_and_vars):
            # Plots the 95% confidence interval
            plt.fill_between(x_samples, mean - 1.96 * np.sqrt(var),
                             mean + 1.96 * np.sqrt(var), color=col, alpha=0.2)

            # Plots the mean function predicted by the GP
            plt.plot(x_samples, mean, c=col, label=('$M_D$' if not labeled else ""))
            labeled = True

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples_list = self.predict_f_samples(x_samples_list, num_f_samples)
            for f_samples, x_samples in zip(f_samples_list, x_samples_list):
                for f_sample in f_samples:
                    plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)

        # Plots the inducing locations if sparse models are contained
        if self.is_sparse:
            labeled = False
            for Z in self.inducing_variables:
                plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label=("Inducing locations" if not labeled else ""),
                         color='purple')
                labeled = True

    def plot_posterior_param_samples(self, mode="marginal", bins=20):
        """
        Plots the results of the HMC sampling process of the hyper-parameters. Either marginal or sequential

        :param mode: Plotting method, either marginal or sequential
        :param bins: Nr. of bins in the marginal histograms
        """
        param_to_name = {param: name for name, param in gf.utilities.parameter_dict(self).items()}
        params = self.trainable_parameters
        samples, _ = self.posterior_sampling_results
        compatible = [len(tf.shape(tf.squeeze(p.numpy()))) == 0 for p in params]
        params = [p for c, p in zip(compatible, params) if c]
        samples = [s for c, s in zip(compatible, samples) if c]

        if mode.lower() in ['sequential', 'sequentially', 'sequence', 'iterations', 'iter']:
            plt.figure()
            for val, param in zip(samples, params):
                plt.plot(tf.squeeze(val), label=param_to_name[param])
            plt.legend(bbox_to_anchor=(1.0, 1.0))
            plt.xlabel("HMC iteration")
            plt.ylabel("constrained parameter values")

        elif mode.lower() in ['marginal', 'histogram', 'hist', 'individual']:
            dim = math.ceil(np.sqrt(len(params)))
            fig, axes = plt.subplots(dim, dim)
            for ax, val, param in zip(axes.flatten(), samples, params):
                ax.hist(np.stack(val).flatten(), bins=20)
                ax.set_title(param_to_name[param])
                ax.axvline(param.numpy(), linestyle='--', c='k')
            fig.suptitle("constrained parameter samples")

        else:
            raise ValueError("{} is not a valid mode".format(mode))

    def _ensure_same_params(self, params: List[str]) -> None:
        """
        Sets all parameters of the models corresponding to the 'params' list to be the same object.

        Currently, the only options are 'kernel', 'likelihood', 'mean_function'

        :param params: List of strings. Each string corresponds to a parameter.
        """
        for p in params:
            p = p.lower()

            if p in ['k', 'kern', 'kernel']:
                for i in range(1, self.n_models):
                    self.models[i].kernel = self.models[0].kernel

            elif p in ['l', 'lik', 'likelihood']:
                for i in range(1, self.n_models):
                    self.models[i].likelihood = self.models[0].likelihood

            elif p in ['mf', 'mean', 'mean function', 'mean_function']:
                for i in range(1, self.n_models):
                    self.models[i].mean_function = self.models[0].mean_function

            else:
                warnings.warn(f"'{p}' is not a valid name of a parameter that can be shared.")

    def _ensure_different_params(self, params: List[str]) -> None:
        """
        Sets all parameters of the models corresponding to the 'params' list to be the different objects.

        Currently, the only options are 'kernel', 'likelihood', 'mean_function'

        :param params: List of strings. Each string corresponds to a parameter.
        """
        for p in params:
            p = p.lower()

            if p in ['k', 'kern', 'kernel']:
                for i in range(1, self.n_models):
                    if self.models[i].kernel is self.models[0].kernel:
                        self.models[i].kernel = gf.utilities.deepcopy(self.models[0].kernel)

            elif p in ['l', 'lik', 'likelihood']:
                for i in range(1, self.n_models):
                    if self.models[i].likelihood is self.models[0].likelihood:
                        self.models[i].likelihood = gf.utilities.deepcopy(self.models[0].likelihood)

            elif p in ['mf', 'mean', 'mean function', 'mean_function']:
                for i in range(1, self.n_models):
                    if self.models[i].mean_function is self.models[0].mean_function:
                        self.models[i].mean_function = gf.utilities.deepcopy(self.models[0].mean_function)

            else:
                warnings.warn(f"'{p}' is not a valid name of a parameter that can be shared.")


class ContinuousModel(GPMContainer):
    """
    Simplification of a GPMContainer with only one model.

    All inputs and outputs are adjusted accordingly.
    """

    def __init__(
            self,
            model_or_kernel: Union[GPModel, Kernel],
            data: Optional[RegressionData] = None
    ):
        """
        :param data: Data used for GP regression.
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be generated.
        :param mean_function: Mean function used for the regression.
        :param num_latent_gps: Number of latent Gaussian processes.
        """
        super().__init__(model_or_kernel, [data], intervention_points=[])

    @property
    def model(self):
        return self.models[0]

    @property
    def data(self):
        return self.data_list[0]

    def predict_f(self, Xnew: Union[InputData, ndarray], full_cov=False, full_output_cov=False) -> MeanAndVariance:
        return super().predict_f([Xnew], full_cov, full_output_cov)[0]

    def predict_f_samples(self, Xnew: Union[InputData, ndarray], num_samples: Optional[int] = None,
                          full_cov: bool = True, full_output_cov: bool = False) -> Tensor:
        return super().predict_f_samples([Xnew], num_samples, full_cov, full_output_cov)[0]

    def predict_y(self, Xnew: Union[InputData, ndarray], full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        return super().predict_y([Xnew], full_cov, full_output_cov)[0]

    def predict_log_density(self, data: RegressionData, full_cov: bool = False, full_output_cov: bool = False):
        return super().predict_log_density([data], full_cov, full_output_cov)[0]

    def plot_regression(self, n_samples=100, padding: Union[float, Tuple[float]] = 0.2, num_f_samples=5, plot_data=True,
                        predict_y=False):
        """
        Plots the regressions of the model.

        :param padding: Proportion of the x-range that is added to the sides of the plot.
                        Can also be a tuple to allow for different paddings on the left and right.
        :param plot_data: Plots the training data if true.
        :param n_samples: Number of x-samples used for the plot.
        :param num_f_samples: Number of samples of the latent function that are plotted.
        :param predict_y: Plots the prediction of new data points if true.
                          Plots the prediction of the latent function otherwise.
        """
        colours = ['green', 'blue', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        col = colours[(self.n_models - 1) % len(colours)]
        markers = ['x', '+', '.', '*', 'd', 'v', 's', 'p', 'X', 'P', 'h']

        # Formats the margin as a tuple with duplicate elements.
        # Ensures the code works for a single float as well as a tuple of two floats.
        if type(padding) is not tuple:
            padding = (padding, padding)

        if plot_data:
            x, y = self.data
            plt.plot(x[:, 0], y[:, 0], linestyle='none', marker=markers[0], color='k', label=f'$obs$')

        # Finds minimum and maximum x values.
        x_vals = self.data[0]
        min_x, max_x = (min(x_vals[:, 0]), max(x_vals[:, 0]))
        x_range = max_x - min_x
        min_x, max_x = (min_x - (x_range * padding[0]), max_x + (x_range * padding[1]))
        x_samples = np.linspace(min_x, max_x, n_samples)

        # Which prediction function to use. Depends on the value of predict_y.
        predict = self.predict_y if predict_y else self.predict_f

        # Predicts the means and variances for both x_samples
        mean, var = predict(x_samples)

        # Plots the 95% confidence interval
        plt.fill_between(x_samples, mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
                         mean[:, 0] + 1.96 * np.sqrt(var[:, 0]), color=col, alpha=0.2)

        # Plots the mean function predicted by the GP
        plt.plot(x_samples, mean[:, 0], c=col, label='$M_C$')

        if num_f_samples > 0 and not predict_y:
            # Plots samples of the latent function.
            # Only if num_f_samples > 0 and the latent function is plotted instead of
            # the prediction of held-out data points
            f_samples = self.predict_f_samples(x_samples, num_f_samples)
            for f_sample in f_samples:
                plt.plot(x_samples, f_sample[:, 0], linewidth=0.2, c=col)


class DiscontinuousModel(GPMContainer):
    """
    Simplification of a GPMContainer with two models.
    """

    def __init__(
            self,
            model_or_kernel: Union[GPModel, Kernel],
            data: Optional[List[RegressionData]],
            intervention_point: Tensor,
            share_params: Optional[bool] = True
    ):
        """
        :param data: List of data used for GP regression.
        :param model_or_kernel: Model or kernel object used for regression.
                                If a kernel is passed, a GPR object will be used.
        :param intervention_point: Input point at which to switch sub-models
        :param share_params: Whether or not the sub models have the same hyper parameters.
        """
        super().__init__(model_or_kernel, data, intervention_points=[intervention_point], share_params=share_params)

    @property
    def control_model(self):
        return self.models[0]

    @property
    def intervention_model(self):
        return self.models[1]

    def old_equal_params(self):
        """
        Old method of sharing the hyper parameters between the control and intervention model.
        """
        control_params = gf.utilities.parameter_dict(self.control_model)
        intervention_params = gf.utilities.parameter_dict(self.intervention_model)
        new_params = {k1: (v1 + v2) / 2 for (k1, v1), (k2, v2) in
                      zip(control_params.items(), intervention_params.items())}
        gf.utilities.multiple_assign(self.control_model, new_params)
        gf.utilities.multiple_assign(self.intervention_model, new_params)
