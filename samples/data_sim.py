from third_party.Synthetic_PV_Profiles import CityPV_MultiModal
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os, sys, copy


X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR + '/experiments/inputs/'
PV_DIR = os.path.join(PROJECT_DIR, 'third_party', 'Synthetic_PV_Profiles')
sys.path.insert(1, PROJECT_DIR)
sys.path.insert(1, DATA_DIR)
sys.path.insert(1, PV_DIR)


class MetaDataset():

    def __init__(self, random_state=None):
        if random_state == None:
            self.random_state = np.random
        else:
            self.random_state = random_state
        self.x_mean, self.y_mean = None, None
        self.x_std, self.y_std = None, None


    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError


    def generate_meta_test_data(self, n_tasks:int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError


class PhysionetDataset(MetaDataset):

    def __init__(self, random_state=None, variable_id=0, dtype=np.float32, physionet_dir=None):
        super().__init__(random_state)
        self.dtype = dtype
        if physionet_dir is not None:
            self.data_dir = physionet_dir
        elif PHYSIONET_DIR is not None:
            self.data_dir = PHYSIONET_DIR
        else:
            raise ValueError("No data directory provided.")
        self.variable_list = ['GCS', 'Urine', 'HCT', 'BUN', 'Creatinine', 'DiasABP']

        assert variable_id < len(self.variable_list), "Unknown variable ID"
        self.variable = self.variable_list[variable_id] # the variable to work with, GCS or HCT

        self.data_path = os.path.join(self.data_dir, "set_a_merged.h5")

        with pd.HDFStore(self.data_path, mode='r') as hdf_file:
            self.keys = hdf_file.keys()                 # id of patients in set-a


    def generate_meta_train_data(self, n_tasks, n_samples=47):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. If there are less than n_tasks patients that have
        any measurements of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks <= 500, "We don't have that many tasks"
        assert n_samples < 48, "We don't have that many samples"

        meta_train_tuples = []

        for patient in self.keys:
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype) # times when var was measured
            values = df.values.astype(self.dtype)      # target var at measured times
            times_context = [time for time in times if time <= n_samples] # times for training: in the first n_samples and measured
            if len(times_context) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                values_context = values[:len(times_context)]              # target var at training times
                if values_context.shape[0] >= 4:                          # if at least 4 training samples
                    meta_train_tuples.append((times_context, values_context))
                else:
                    continue
            if len(meta_train_tuples) >= n_tasks:
                break

        return meta_train_tuples


    def generate_meta_test_data(self, n_tasks, n_samples_context=24,
                                n_samples_test=-1, variable_id=0):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. The remaining measurements are returned as test points,
        i.e., n_samples_test is unused.
        If there are less than n_tasks patients that have any measurements
        of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks <= 1000, "We don't have that many tasks"
        assert n_samples_context < 48, "We don't have that many samples"

        meta_test_tuples = []

        for patient in reversed(self.keys):
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype)
            values = df.values.astype(self.dtype)
            times_context = [time for time in times if time <= n_samples_context]
            times_test = [time for time in times if time > n_samples_context]
            if len(times_context) > 0 and len(times_test) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                times_test = np.array(times_test, dtype=self.dtype)
                values_context = values[:len(times_context)]
                values_test = values[len(times_context):]
                if values_context.shape[0] >= 4:
                    meta_test_tuples.append((times_context, values_context,
                                              times_test, values_test))
                else:
                    continue
            if len(meta_test_tuples) >= n_tasks:
                break

        return meta_test_tuples





class SinusoidDataset(MetaDataset):

    def __init__(self, amp_low=0.7, amp_high=1.3,
                 period_low=1.5, period_high=1.5,
                 x_shift_mean=0.0, x_shift_std=0.1,
                 y_shift_mean=5.0, y_shift_std=0.1,
                 slope_mean=0.5, slope_std=0.2,
                 noise_std=0.1, x_low=-5, x_high=5, random_state=None):

        super().__init__(random_state)
        assert y_shift_std >= 0 and noise_std >= 0, "std must be non-negative"
        self.amp_low, self.amp_high= amp_low, amp_high
        self.period_low, self.period_high = period_low, period_high
        self.y_shift_mean, self.y_shift_std = y_shift_mean, y_shift_std
        self.x_shift_mean, self.x_shift_std = x_shift_mean, x_shift_std
        self.slope_mean, self.slope_std = slope_mean, slope_std
        self.noise_std = noise_std
        self.x_low, self.x_high = x_low, x_high

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_test_tuples.append((X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_sinusoid(self):
        amplitude = self.random_state.uniform(self.amp_low, self.amp_high)
        x_shift = self.random_state.normal(loc=self.x_shift_mean, scale=self.x_shift_std)
        y_shift = self.random_state.normal(loc=self.y_shift_mean, scale=self.y_shift_std)
        slope = self.random_state.normal(loc=self.slope_mean, scale=self.slope_std)
        period = self.random_state.uniform(self.period_low, self.period_high)
        return lambda x: slope * x + amplitude * np.sin(period * (x - x_shift)) + y_shift


class SinusoidNonstationaryDataset(MetaDataset):

    def __init__(self, noise_std=0.0,  x_low=-5, x_high=5, random_state=None):

        super().__init__(random_state)
        self.noise_std = noise_std
        self.x_low, self.x_high = x_low, x_high

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_fun()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X)
            meta_test_tuples.append((X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_fun()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_fun(self):
        intersect = self.random_state.normal(loc=-2., scale=0.2)
        slope = self.random_state.normal(loc=1, scale=0.3)
        freq = lambda x: 1 + np.abs(x)
        mean = lambda x: intersect + slope * x
        return lambda x: mean(x) + np.sin(freq(x) * x) + self.random_state.normal(loc=0, scale=self.noise_std, size=x.shape)


class GPFunctionsDataset(MetaDataset):

    def __init__(self, noise_std=0.1, lengthscale=1.0, mean=0.0, x_low=-5, x_high=5, random_state=None):
        self.noise_std, self.lengthscale, self.mean = noise_std, lengthscale, mean
        self.x_low, self.x_high = x_low, x_high
        super().__init__(random_state)

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, self.lengthscale)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self.mean + np.dot(L, self.random_state.normal(size=(n, 1)))
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y


class CauchyDataset(MetaDataset):

    def __init__(self, noise_std=0.05, ndim_x=2, random_state=None):
        self.noise_std = noise_std
        self.ndim_x = ndim_x
        super().__init__(random_state)

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5, size=(n_samples, self.ndim_x),
                              random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5,
                              size=(n_samples_context + n_samples_test, self.ndim_x),
                              random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def _mean(self, x):
        loc1 = -1 * np.ones(x.shape[-1])
        loc2 = 2 * np.ones(x.shape[-1])
        cauchy1 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc1, axis=-1))**2))
        cauchy2 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc2, axis=-1))**2))
        return 6 * cauchy1 + 3 * cauchy2 + 1

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, 0.5)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self._mean(X) + np.dot(L, self.random_state.normal(scale=0.2, size=(n, 1))).flatten()
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y.reshape(-1, 1)




"""Normal"""


class GaussianDataset(MetaDataset):
    def __init__(self, mu_t, cov_t, x_dist, ndim_x=1, sigma_x=None, mu_x=None,
                 noise_std=0.0,  x_low=None, x_high=None, random_state=None, add_intercept=False):
        super().__init__(random_state)
        # check inputs
        assert x_dist in ["normal", "trunc_normal", "uniform"], "Unsupported input distribution.Supported options: normal, trunc_normal, uniform"
        if x_dist in ["trunc_normal", "uniform"] and ((x_low is None) or (x_high is None)):
            print("[WARNING] lower and upper bound not specified for " + x_dist + " input distribution.")
        if x_dist == "normal" and ((x_low is not None) or (x_high is not None)):
            print("[WARNING] lower and upper bound not considered for normal input distribution.")
        if x_dist == "uniform" and ((mu_x is not None) or (sigma_x is not None)):
            print("[WARNING] mean and std not considered for uniform input distribution.")

        # init
        self.mu_t = np.reshape(mu_t, mu_t.size)              # task environment mean
        self.cov_t = np.reshape(cov_t, (self.mu_t.size,
                                        self.mu_t.size))     # task environment std

        self.ndim_x = ndim_x                                 # number of features
        self.mu_x = np.reshape(mu_x, (ndim_x, 1))            # mean of features
        self.sigma_x = np.reshape(sigma_x, (ndim_x, ndim_x)) # std of features
        self.x_dist = x_dist                                 # distribution of features
        self.x_low, self.x_high = x_low, x_high              # range of features

        self.noise_std = noise_std
        self.add_intercept = add_intercept

        self.true_data_dist = []                             # true data distributions
        self.nrmz_data_dist = []                             # normalized data distributions

    def generate_clients_data(self, n_tasks, n_samples_obs, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        size = (n_samples_obs + n_samples_test, self.ndim_x)
        for i in range(n_tasks):
            f = self._sample_fun()
            # input
            if self.x_dist == "uniform":
                X = self.random_state.uniform(self.x_low, self.x_high, size=size)
            if self.x_dist == "trunc_normal":
                X = truncnorm.rvs((self.x_low - self.mu_x) / self.sigma_x,
                                  (self.x_high - self.mu_x) / self.sigma_x,
                                  loc=self.mu_x, scale=self.sigma_x,
                                  size=size, random_state=self.random_state)
            if self.x_dist == "normal":
                X = self.random_state.normal(loc=self.mu_x, scale=self.sigma_x, size=size)
            # add intercept as the first column
            if self.add_intercept:
                X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
            # noise free output
            Y = f(X)
            # add noise
            if self.noise_std > 0:
                noise = self.random_state.normal(loc=0, scale=self.noise_std, size=Y.size)
                noise = np.reshape(noise, Y.shape)
                Y = Y+noise
            meta_test_tuples.append((X[:n_samples_obs], Y[:n_samples_obs],
                                     X[n_samples_obs:], Y[n_samples_obs:]))
        return meta_test_tuples


    def _sample_fun(self):
        # TODO: if add_intercept, mu_t and cov_t must be updated
        w_star = self.random_state.multivariate_normal(mean=self.mu_t.flatten(),
                                                       cov=self.cov_t, size=1)
        self.true_data_dist.append(w_star)
        return lambda x: np.matmul(x, w_star)

# --------------------------------------------
class GaussianDatasetMix(GaussianDataset):
    def __init__(self, mu_t, cov_t, x_dist, ndim_x=1, sigma_x=None, mu_x=None,
                 noise_std=0.0,  x_low=None, x_high=None, random_state=None, add_intercept=False):

        super().__init__(mu_t=mu_t[0], cov_t=cov_t[0], x_dist=x_dist,
                         ndim_x=ndim_x, sigma_x=sigma_x, mu_x=mu_x,
                         noise_std=noise_std,  x_low=x_low, x_high=x_high,
                         random_state=random_state, add_intercept=add_intercept)


    def generate_clients_data(self, num_clients, n_samples_obs, n_samples_test, weight_modes=[1]):
        # check if a list is given for weight modes
        try:
           _ = (e for e in weight_modes)
        except TypeError:
           weight_modes = [weight_modes]
        # number of modes, clients per mode
        num_modes = len(weight_modes)
        num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]

        # generate data from each mode
        for mode_num in np.arange(num_modes):
            task_environment = GaussianDataset(mu_t=self.mu_t[mode_num], cov_t=self.cov_t[mode_num], x_dist=self.x_dist,
                                               ndim_x=self.ndim_x, sigma_x=self.sigma_x, mu_x=self.mu_x,
                                               noise_std=self.noise_std, x_low=self.x_low, x_high=self.x_high,
                                               random_state=self.random_state, add_intercept=self.add_intercept)

            data_temp = task_environment.generate_clients_data(n_tasks=num_clients_per_mode[mode_num],
                                                                    n_samples_obs=n_samples_obs,
                                                                    n_samples_test=n_samples_test)
            # concatenate modes
            if mode_num == 0:
                clients_data = data_temp
                self.true_data_dist = task_environment.true_data_dist
                self.nrmz_data_dist = task_environment.nrmz_data_dist
            else:
                clients_data=[*clients_data, *data_temp]
                self.true_data_dist=[*self.true_data_dist, *task_environment.true_data_dist]
                self.nrmz_data_dist=[*self.nrmz_data_dist, *task_environment.nrmz_data_dist]
            # record true distributions
        return clients_data


    def _sample_fun(self):
        raise NotImplementedError


# -----------------------------------------------------

"""Polynomial"""


class PolyDataset(GaussianDataset):
    def __init__(self, mu_t, cov_t, x_dist, poly_degree=1, sigma_x=None, mu_x=None,
                 noise_std=0.0,  x_low=None, x_high=None, random_state=None, add_intercept=False):
        assert poly_degree > 0
        self.poly_degree = poly_degree

        super().__init__(mu_t=mu_t, cov_t=cov_t, x_dist=x_dist,
                         ndim_x=1, sigma_x=sigma_x, mu_x=mu_x,
                         noise_std=noise_std,  x_low=x_low, x_high=x_high,
                         random_state=random_state, add_intercept=add_intercept)
        # first col is scalar => mu_x and sigma_x are scalar
        self.sigma_x=self.sigma_x[0,0]
        self.mu_x = self.mu_x[0,0]



    def generate_clients_data(self, n_tasks, n_samples_obs, n_samples_test):
        assert n_samples_test > 0
        clients_data_tuple = []
        size_col1 = n_samples_obs + n_samples_test

        for _ in range(n_tasks):
            f = self._sample_fun()
            # input
            X = np.empty(shape=(n_samples_obs + n_samples_test, self.poly_degree))
            if self.x_dist == "uniform":
                X[:, 0] = self.random_state.uniform(self.x_low, self.x_high,
                                                    size=size_col1)
            if self.x_dist == "trunc_normal":
                X[:, 0] = truncnorm.rvs((self.x_low - self.mu_x) / self.sigma_x,
                                  (self.x_high - self.mu_x) / self.sigma_x,
                                  loc=self.mu_x, scale=self.sigma_x,
                                  size=size_col1,
                                  random_state=self.random_state)
            if self.x_dist == "normal":
                X[:, 0] = self.random_state.normal(loc=self.mu_x, scale=self.sigma_x,
                                                   size=size_col1)


            # add other degrees
            for degree in np.arange(1, self.poly_degree):
                X[:, degree] = X[:, degree-1] * X[:, 0]

            # add intercept as the first column
            if self.add_intercept:
                X = np.append(np.ones(size_col1), X, axis = 0)

            # noise free output
            Y = f(X)

            # add noise
            if self.noise_std > 0:
                noise = self.random_state.normal(loc=0, scale=self.noise_std, size=Y.size)
                noise = np.reshape(noise, Y.shape)
                Y = Y+noise
            clients_data_tuple.append((X[:n_samples_obs], Y[:n_samples_obs],
                                       X[n_samples_obs:], Y[n_samples_obs:]))
        return clients_data_tuple


    def _sample_fun(self):
        w_star = self.random_state.multivariate_normal(mean=self.mu_t.flatten(),
                                                       cov=self.cov_t, size=1)
        w_star = np.reshape(w_star, (-1, 1))
        self.true_data_dist.append(w_star)
        return lambda x: np.matmul(x, w_star)

# --------------------------------------------
class PolyDatasetMix(PolyDataset):
    def __init__(self, mu_t, cov_t, x_dist, poly_degree=1, sigma_x=None, mu_x=None,
                 noise_std=0.0,  x_low=None, x_high=None, random_state=None, add_intercept=False):

        super().__init__(mu_t=mu_t[0], cov_t=cov_t[0], x_dist=x_dist,
                         poly_degree=poly_degree, sigma_x=sigma_x, mu_x=mu_x,
                         noise_std=noise_std,  x_low=x_low, x_high=x_high,
                         random_state=random_state, add_intercept=add_intercept)
        self.mu_t = mu_t
        self.cov_t = cov_t
        assert len(mu_t) == len(cov_t)
        self.num_modes = len(mu_t)


    def generate_clients_data(self, num_clients, n_samples_obs, n_samples_test, weight_modes=[1]):
        # check if a list is given for weight modes
        try:
           _ = (e for e in weight_modes)
        except TypeError:
           weight_modes = [weight_modes]
        # number of modes, clients per mode
        num_modes = len(weight_modes)
        num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]

        # generate data from each mode
        for mode_num in np.arange(num_modes):
            task_environment = PolyDataset(mu_t=self.mu_t[mode_num], cov_t=self.cov_t[mode_num],
                                           x_dist=self.x_dist, poly_degree=self.poly_degree,
                                           sigma_x=self.sigma_x, mu_x=self.mu_x,
                                           noise_std=self.noise_std, x_low=self.x_low, x_high=self.x_high,
                                           random_state=self.random_state, add_intercept=self.add_intercept)

            data_temp = task_environment.generate_clients_data(n_tasks=num_clients_per_mode[mode_num],
                                                               n_samples_obs=n_samples_obs,
                                                               n_samples_test=n_samples_test)
            # concatenate modes
            if mode_num == 0:
                clients_data = data_temp
                self.true_data_dist = task_environment.true_data_dist
                self.nrmz_data_dist = task_environment.nrmz_data_dist
            else:
                clients_data=[*clients_data, *data_temp]
                self.true_data_dist=[*self.true_data_dist, *task_environment.true_data_dist]
                self.nrmz_data_dist=[*self.nrmz_data_dist, *task_environment.nrmz_data_dist]
            # record true distributions
        return clients_data

    def _sample_fun(self):
        raise NotImplementedError

    def get_true_fun(self, X, client_num):
        w_star = self.true_data_dist[client_num]
        X_aug = np.empty(shape=(X.size, self.poly_degree))
        X_aug[:, 0] = X
        # add other degrees
        for degree in np.arange(1, self.poly_degree):
            X_aug[:, degree] = X_aug[:, degree-1] * X_aug[:, 0]
        # add intercept as the first column
        if self.add_intercept:
            X_aug = np.append(np.ones(X.size), X_aug, axis = 0)
        return np.matmul(X_aug, w_star).flatten()




# --------------------------------------------
class NonlinearPolyDatasetMix(PolyDatasetMix):
    def __init__(self, mu_t, cov_t, x_dist, poly_degree=1, sigma_x=None, mu_x=None,
                 noise_std=0.0,  x_low=None, x_high=None, random_state=None, add_intercept=False):

        super().__init__(mu_t=mu_t, cov_t=cov_t, x_dist=x_dist,
                         poly_degree=poly_degree, sigma_x=sigma_x, mu_x=mu_x,
                         noise_std=noise_std,  x_low=x_low, x_high=x_high,
                         random_state=random_state, add_intercept=add_intercept)


    def generate_clients_data(self, num_clients, n_samples_obs, n_samples_test, weight_modes=[1]):
        clients_data = super().generate_clients_data(num_clients=num_clients,
                                                     n_samples_obs=n_samples_obs,
                                                     n_samples_test=n_samples_test,
                                                     weight_modes=weight_modes)
        for n in np.arange(len(clients_data)):
            x_obs, y_obs, x_tru, y_tru = clients_data[n]
            col = 1 if self.add_intercept else 0
            clients_data[n] = (x_obs[:,col], y_obs, x_tru[:,col], y_tru)
        return clients_data

    def _sample_fun(self):
        raise NotImplementedError

# --------------------------------------------
class PVDataset(MetaDataset):
    def __init__(self, env_options, random_state=None):
        super().__init__(random_state)

        # check provided options
        req_fields = [# required for building the task env
                'city_names', 'tilt_std', 'az_std', 'weather_dev',
                'irrad_std' ,'altitude_dev', 'shadow_peak_red',
                # required for simulating pv
                'module_name', 'inverter_name'
                # required for generating data
                'train_scenarios']
        optional_fields = [# optional for building the task env
                'tilt_mean', 'az_mean',
                # optional for simulating pv data
                'lags', 'hours', 'months',
                'num_clients', 'num_clients_per_mode',
                'use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad',
                # optional for generating datasets
                'remove_constant_cols'
                ]
        #TODO: train_scenarios must have m_train, train_years, and optionally 'exclude_last_year'
        assert [x in env_options.keys() for x in req_fields]
        assert [x in req_fields + optional_fields for x in env_options.keys()]
        assert 'num_clients' in env_options.keys() or 'num_clients_per_mode' in env_options.keys()

        # parse optional options
        for key in optional_fields:
            if not key in env_options.keys():
                if key in ['use_station_irrad_direct', 'use_station_irrad_diffuse', 'delay_irrad', 'remove_constant_cols']:
                    env_options[key] = True
                else:
                    env_options[key] = None
        if env_options['num_clients_per_mode'] is not None:
            if env_options['num_clients'] is None:
                env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])
            else:
                assert env_options['num_clients'] == np.sum(env_options['num_clients_per_mode'])
        if env_options['num_clients'] is not None and env_options['num_clients_per_mode'] is None:
            env_options['num_clients_per_mode'] = [int(env_options['num_clients']/len(env_options['city_names']))]*len(env_options['city_names'])
            env_options['num_clients'] = np.sum(env_options['num_clients_per_mode'])

        # create task environment
        self.task_environment = CityPV_MultiModal(city_names=env_options['city_names'],
                tilt_mean=env_options['tilt_mean'], az_mean=env_options['az_mean'],
                tilt_std=env_options['tilt_std'], az_std=env_options['az_std'],
                weather_dev=env_options['weather_dev'], irrad_std=env_options['irrad_std'],
                altitude_dev=env_options['altitude_dev'], shadow_peak_red=env_options['shadow_peak_red'],
                random_state=self.random_state)

        # summarize info
        env_options['info'] = '{:2.0f} households at '.format(env_options['num_clients'])+ " ".join(env_options['city_names']) + ' - '
        for key in ['tilt_std', 'az_std', 'weather_dev', 'irrad_std', 'altitude_dev', 'shadow_peak_red']:
            if not isinstance(env_options[key], list): # TODO
                env_options['info'] += key+': {:.1f}, '.format(env_options[key])
        for name_str in ['module_name', 'inverter_name']:
            if len(env_options[name_str])==1:
                env_options['info'] += 'same ' + name_str + ', '
            elif len(env_options[name_str])==env_options['num_clients']:
                env_options['info'] += 'different ' + name_str + ', '
            else:
                env_options['info'] += + name_str + 'not specified, '

        self.env_dict = env_options
        self._simulate_pv()


    def _simulate_pv(self):
        self.task_environment.simulate_pv(num_clients_per_mode=self.env_dict['num_clients_per_mode'],
                                          module_name=self.env_dict['module_name'], inverter_name=self.env_dict['inverter_name'],
                                          lags=self.env_dict['lags'], months=self.env_dict['months'],
                                          hours=self.env_dict['hours'],
                                          use_station_irrad_direct=self.env_dict['use_station_irrad_direct'],
                                          use_station_irrad_diffuse=self.env_dict['use_station_irrad_diffuse'],
                                          delay_irrad=self.env_dict['delay_irrad'])
        # properties of task env that might have changed
        self.env_dict['months'] = self.task_environment.months
        self.env_dict['hours'] = self.task_environment.hours
        self.env_dict['lags'] = self.task_environment.lags

        # new properties
        self.env_dict['clients_config']=self.task_environment.clients_config


    def generate_clients_data(self):
        self.env_dict['feature_names'] = None
        for scenario_name, scenario in self.env_dict['train_scenarios'].items():
            self.task_environment.construct_regression_matrices(
                            m_train=scenario['m_train'], train_years=scenario['train_years'],
                            exclude_last_year=scenario['exclude_last_year'],
                            remove_constant_cols=self.env_dict['remove_constant_cols'])
            self.env_dict['train_scenarios'][scenario_name]['clients_data'] = copy.deepcopy(self.task_environment.clients_data_tuple)
            self.env_dict['train_scenarios'][scenario_name]['time_series'] = copy.deepcopy(self.task_environment.clients_time_series)
            if self.env_dict['feature_names'] is None:
                self.env_dict['feature_names'] = self.task_environment.feature_names
            else:
                assert self.env_dict['feature_names'] == self.task_environment.feature_names
        return self.env_dict

'''
def set_lags(env_dict, lags, random_state):

    # function to change lags used for extracting auto-regressors, while
    # maintaining the same set of houses.
    # generate_clients_data must be called afterwards.

    # create task environment
    task_environment = CityPV_MultiModal(city_names=env_dict['city_names'],
            tilt_mean=env_dict['tilt_mean'], az_mean=env_dict['az_mean'],
            tilt_std=env_dict['tilt_std'], az_std=env_dict['az_std'],
            weather_dev=env_dict['weather_dev'], irrad_std=env_dict['irrad_std'],
            altitude_dev=env_dict['altitude_dev'], shadow_peak_red=env_dict['shadow_peak_red'],
            random_state=random_state)

    self.task_environment.set_lags(lags)
'''


def remove_feature(env_dict, feature_name, in_place=False):
    '''
    removes a specified feature from task environemnt.
    inplace operation
    '''
    if not in_place:
        env_dict = copy.deepcopy(env_dict)
    assert isinstance(feature_name, str)
    assert feature_name in env_dict['feature_names']
    # find index of features except this feature
    feature_index_to_keep = np.delete(
                                np.arange(len(env_dict['feature_names'])),
                                env_dict['feature_names'].index(feature_name)
                                )
    # remove from feature_names
    env_dict['feature_names'].remove(feature_name)
    # no need to remove from time_series
    # remove from X_train and X_test at each scenario and for all clients
    for client_num in np.arange(env_dict['num_clients']):
        for scenario in env_dict['train_scenarios']:
            x_train, y_train, x_valid, y_valid = env_dict['train_scenarios'][scenario]['clients_data'][client_num]
            x_train = x_train[:, feature_index_to_keep]
            x_valid = x_valid[:, feature_index_to_keep]
            env_dict['train_scenarios'][scenario]['clients_data'][client_num] = (x_train, y_train, x_valid, y_valid)
            assert x_train.shape[1] == len(env_dict['feature_names'])
            assert x_valid.shape[1] == len(env_dict['feature_names'])
    if in_place:
        return
    else:
        return env_dict






'''
    def _generate_data_2y(self, env_dict):
        shuffle=False
        clients_data_2y = [None]*env_dict['num_clients']
        if env_dict['train_frac_2y'] is not None:
            for client_num in np.arange(env_dict['num_clients']):
                x_train, y_train, x_valid, y_valid = env_dict['clients_data_full'][client_num]
                x_all = np.concatenate((x_train, x_valid), axis=0)
                y_all = np.concatenate((y_train, y_valid), axis=0)
                if shuffle:
                    train_inds = self.random_state.choice(np.arange(x_all.shape[0]),
                                                    size=int(x_all.shape[0]*env_dict['train_frac_2y']), replace=False)
                else: # use the last portion for validation
                    train_inds = np.arange(int(x_all.shape[0]*env_dict['train_frac_2y']))
                valid_inds = list(set(np.arange(x_all.shape[0])) - set(train_inds))
                # divide again and put back
                clients_data_2y[client_num] = (x_all[train_inds, :], y_all[train_inds, :],
                                            x_all[valid_inds, :], y_all[valid_inds, :])
        env_dict['clients_data_2y']=clients_data_2y
        return env_dict


    def _generate_data_red(self, env_dict):
        # --- data reduced ----
        clients_data_red = [None] * env_dict['num_clients']
        if env_dict['m_train_red'] is not None and env_dict['m_valid_red'] is not None:
            for client_num in np.arange(env_dict['num_clients']):
                data = env_dict['clients_data_full'][client_num]
                # randomly select train_inds and valid_inds
                train_inds = self.random_state.choice(np.arange(data[0].shape[0]),
                                                size=env_dict['m_train_red'], replace=False)
                valid_inds = self.random_state.choice(np.arange(data[3].shape[0]),
                                                size=env_dict['m_valid_red'], replace=False)
                # new train and validation data
                clients_data_red[client_num] = (data[0][train_inds, :], data[1][train_inds],
                                            data[2][valid_inds, :], data[3][valid_inds])
                # mark samples used for train or validation
                is_test_2018 = [True]*len(env_dict['clients_ts_2018'][client_num].index) # mark all as test
                for i in train_inds:
                    is_test_2018[i] = False            # train points are not test
                is_test_2019 = [True]*len(env_dict['clients_ts_2019'][client_num].index) # mark all as test
                for i in valid_inds:
                    is_test_2019[i] = False            # valid points are not test
                env_dict['clients_ts_2018'][client_num]['is_test'] = is_test_2018
                env_dict['clients_ts_2019'][client_num]['is_test'] = is_test_2019
        env_dict['clients_data_red']=clients_data_red
        return env_dict
'''
# -----------------------------------------------------
# ---------------------- ICU --------------------------
# -----------------------------------------------------
class ICUDataset(MetaDataset):
    def __init__(self, target_var='GCS', random_state=None, data_dir=None):
        super().__init__(random_state)
        self.data_dir = os.path.realpath(os.path.dirname(__file__))+'/inputs/ICU/set-a' if data_dir is None else data_dir
        self.all_files_names = os.listdir(self.data_dir)
        if '.ipynb_checkpoints' in self.all_files_names: self.all_files_names.remove('.ipynb_checkpoints')
        self.target_var = target_var


    def generate_clients_data(self, num_clients, min_n_obs=1, max_n_obs=1e3,
                              min_n_test=1):
        # check inputs
        assert 0 < min_n_obs <= max_n_obs
        assert min_n_test > 0

        # init
        file_ind = 0
        client_num = 0
        data_clients = [None]*num_clients

        while client_num < num_clients:
            # read data file
            data = pd.read_csv(self.data_dir + '/'+ self.all_files_names[file_ind], sep=",", header=0)

            # only select rows measuring the target var
            data = data[data['Parameter']==self.target_var].drop('Parameter', axis=1).dropna().reset_index()

            # first day is train, the rest is test
            train_rows = [i for i, t in enumerate(data['Time']) if int(t.split(':')[0]) < 24]

            if len(train_rows) >= min_n_obs:
                # discard additional samples
                train_rows = train_rows[0: min(len(train_rows), max_n_obs)]

                # use the rest for test
                test_rows = np.arange(train_rows[-1]+1, len(data['Time']))

                # check number of test samples
                if len(test_rows) >= min_n_test:
                    train_x = np.array([int(t.split(':')[0])*60 + int(t.split(':')[1]) for t in data.iloc[train_rows]['Time']])
                    test_x  = np.array([int(t.split(':')[0])*60 + int(t.split(':')[1]) for t in data.iloc[test_rows]['Time']])
                    train_y = data.iloc[train_rows]['Value'].to_numpy()
                    test_y  = data.iloc[test_rows]['Value'].to_numpy()
                    if train_x.std() > 1e-3 and test_x.std() > 1e-3 :
                        data_clients[client_num] = (train_x, train_y, test_x, test_y)
                        client_num += 1
            # search next file
            file_ind += 1

        return data_clients


# -----------------------------------------------------
# -------------------- VEHICLE ------------------------
# -----------------------------------------------------
import json
class VehicleDataset(MetaDataset):
    def __init__(self, random_state=None, data_dir=None):
        super().__init__(random_state)
        self.data_dir = os.path.realpath(os.path.dirname(__file__))+'/inputs/Vehicle_Sensors' if data_dir is None else data_dir

        # read data
        client_ids, train_data, test_data = self.read_data()

        # check train and test exist for all clients
        self.num_clients = len(client_ids)

        # concatenate files to get all data available to each client
        self.clients_all_data = [None] * self.num_clients # tuple of all x and all y

        for client_num, client_id in enumerate(client_ids):
            x_train = np.array(train_data[client_id]['x'])
            y_train = np.array(train_data[client_id]['y'], dtype=np.double)
            x_test  = np.array(test_data[client_id]['x'])
            y_test  = np.array(test_data[client_id]['y'], dtype=np.double)

            # check all clients have the same number of features
            if client_num==0:
                self.num_features = x_train.shape[1]
            else:
                assert self.num_features == x_train.shape[1]

            # check number of samples
            assert x_train.shape[0] == y_train.shape[0]
            assert x_test.shape[0]  ==  y_test.shape[0]

            # all inputs and outputs available to this client
            x = np.concatenate((x_train, x_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)

            # shuffle
            xy = np.concatenate((x, y), axis=1)
            self.random_state.shuffle(xy)

            # form data tuples
            self.clients_all_data[client_num] = (xy[:, 0:-1], xy[:, -1].reshape(-1,1))

        # shuffle clients order
        self.random_state.shuffle(self.clients_all_data)



    def read_data(self):
        '''parses data in given train and test data directories

        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        clients = []
        train_data = {}
        test_data = {}

        train_files = os.listdir(self.data_dir+'/train')
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(self.data_dir+'/train',f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            train_data.update(cdata['user_data'])


        test_files = os.listdir(self.data_dir+'/test')
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(self.data_dir+'/test',f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

        clients = list(sorted(train_data.keys()))

        return clients, train_data, test_data



    def generate_clients_data(self, num_clients=None, train_percentage = 0.75,
                              min_n_obs = None,  max_n_obs=None, min_n_test=None, max_n_test=None):
        '''
        generate data for the vehicle dataset
        - num_clients: if None, looks for as many clients as possible
        '''
        # check inputs
        min_n_obs = 1    if min_n_obs is None else min_n_obs
        max_n_obs = 2e3  if max_n_obs is None else max_n_obs
        min_n_test = 1   if min_n_test is None else min_n_test
        max_n_test = 2e3 if max_n_test is None else max_n_test
        num_clients = 100 if num_clients is None else num_clients
        assert 0 < min_n_obs  <= max_n_obs
        assert 0 < min_n_test <= max_n_test

        # init
        candid_ind = 0
        client_num = 0
        data_clients = [None]*num_clients

        while client_num < num_clients:
            # pick a candid client
            data = self.clients_all_data[candid_ind]
            n_samples_total = np.shape(data[0])[0]
            n_samples_train = int(min(n_samples_total * train_percentage, max_n_obs))
            n_samples_test  = int(min(n_samples_total-n_samples_train, max_n_test))

            if (n_samples_train >= min_n_obs) and (n_samples_test >= min_n_test):
                data_clients[client_num] = (data[0][0:n_samples_train, :],
                                            data[1][0:n_samples_train, :],
                                            data[0][n_samples_train:n_samples_train+n_samples_test, :],
                                            data[1][n_samples_train:n_samples_train+n_samples_test, :])
                # successfully found a client
                client_num += 1
            # search next file
            if candid_ind < len(self.clients_all_data)-1:
                candid_ind += 1
            else: # no more files
                self.num_clients = client_num
                data_clients = data_clients[0:client_num]
                break

        print('[INFO] found {:2.0f} clients'.format(self.num_clients))
        return data_clients






# -----------------------------------------------------

if __name__ == "__main__":

    random_state = 3

    # geographical characteristics of the location
    latitude=46.520
    longitude=6.632
    city_name='Lausanne'
    altitude=496
    timezone='Etc/GMT-1'

    # clients distribution
    num_modes = 1
    weight_modes = [1/num_modes] * num_modes
    mean_tilt  = latitude
    mean_azimuth = 180
    sigma_tilt = 15
    sigma_azimuth  = 45
    mu_t = [[mean_tilt, mean_azimuth]]
    cov_t = [np.diag([sigma_tilt**2, sigma_azimuth **2])]

    # FL info
    num_clients = 25
    num_clients_per_mode = [int(weight_modes[i]*num_clients) for i in np.arange(num_modes)]*num_modes

    # Configuration w.r.t. data
    generate_normalized_data = True

    # generate data from each mode
    task_environment = PVDatasetMix(mu_t=mu_t, cov_t=cov_t,
                                    city_name=city_name,
                                    random_state=random_state)


    print('[INFO] generating data for {:2.0f} clients'.format(num_clients))
    clients_data, clients_train_ts, clients_test_ts = task_environment.generate_clients_data(num_clients=num_clients,
                                                          weight_modes=weight_modes)
    #print(task_environment.true_data_dist)
    #print(y_obs.shape)



