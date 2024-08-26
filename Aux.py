import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy import interpolate
from scipy.special import logsumexp
from scipy.stats import norm, gennorm, truncnorm
from scipy.stats.qmc import LatinHypercube

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Input, Dropout
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from sklearn.gaussian_process.kernels import RBF


# Functions needed to set NNs
def neg_log_likelihood_loss(y_true, y_pred, noise_scale=1.):
    # y_true and y_pred are n_train x n_features
    scale_squared = np.atleast_2d(noise_scale ** 2)
    return 0.5 * tf.reduce_sum((y_true - y_pred) ** 2 / scale_squared)


class DegenerateCovRegularizer(Regularizer):
    def __init__(self, sing_values, sing_vectors, w0=0.):
        self.sing_values = sing_values
        self.sing_vectors = sing_vectors
        if (self.sing_values is not None) and (self.sing_values is not None):
            self.pre_computed_prod = np.diag(1. / self.sing_values) @ self.sing_vectors.T
        self.w0 = w0

    def __call__(self, w):
        if self.sing_values is None or self.sing_vectors is None:
            return 0.
        n_sing = len(self.sing_values)
        # neglog_sqrt_pseudo_det = 0.5*K*np.log(2*np.pi/(K-1)) + np.sum(np.log(S_vec))
        prod_term = self.pre_computed_prod @ tf.reshape(w - self.w0, (-1, 1))
        neglog_prior_weights = 0.5 * (n_sing - 1) * tf.transpose(prod_term) @ prod_term  # +neglog_sqrt_pseudo_det
        return neglog_prior_weights[0, 0]

    def get_config(self):
        return {'sing_values': self.sing_values, 'sing_vectors': self.sing_vectors, 'w0': self.w0}


class FactorizedRegularizer(Regularizer):
    def __init__(self, coef, beta, w0=0.):
        self.coef = coef
        self.beta = beta
        self.w0 = w0

    def __call__(self, w):
        return self.coef * tf.reduce_sum(tf.math.pow(tf.math.abs(w - self.w0), self.beta))

    def get_config(self):
        return {'coef': self.coef, 'beta': self.beta, 'w0': self.w0}


class LayerFullNN(Layer):
    def __init__(self, layers_shape, act=tf.nn.relu, n_in=1, n_out=1, sing_values=None, sing_vectors=None,
                 prior_weights=None):
        super().__init__()
        self.act = act
        self.kernels_shape = [(l1, l2) for l1, l2 in zip(
            (n_in,) + layers_shape, layers_shape + (n_out,))]
        self.bias_shape = [(l1,) for l1 in layers_shape + (n_out,)]
        n_weights = int(np.sum([np.prod(ks) for ks in self.kernels_shape]) + np.sum(
            [np.prod(bs) for bs in self.bias_shape]))
        # Same as He initialization
        init_scale = np.zeros((n_weights,))
        nc = 0
        for ks, bs in zip(self.kernels_shape, self.bias_shape):
            init_scale[nc:nc + np.prod(ks)] = np.sqrt(2. / ks[0])
            nc += np.prod(ks) + np.prod(bs)
        # Prior weights, if given, is in per layer format, concatenate
        if prior_weights is None:
            w0 = 0.
        else:
            w0 = np.concatenate([w_.reshape((-1,)) for w_ in prior_weights], axis=0)
            assert len(w0) == n_weights
        # Create the trainable set of parameters
        self.weights_full = self.add_weight(
            initializer=RandomNormal(stddev=init_scale), shape=(n_weights,), trainable=True,
            regularizer=DegenerateCovRegularizer(sing_values=sing_values, sing_vectors=sing_vectors, w0=w0)
        )

    def call(self, inputs):
        ks, bs = self.kernels_shape[0], self.bias_shape[0]
        _w = tf.reshape(self.weights_full[:np.prod(ks)], ks)
        _b = tf.reshape(self.weights_full[np.prod(ks):np.prod(ks) + np.prod(bs)], bs)
        x = inputs @ _w + _b
        nc = np.prod(ks) + np.prod(bs)
        for ks, bs in zip(self.kernels_shape[1:], self.bias_shape[1:]):
            x = self.act(x)
            _w = tf.reshape(self.weights_full[nc:nc + np.prod(ks)], ks)
            _b = tf.reshape(self.weights_full[nc + np.prod(ks):nc + np.prod(ks) + np.prod(bs)], bs)
            x = x @ _w + _b
            nc += np.prod(ks) + np.prod(bs)
        return x


def create_nn_full(layers_shape, sing_values, sing_vectors, act=tf.nn.relu, n_in=1, n_out=1, prior_weights=None,
                   noise_scale=1., optimizer=Adam, verbose=False):
    # Define neural network
    inputs = Input(shape=(n_in,))
    outputs = LayerFullNN(
        layers_shape=layers_shape, act=act, n_in=n_in, n_out=n_out, sing_values=sing_values,
        sing_vectors=sing_vectors, prior_weights=prior_weights
    )(inputs)
    model = Model(inputs, outputs)
    if verbose:
        model.summary()
    model.compile(
        optimizer=optimizer, loss=lambda y_true, y_pred: neg_log_likelihood_loss(y_true, y_pred, noise_scale))
    return model


def create_nn_per_layer(layers_shape, lmda_reg=0., power_reg=2., act=tf.nn.relu, n_in=1, n_out=1, prior_weights=None,
                        noise_scale=1., optimizer=Adam(), verbose=False):
        # Pre-process some inputs
        if isinstance(lmda_reg, (float, int)):
            lmda_reg = [lmda_reg, ] * (2 * len(layers_shape) + 2)
        if isinstance(power_reg, (float, int)):
            power_reg = [power_reg, ] * (2 * len(layers_shape) + 2)
        if prior_weights is None:
            prior_weights = [0., ] * (2 * len(layers_shape) + 2)
        # Define neural network
        inputs = Input(shape=(n_in,))
        hidden = Dense(
            layers_shape[0], activation=act,
            kernel_regularizer=FactorizedRegularizer(coef=lmda_reg[0], beta=power_reg[0], w0=prior_weights[0]),
            bias_regularizer=FactorizedRegularizer(coef=lmda_reg[1], beta=power_reg[1], w0=prior_weights[1])
        )(inputs)
        for i in range(1, len(layers_shape)):
            hidden = Dense(
                layers_shape[i], activation=act,
                kernel_regularizer=FactorizedRegularizer(coef=lmda_reg[2 * i], beta=power_reg[2 * i], w0=prior_weights[0]),
                bias_regularizer=FactorizedRegularizer(coef=lmda_reg[2 * i + 1], beta=power_reg[2 * i + 1], w0=prior_weights[2 * i + 1])
            )(hidden)
        outputs = Dense(
            n_out,
            kernel_regularizer=FactorizedRegularizer(coef=lmda_reg[-2], beta=power_reg[-2], w0=prior_weights[-2]),
            bias_regularizer=FactorizedRegularizer(coef=lmda_reg[-1], beta=power_reg[-1], w0=prior_weights[-1])
        )(hidden)

        model = Model(inputs, outputs)
        if verbose:
            model.summary()
        model.compile(
            optimizer=optimizer, loss=lambda y_true, y_pred: neg_log_likelihood_loss(y_true, y_pred, noise_scale))
        return model


def fit_prior_to_ensemble(pretrained_weights, which='factorized_gaussian', range_beta=(0.1, 10)):
    # Fit prior distribution to weight from ensemble
    # the first value is lambda=1/scale**beta and the second is beta
    # for gaussian this means first is 1/variance and second is 2
    # Specific case of degenerate gaussian
    if which == 'degenerate_gaussian':
        weights_from_ens = np.concatenate(pretrained_weights, axis=0)
        w0 = np.mean(weights_from_ens, axis=0)
        _, s, vt = svd(weights_from_ens - w0, full_matrices=False)
        return s[:-1], vt[:-1, ...].T

    # Per layer cases
    weights_from_ens = [w_[np.newaxis, ...] for w_ in pretrained_weights[0]]
    for w_n in pretrained_weights[1:]:
        weights_from_ens = [np.concatenate([w_ens, w_[np.newaxis, ...]], axis=0)
                            for w_ens, w_ in zip(weights_from_ens, w_n)]
    if which == 'factorized_gaussian':
        # should be a fit to w_-w0 where w0 is the mean of the distribution
        params = [(1./(2 * np.var(w_ - np.mean(w_, axis=0))), 2) for w_ in weights_from_ens]
    elif which == 'factorized_gen_normal':
        params = []
        for j, w_ in enumerate(weights_from_ens):
            # should be a fit to w_-w0 where w0 is the mean of the distribution
            demeaned_w_ = w_ - np.mean(w_, axis=0)
            gnfit = gennorm.fit(demeaned_w_.reshape((-1,)), floc=0)
            if gnfit[0] < range_beta[0]:
                gnfit = gennorm.fit(demeaned_w_.reshape((-1,)), floc=0, fbeta=range_beta[0])
            elif (range_beta[1] is not None) and (gnfit[0] > range_beta[1]):
                gnfit = gennorm.fit(demeaned_w_.reshape((-1,)), floc=0, fbeta=range_beta[1])
            params.append((1. / gnfit[2] ** gnfit[0], gnfit[0]))
    else:
        raise NotImplementedError
    lmdas, powers = tuple(zip(*params))
    return lmdas, powers


# GP prior functions
def sample_from_GP(xvals, nout=1, k0=1., L=1.):
    if len(xvals.shape) != 2:
        raise ValueError('Input xvals must be 2D (n_data x n_features)')
    cov_matrix = k0 * RBF(length_scale=L)(xvals)
    realization = np.random.multivariate_normal(
        mean=np.zeros((cov_matrix.shape[0], )), cov=cov_matrix, size=(nout,)).T
    return realization


def sample_from_GP_with_data(xvals, Xdata, Ydata, k0, L):
    ndata, nstar = len(Xdata), len(xvals)
    full_X = np.concatenate([Xdata, xvals], axis=0)
    full_cov = k0 * RBF(length_scale=L)(full_X)
    sigma_data, sigma_star = full_cov[:ndata, :ndata], full_cov[ndata:, ndata:]
    sigma_data_star = full_cov[:ndata, ndata:]
    tmp_mat = np.linalg.solve(sigma_data, sigma_data_star)
    post_mean = np.matmul(tmp_mat.T, Ydata)[:, 0]
    post_cov = sigma_star - np.matmul(tmp_mat.T, sigma_data_star)
    realization = np.random.multivariate_normal(post_mean, cov=post_cov)
    return realization


# Functions needed to assess accuracy of prediction uncertainty
def compute_lppd_from_ensemble_preds(y_true, y_pred, noise_scale):
    # y_true is n_test x n_features
    # y_pred is n_ens x n_test x n_features
    n_ens, n_test, n_features = y_pred.shape
    if isinstance(noise_scale, (int, float)):
        noise_scale = noise_scale * np.ones((n_features, ))
    log_probas = 0.
    errors = np.tile(y_true[np.newaxis, ...], (n_ens, 1, 1)) - y_pred
    for i in range(n_features):
        log_probas += norm.logpdf(errors[:, :, i], loc=0, scale=noise_scale[i])
    tmp_values = logsumexp(log_probas, axis=0)
    assert tmp_values.size == n_test
    lppd_value = n_test * np.log(1. / n_ens) + np.sum(tmp_values)
    return lppd_value


class ECDF:
    """
    Modified from scipy.stats v1.13.1 https://github.com/scipy/scipy/blob/v1.13.1/scipy/stats/_survival.py#L22
    """

    def __init__(self, sample):
        sample = np.sort(sample)
        q, counts = np.unique(sample, return_counts=True)
        # [1].81 "the fraction of [observations] that are less than or equal to x
        events = np.cumsum(counts)
        n = sample.size
        p = events / n

        f0 = 0  # leftmost function value
        f1 = 1 - f0
        # fill_value can't handle edge cases at infinity
        x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
        y = np.insert(p, [0, len(p)], [f0, f1])
        # `or` conditions handle the case of empty x, points
        self._f = interpolate.interp1d(
            x, y, kind='previous', assume_sorted=True)

    def evaluate(self, x):
        return self._f(x)


def calibration_curve(y_true, y_pred, std_pred, err=np.linspace(-3, 3, 100)):
    norm_residuals = (y_true - y_pred) / std_pred
    xcal, ycal = norm.cdf(err), ECDF(norm_residuals).evaluate(err)
    xcal_new = np.concatenate([[0.], xcal, [1.]], axis=0)
    xcal_weights = (np.diff(xcal_new[1:]) + np.diff(xcal_new[:-1])) / 2
    cal_area = np.sum(np.abs(ycal - xcal) * xcal_weights)
    return xcal, ycal, cal_area


class VanillaEnsemble:
    def __init__(self, n_ens, nn_kwargs, sample_from_aleatory=None, monitor_loss=True):
        self.n_ens = n_ens
        self.nn_kwargs = nn_kwargs  # n_in, n_out, layers_shape, act in a dictionary
        # given (x,y) return (x,y) bootstrapped or noise added
        self.sample_from_aleatory = sample_from_aleatory
        if self.sample_from_aleatory is None:
            self.sample_from_aleatory = self.bootstrap_data
        self.monitor_loss = monitor_loss
        self.posterior_weights = None
        self.data = None

    @staticmethod
    def bootstrap_data(x, y):
        ind_shuffle = np.random.choice(len(x), len(x), replace=True)
        return x[ind_shuffle], y[ind_shuffle]

    def fit(self, x, y, epochs):
        self.posterior_weights = []
        self.data = (x, y)
        losses = []
        for n in range(self.n_ens):
            # create nn
            tmp_model = create_nn_per_layer(lmda_reg=0., **self.nn_kwargs)
            x_al, y_al = self.sample_from_aleatory(x, y)
            history = tmp_model.fit(x_al, y_al, epochs=epochs, verbose=False, shuffle=True)
            losses.append(history.history['loss'])
            self.posterior_weights.append(tmp_model.get_weights())
            del tmp_model
        self.plot_loss(losses)

    def predict(self, x, which_dist='posterior'):
        y_pred = []
        tmp_model = create_nn_per_layer(lmda_reg=0., **self.nn_kwargs)
        for weights in self.posterior_weights:
            tmp_model.set_weights(weights)
            y_pred.append(tmp_model.predict(x, verbose=False))
        return np.array(y_pred)

    def plot_loss(self, losses):
        if self.monitor_loss:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            for loss in losses[:5]:
                ax.plot(loss)
            ax.set_xlabel('epochs')
            ax.set_ylabel('loss')
            ax.grid(True)
            ax.set_yscale('log')
            plt.show()


class AnchoredEnsemble(VanillaEnsemble):
    def __init__(self, n_ens, nn_kwargs, sample_from_prior=None, n_prior=400, sample_from_aleatory=None,
                 which_reg='factorized_gaussian', monitor_loss=True, prior_weights=None):
        super().__init__(n_ens=n_ens, nn_kwargs=nn_kwargs, sample_from_aleatory=sample_from_aleatory,
                         monitor_loss=monitor_loss)
        self.n_prior = n_prior
        self.sample_x_prior = lambda n: truncnorm(a=-2, b=2).ppf(LatinHypercube(d=1).random(n)).reshape((-1, 1))
        self.sample_y_prior = sample_from_prior      # return x and y from prior, number of points is decided within function
        self.prior_weights, self.prior_sets = prior_weights, None
        self.which_reg = which_reg
        if self.which_reg not in ['factorized_gaussian', 'factorized_gen_normal', 'degenerate_gaussian']:
            raise ValueError
        self.fit_p1, self.fit_p2 = None, None

    def fit(self, x, y, epochs, epochs_prior_fit=None):
        if self.prior_weights is None:
            self.prior_weights, self.prior_sets = [], []
            if self.which_reg == 'degenerate_gaussian':
                tmp_model = create_nn_full(sing_values=None, sing_vectors=None, **self.nn_kwargs)
            else:
                tmp_model = create_nn_per_layer(lmda_reg=0., **self.nn_kwargs)
            init_weights = tmp_model.get_weights()
            print('Pre-training to prior')
            if epochs_prior_fit is None:
                epochs_prior_fit = epochs
            losses = []
            for n in range(self.n_ens):
                x_prior = self.sample_x_prior(self.n_prior)
                y_prior = self.sample_y_prior(x_prior)
                self.prior_sets.append((x_prior, y_prior))
                tmp_model.set_weights([w_ + 0.001 * np.random.randn(*w_.shape) for w_ in init_weights])
                history = tmp_model.fit(x_prior, y_prior, epochs=epochs_prior_fit, shuffle=True, verbose=False)
                losses.append(history.history['loss'])
                self.prior_weights.append(tmp_model.get_weights())
            self.plot_loss(losses)
            del tmp_model
        # fit distribution to weights
        self.fit_p1, self.fit_p2 = fit_prior_to_ensemble(self.prior_weights, which=self.which_reg)
        print('Fitting to data')
        self.posterior_weights, self.data = [], (x, y)
        losses = []
        for n, prior_w in enumerate(self.prior_weights):
            if self.which_reg == 'degenerate_gaussian':
                tmp_model = create_nn_full(
                    sing_values=self.fit_p1, sing_vectors=self.fit_p2, prior_weights=prior_w, **self.nn_kwargs)
            else:
                tmp_model = create_nn_per_layer(
                    lmda_reg=self.fit_p1, power_reg=self.fit_p2, prior_weights=prior_w, **self.nn_kwargs)
            tmp_model.set_weights(prior_w)
            x_al, y_al = self.sample_from_aleatory(x, y)
            history = tmp_model.fit(x_al, y_al, epochs=epochs, shuffle=True, verbose=False)
            losses.append(history.history['loss'])
            self.posterior_weights.append(tmp_model.get_weights())
            del tmp_model
        self.plot_loss(losses)

    def predict(self, x, which_dist='posterior'):
        y_pred = []
        if which_dist == 'posterior':
            all_weights = self.posterior_weights
        elif which_dist == 'prior':
            all_weights = self.prior_weights
        else:
            return ValueError
        # create the NN then predict for all weights in the ensemble
        if self.which_reg == 'degenerate_gaussian':
            tmp_model = create_nn_full(sing_values=None, sing_vectors=None, **self.nn_kwargs)
        else:
            tmp_model = create_nn_per_layer(lmda_reg=0., **self.nn_kwargs)
        for weights in all_weights:
            tmp_model.set_weights(weights)
            y_pred.append(tmp_model.predict(x, verbose=False))
        return np.array(y_pred)
