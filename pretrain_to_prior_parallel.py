import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from Aux import create_nn_per_layer, sample_from_GP
import pickle
import tensorflow as tf
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm, truncnorm
from scipy.interpolate import interp1d


which_example = 'materials_5outputs'


# Define some prior stuff
def sample_prior_set_norm(d=1, n_prior=500):
    x_values = LatinHypercube(d=d).random(n_prior).reshape((-1, d))
    for d_ in range(d):
        x_values[:, d_] = truncnorm(a=-2, b=2).ppf(x_values[:, d_])
    return x_values


def evaluate_prior_lin(x_values, mean_coefs, L, k0=1., nout=1, threshold_pos=None):
    y_prior = np.matmul(x_values, mean_coefs) + sample_from_GP(x_values, nout=nout, k0=k0, L=L)
    if threshold_pos is not None:
        for idx_y in range(y_prior.shape[1]):
            y_prior[:, idx_y] = np.maximum(y_prior[:, idx_y], threshold_pos[idx_y])
    return y_prior


def normalize_from_bounds(X, bounds, type_norm='[0,1]'):
    if type_norm == '[0,1]':
        return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    elif type_norm == '[-1,1]':
        return (X - (bounds[:, 1]+bounds[:, 0])/2.) / ((bounds[:, 1]-bounds[:, 0])/2.)


def evaluate_prior_for_materials(x_values):
    # mean
    theta_mat = np.array([[0.65577763, -0.19085542, 0.9333194, -0.87567097, 1.19500706],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
    y_prior = np.matmul(x_values, theta_mat)
    # uncertainty and positivity constraint
    idx = [[0, 1, 2], [0, 2, 3], [0, 1], [0, ], [0, 1, 2, 3]]
    bounds_outputs = np.array([(300., 800.), (0.16, 0.5), (100., 170.), (0.277, 0.299), (0., 0.8)])
    threshold_pos = normalize_from_bounds(0.01 * np.ones((1, 5)), bounds_outputs, type_norm='[-1,1]')[0]
    for j in range(5):
        y_tmp = y_prior[:, j] + sample_from_GP(x_values[:, idx[j]], nout=1, k0=0.2, L=0.8)[:, 0]
        y_prior[:, j] = np.maximum(y_tmp, threshold_pos[j])
    return y_prior


# Define some NN stuff
layers_shape = (20, 20, 20, 20)
act = lambda features: tf.nn.leaky_relu(features, alpha=0.01)
optimizer = Adam(learning_rate=0.001)
# Define some prior stuff
if which_example == '1d_synthetic':
    n_in, n_out = 1, 1
    priorA = lambda x: evaluate_prior_lin(x, mean_coefs=np.array([[0.]]), L=0.1)
    priorB = lambda x: evaluate_prior_lin(x, mean_coefs=np.array([[2.]]), L=0.1)
    priorC = lambda x: evaluate_prior_lin(x, mean_coefs=np.array([[2.]]), L=1.)
    all_prior_funcs = [priorA, priorB, priorC]
    all_epochs = [2000, 2000, 300]
elif which_example == 'materials_5outputs':
    n_in, n_out = 4, 5
    theta_mat = np.array([[0.65577763, -0.19085542, 0.9333194, -0.87567097, 1.19500706],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]])
    idx_prior = [[0, 1, 2], [0, 2, 3], [0, 1], [0, ], [0, 1, 2, 3]]
    bounds_outputs = np.array([(300., 800.), (0.16, 0.5), (100., 170.), (0.277, 0.299), (0., 0.8)])
    threshold_pos = normalize_from_bounds(0.01 * np.ones((1, 5)), bounds_outputs, type_norm='[-1,1]')[0]
    #priorA = lambda x: evaluate_prior_lin(x, mean_coefs=theta_mat, L=0.4, k0=1., nout=n_out, threshold_pos=threshold_pos)
    #priorB = lambda x: evaluate_prior_lin(x, mean_coefs=theta_mat, L=0.4, k0=0.5, nout=n_out, threshold_pos=threshold_pos)
    #priorC = lambda x: evaluate_prior_lin(x, mean_coefs=theta_mat, L=0.8, k0=0.2, nout=n_out, threshold_pos=threshold_pos)
    all_prior_funcs = [evaluate_prior_for_materials, ]
    all_epochs = [1000, ]
else:
    raise ValueError


def train_one_NN(i, N, j, init_weights):
    evaluate_prior, epochs = all_prior_funcs[j], all_epochs[j]
    if (i+1) % 10 == 0:
        print('Training NN {} / {}'.format(i + 1, N))
    x_values = sample_prior_set_norm(d=n_in, n_prior=500)
    y_values = evaluate_prior(x_values)
    modif_init_weights = []
    for w_init in init_weights:
        new_w_init = w_init + 0.01 * np.random.randn(*w_init.shape)
        modif_init_weights.append(new_w_init)
    tmp_model = create_nn_per_layer(
        n_in=n_in, n_out=n_out, layers_shape=layers_shape, act=act, lmda_reg=0., optimizer=optimizer)
    tmp_model.set_weights(modif_init_weights)
    history = tmp_model.fit(x_values, y_values, epochs=epochs, shuffle=True, verbose=0)
    loss = history.history['loss'].copy()
    prior_weights = tmp_model.get_weights().copy()
    return prior_weights, loss


def main(N, j, init_weights):
    pool = multiprocessing.Pool()
    results = pool.starmap(train_one_NN, [(i, N, j, init_weights) for i in range(N)])
    pool.close()
    pool.join()

    all_prior_weights = []
    all_losses = []
    for i, result in enumerate(results):
        all_prior_weights.append(result[0])
        all_losses.append(result[1])

    return all_prior_weights, all_losses


if __name__ == "__main__":
    start_time = time.time()
    N = 50
    tmp_model = create_nn_per_layer(
        n_in=n_in, n_out=n_out, layers_shape=layers_shape, act=act, lmda_reg=0., optimizer=optimizer)
    init_weights = tmp_model.get_weights()

    dict_results = {'init_weights': init_weights}
    for j in range(len(all_prior_funcs)):
        print()
        print("!!!!!!!!!!!!!! Starting prior {} !!!!!!!!!!!!!!".format(j))
        print()
        start_time = time.time()
        all_ws, losses = main(N, j, init_weights)
        print("--- %s seconds ---" % (time.time() - start_time))
        dict_results['prior_{}'.format(j)] = (all_ws, losses)

    with open('priors_081124_N[50]_{}.pkl'.format(which_example), 'wb') as handle:
        pickle.dump(dict_results, handle)
        