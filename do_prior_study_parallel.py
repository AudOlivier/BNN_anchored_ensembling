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


def sample_prior_set_norm(n_prior=400):
    x_values = truncnorm(a=-2, b=2).ppf(LatinHypercube(d=1).random(n_prior)).reshape((-1, 1))
    return x_values


# Define some prior stuff
def sample_from_mean_plus_GP(xvals, k0, L, mean_func):
    mean_ = mean_func(xvals)
    return mean_ + sample_from_GP(xvals, k0=k0, L=L)


def evaluate_prior_lin(x_values, k0, L):
    y_prior = sample_from_mean_plus_GP(x_values, k0=k0, L=L, mean_func=lambda x: 2*x)
    return y_prior


def evaluate_prior_cub(x_values, k0, L):
    y_prior = sample_from_mean_plus_GP(x_values, k0=k0, L=L, mean_func=lambda x: 5*(-1+2*np.random.rand()) * x**3)
    return y_prior


# Define some NN stuff
layers_shape = (50, 50, 50, 50)
act = lambda features: tf.nn.leaky_relu(features, alpha=0.01)
optimizer = Adam(learning_rate=0.001)
# Define some prior stuff
priorA = lambda x: evaluate_prior_lin(x, k0=0.6, L=0.8)
priorB = lambda x: evaluate_prior_lin(x, k0=0.6, L=0.2)
priorC = lambda x: evaluate_prior_lin(x, k0=0.6, L=0.05)
priorD = lambda x: evaluate_prior_cub(x, k0=0.1, L=0.2)
all_prior_funcs = [priorA, priorB, priorC, priorD]
all_epochs = [100, 200, 2000, 250]


def train_one_NN(i, N, j, init_weights, x_test):
    evaluate_prior, epochs = all_prior_funcs[j], all_epochs[j]
    if (i+1) % 10 == 0:
        print('Training NN {} / {}'.format(i + 1, N))
    x_values = sample_prior_set_norm()
    y_values = evaluate_prior(x_values)
    modif_init_weights = []
    for w_init in init_weights:
        new_w_init = w_init + 0.01 * np.random.randn(*w_init.shape)
        modif_init_weights.append(new_w_init)
    tmp_model = create_nn_per_layer(layers_shape=layers_shape, act=act, lmda_reg=0., optimizer=optimizer)
    tmp_model.set_weights(modif_init_weights)
    history = tmp_model.fit(x_values, y_values, epochs=epochs, shuffle=True, verbose=0)
    loss = history.history['loss'].copy()
    prior_weights = tmp_model.get_weights().copy()
    preds = [interp1d(x_values[:, 0], y_values[:, 0])(x_test[:, 0]).reshape((-1, 1)),
             tmp_model.predict(x_test, verbose=False)]
    return prior_weights, preds, loss


def main(N, j, init_weights, x_test):
    pool = multiprocessing.Pool()
    results = pool.starmap(train_one_NN, [(i, N, j, init_weights, x_test) for i in range(N)])
    pool.close()
    pool.join()

    all_prior_weights = []
    all_preds = []
    all_losses = []
    for i, result in enumerate(results):
        all_prior_weights.append(result[0])
        all_preds.append(result[1])
        all_losses.append(result[2])

    # Look at prior weights
    all_ws = [[] for _ in range(len(all_prior_weights[0]))]
    for nc in range(len(all_prior_weights[0])):
        for nc2 in range(len(all_prior_weights)):
            all_ws[nc].append(all_prior_weights[nc2][nc].reshape((-1,)))
    all_ws = [np.array(w_) for w_ in all_ws]
    #all_ws_concat = np.concatenate(all_ws[::2], axis=1)
    #print(all_ws_concat.shape)
    #var0 = np.var(all_ws_concat)
    #all_vars0 = [np.var(w_) for w_ in all_ws]
    #print(all_vars0)
    #print(var0)

    return all_ws, all_preds, all_losses


if __name__ == "__main__":
    start_time = time.time()
    N = 200
    tmp_model = create_nn_per_layer(layers_shape=layers_shape, act=act, lmda_reg=0., optimizer=optimizer)
    init_weights = tmp_model.get_weights()
    x_test = np.linspace(-1, 1, 100).reshape((-1, 1))

    #dict_results = {'init_weights': init_weights, 'x_test': x_test}
    with open('priors_072924_N[200].pkl', 'rb') as handle:
        dict_results = pickle.load(handle)
    for j, name in enumerate(['A', 'B', 'C', 'D']):
        if name in ['A', 'B', 'C']:
            continue
        print()
        print("!!!!!!!!!!!!!! Starting prior {} !!!!!!!!!!!!!!".format(name))
        print()
        start_time = time.time()
        all_ws, preds, losses = main(N, j, init_weights, x_test)
        print("--- %s seconds ---" % (time.time() - start_time))
        dict_results['prior_{}'.format(name)] = (all_ws, preds, losses)

    #with open('priors_073024_N[200].pkl', 'wb') as handle:
    #    pickle.dump(dict_results, handle)
        