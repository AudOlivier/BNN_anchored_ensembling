
import time
from Aux import *
import multiprocessing

# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_probability as tfp
# from scipy.stats.qmc import LatinHypercube
#
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, Dropout
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import Adam
#
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from scipy.optimize import fsolve

def main():
    start_time = time.time()
    N = 50
    n_iter = 4

    data_x = np.array([-4.07, -1.54, -2.98, -4.1, -0.26, -2.66, -4.66, -1.69, -0.92, -0.85, -2.35, -0.53, -3.28, -4.49,
                       -2.09, -3.8, -2.94, -2.58, -3.41, -4.25, -0.59, -2.26, -4.37, -2.15, -1.07, -4.08, -1.42, -2.62,
                       -2.0, -0.62, -3.0, -0.67, -3.61, -0.18, -4.98, -4.19, -1.02, -3.6, -3.19, -1.74, -2.28, -0.88,
                       -1.2, -0.33, -3.13, -4.44, -2.1, -4.29, -1.95, -2.83, 4.62, 4.3, 2.76, 3.87, 4.76, 4.0, 2.92,
                       2.84, 3.29, 3.78]).reshape((-1, 1))
    data_x = data_x / 8
    data_y = np.array([-1.58, -0.26, -0.08, -1.37, -0.37, 0.47, -0.44, 0.14, -1.35, -1.2, 0.7, -0.49, -0.56, -0.77,
                       0.47, -1.5, 0.02, 0.35, -0.99, -1.49, -0.93, 0.88, -1.27, 0.62, -0.78, -1.47, -0.2, 0.51, 0.82,
                       -1.16, -0.35, -1.19, -1.16, -0.21, -0.03, -0.93, -0.94, -0.91, -0.5, 0.03, 0.98, -1.05, -0.62,
                       -0.63, -0.18, -0.86, 0.79, -1.49, 0.42, 0.43, 0.78, 0.88, -0.52, 1.46, 0.53, 1.16, -0.03, -0.83,
                       0.48, 1.34]).reshape((-1, 1))

    data = (data_x, data_y)

    data_val_x = np.array([-4.84, -4.86, -3.27, -3.47, -1.21, -2.54, -2.77, -3.72, -1.41, -2.54, 2.59, 3.99]
                          ).reshape((-1, 1))
    data_val_x = data_val_x / 8
    data_val_y = np.array([-0.11, -0.22, -0.59, -0.94, -0.96, 0.73, 0.63, -1.59, -0.8, 0.43, -0.4, 1.44]
                          ).reshape((-1, 1))

    pool = multiprocessing.Pool()
    paramlist = []
    for i in range(N):
        paramlist.extend([[i, N, n_iter, data, data_val_x]])
    results = pool.map(do_one_ensemble, paramlist)
    pool.close()
    pool.join()

    print(results)

    all_weights = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}
    all_preds = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}
    xx = np.linspace(-1, 1)
    all_preds['x_preds'] = xx
    all_lppd = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}
    for result in results:
        for n0 in range(n_iter):
            all_weights['iter_{}'.format(n0)].append(result[0]['iter_{}'.format(n0)])  # all_preds

            test = result[1]['iter_{}'.format(n0)][0]
            all_preds['iter_{}'.format(n0)].append(result[1]['iter_{}'.format(n0)][0]) #all_preds

            y_pred = np.array(result[2]['iter_{}'.format(n0)])[:, :, 0]
            all_lppd['iter_{}'.format(n0)] = compute_lppd(data_val_y, y_pred)

    print("--- %s seconds ---" % (time.time() - start_time))

    fig, ax = plt.subplots(ncols=4, figsize=(16, 3))
    for i in range(4):
        yy_mc = np.array(all_preds['iter_{}'.format(i)])[:, :, 0]
        yy_mean, yy_std = np.mean(yy_mc, axis=0), np.std(yy_mc, axis=0)
        ps = np.percentile(yy_mc, q=[2.5, 97.5], axis=0)
        ax[i].plot(xx, yy_mean, color='black')
        ax[i].fill_between(xx, yy_mean - 2 * yy_std, yy_mean + 2 * yy_std, color='orange', alpha=0.3)
        # ax[i].fill_between(xx, ps[0, :], ps[1, :], color='green', alpha=0.3)
        ax[i].set_title('iter={} [lppd={:.2f}]'.format(i, all_lppd['iter_{}'.format(i)]), fontsize=15)
        ax[i].plot(data_x, data_y, linestyle='none', marker='.', color='blue')
        ax[i].plot(data_val_x, data_val_y, linestyle='none', marker='+', color='red')
        set_ax_lims(ax[i])
        for pos in ['right', 'top']:  # , 'bottom', 'left']:
            ax[i].spines[pos].set_visible(False)
    fig.tight_layout()
    plt.show()

    print('end... again?')
def do_one_ensemble(params):
    i, N, n_iter, data, data_val_x = params
    print('Training NN {} / {}'.format(i + 1, N))
    sampler = LatinHypercube(d=1)
    layers_shape = (50, 50, 50, 50)
    act = 'relu'
    tmp_model = architecture_l2(layers_shape=layers_shape, act=act, lmda_reg=0.)
    n_data = len(data[0])
    x_prior = -1. + 2. * sampler.random(n=3 * n_data).reshape((-1,))

    all_weights = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}
    xx = np.linspace(-1, 1)
    all_preds = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}
    all_preds['x_preds'] = xx
    all_lppd = {'iter_{}'.format(n0): [] for n0 in range(n_iter)}

    for n0 in range(n_iter):
        # Sample n_data values from the prior of previous posterior function
        if n0 == 0:
            y_prior = sample_from_prior(x_prior)
        else:
            y_prior = tmp_model.predict(x_prior, verbose=False)
        # Concatenate data + samples from prior / previous posterior
        tmp_data_x = np.concatenate([data[0], x_prior.reshape((-1, 1))], axis=0)
        tmp_data_y = np.concatenate([data[1], y_prior.reshape((-1, 1))], axis=0)
        # Re-sample the data from the known likelihood model - could do bootstrapping instead?
        tmp_data_y += np.random.normal(loc=0., scale=0.2, size=tmp_data_y.shape)
        ind_shuffle = np.random.choice(len(tmp_data_x), len(tmp_data_x), replace=False)
        # train
        tmp_model.fit(tmp_data_x[ind_shuffle], tmp_data_y[ind_shuffle], batch_size=len(tmp_data_x),
                      epochs=1000, shuffle=True, verbose=False)
        all_weights['iter_{}'.format(n0)].append(tmp_model.get_weights())
        all_preds['iter_{}'.format(n0)].append(tmp_model.predict(xx, verbose=False))
        y_val_pred = tmp_model.predict(data_val_x, verbose=False)
        all_lppd['iter_{}'.format(n0)].append(y_val_pred)

    return all_weights, all_preds, all_lppd

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))