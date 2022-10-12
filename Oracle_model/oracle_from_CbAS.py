# Oracle model from the paper 'Conditioning by adaptive sampling for robust design'
# Ensemble of simple MLP models implemented in TF
# Here we use the oracle model in Low-N setting (relatively small training set)
# https://github.com/dhbrookes/CbAS/tree/master/src 

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.dataloader import decode_one_seq
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras import models
import json
from tensorflow.python.keras import backend as K
import tensorflow as tf
import torch

def build_pred_model(n_tokens=4, seq_length=33, enc1_units=50):
    """Returns a keras model for predicting a scalar from sequences"""
    x = Input(shape=(seq_length, n_tokens))
    h = Flatten()(x)
    h = Dense(enc1_units, activation='elu')(h)
    out = Dense(2)(h)
    model = Model(inputs=[x], outputs=[out]) 
    return model

def neg_log_likelihood(y_true, y_pred):
    """Returns negative log likelihood of Gaussian"""
    y_true = y_true[:, 0]
    mean = y_pred[:, 0]
    variance = K.softplus(y_pred[:, 1]) + 1e-6
    log_variance = K.log(variance)
    return 0.5 * K.mean(log_variance, axis = -1) \
        + 0.5 * K.mean(K.square(y_true - mean) / variance, axis = -1)\
            + 0.5 * K.log(2 * np.pi)

def get_unique_X(X, return_idx = False):
    X_seq = [np.argmax(X[i], axis=1) for i in range(len(X))]
    unique_X = set(); unique_idx = []
    for i in range(len(X)):
        x_seq = decode_one_seq(X_seq[i].tolist())
        if x_seq not in unique_X:
            unique_idx.append(i)
            unique_X.add(x_seq)
    if return_idx:
        return unique_idx
    else:
        return X[unique_idx].numpy()

def get_experimental_X_y(X, y_gt, percentile=40, train_size=1000, 
                   random_state=1, return_y_noise=True):
    """Partition a (X, y) data set by a percentile of the y values
        Where the input y_gt is the groundtruth_model(X) value 
    """
    # X = X.numpy()
    np.random.seed(random_state)
    assert (percentile*0.01 * len(y_gt) >= train_size)
    y_percentile = np.percentile(y_gt, percentile)
    idx = np.where(y_gt < y_percentile)[0]
    rand_idx = np.random.choice(idx, size=train_size, replace=False)
    X_train = X[rand_idx]
    y_train = y_gt[rand_idx]
    if return_y_noise:
        y_train = y_gt + np.random.randn(*y_gt.shape) * 0.01
        return X_train, y_train[rand_idx]
    else:
        return X_train, y_train


def get_experimental_X_y_by_EditDist(
    X, y_gt, WT_aa_encoding, 
    train_size=1000, max_edit_distance=1,
    random_state=1, return_y_noise=True):
    """Partition a (X, y) data set, outputing only sequences 
    with edit distance smaller or equal to  max_edit_distance
    """
    X = torch.Tensor(X)
    compare = X.argmax(-1) != WT_aa_encoding
    idx = []
    for i in range(len(X)):
        distance = torch.sum(compare[i])
        #print(distance)
        if distance <= max_edit_distance and distance > 0:
            idx.append(i)
    rand_idx = np.random.choice(idx, size=train_size, replace=False)
    X_train = X[rand_idx].numpy()
    y_train = y_gt[rand_idx]
    if return_y_noise:
        y_train = y_gt + np.random.randn(*y_gt.shape) * 0.01
        return X_train, y_train[rand_idx]
    else:
        return X_train, y_train


def train_and_save_oracles(X_train, y_train, n_models=4, suffix='', n_char=20,
                           protein = "GFP", batch_size=32, ensemble_id=0,
                           train_size=256):
    """Trains a set of n oracles on a given set of data"""
    suffix = suffix + '_train' + str(train_size) + '_id' + str(ensemble_id)
    for i in range(n_models):
        model = build_pred_model(n_tokens=n_char, seq_length=X_train.shape[1], enc1_units=50)
        model.compile(optimizer='adam',
                      loss=neg_log_likelihood,
                      )
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=0, 
                                                   patience=5, 
                                                   verbose=1)

        model.fit(X_train, y_train, 
                  epochs=100, 
                  batch_size=batch_size, 
                  validation_split=0.1, 
                  callbacks=[early_stop],
                  verbose=2)
        model.save("%s_data/oracle_model/oracle_%i%s.h5" % (protein, i, suffix))
        
def load_oracles(protein='GFP', suffix='GFP', n_models=4, 
                 ensemble_id=0, train_size=256):
    suffix = suffix + '_train' + str(train_size) + '_id' + str(ensemble_id)
    weight_paths = [("%s_data/oracle_model/oracle_%i%s.h5" % 
                   (protein, i, suffix)) for i in range(n_models)]
    loss = neg_log_likelihood
    keras.utils.get_custom_objects().update({"neg_log_likelihood": loss})
    oracles = [keras.models.load_model(path) for path in weight_paths]
    return oracles

def get_balaji_predictions(oracles, Xt):
    """Given a set of predictors built according to the methods in 
    the Balaji Lakshminarayanan paper 'Simple and scalable predictive 
    uncertainty estimation using deep ensembles' (2017), returns the mean and
    variance of the total prediction."""
    M = len(oracles)
    N = Xt.shape[0]
    means = np.zeros((M, N))
    variances = np.zeros((M, N))
    for m in range(M):
        y_pred = oracles[m].predict(Xt)
        means[m, :] = y_pred[:, 0]
        variances[m, :] = np.log(1+np.exp(y_pred[:, 1])) + 1e-6
    mu_star = np.mean(means, axis=0)
    var_star = (1/M) * (np.sum(variances, axis=0) + np.sum(means**2, axis=0)) - mu_star**2
    return mu_star, var_star

