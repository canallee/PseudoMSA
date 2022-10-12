import pandas as pd
import torch
from utils.dataloader import *
from utils.GFP import *
from utils.torch_utils import *
from Groundtruth_model.CNN import *
from utils.pesudo_MSA import WT_HIS7

import warnings
warnings.filterwarnings("ignore")



def load_AAV_ground_CNNs(n_models = 6, train_epoch=300):
    def load_checkpoints(CNN, model_id, epoch):
        checkpoints = torch.load(
        os.path.join('AAV_data/ground_model/'+'CNN' + str(model_id) + \
        '_epoch_{}'.format(epoch)))  
        CNN.load_state_dict(checkpoints['CNN_state_dict'])
        return    
    ## load pretrained CNN into CPU, inference by CPU
    CNNs = []
    for i in range(n_models):
        CNN_tmp = CNN_ground_AAV(
            seq_len = 57, hidden_fc=256, hidden_conv=12, n_chars = 21)
        CNN_tmp = CNN_tmp.to(torch.device('cuda'))
        load_checkpoints(CNN_tmp, i, train_epoch)
        CNN_tmp = CNN_tmp.to(torch.device('cpu'))
        CNNs.append(CNN_tmp)
    return CNNs


def load_GFP_ground_CNNs(n_models = 6, train_epoch=300):
    def load_checkpoints(CNN, model_id, epoch):
        checkpoints = torch.load(
        os.path.join('GFP_data/ground_model/'+'CNN' + str(model_id) + \
        '_epoch_{}'.format(epoch)))  
        CNN.load_state_dict(checkpoints['CNN_state_dict'])
        return
    df = pd.read_csv('GFP_data/gfp_data.csv')
    X, _ = get_gfp_X_y_aa(df, large_only=True, ignore_stops=True)
    ## load pretrained CNN into CPU, inference by CPU
    CNNs = []
    for i in range(n_models):
        CNN_tmp = CNN_ground(
            seq_len = X.shape[1], hidden_fc=128, hidden_conv=12, n_chars = 20)
        CNN_tmp = CNN_tmp.to(torch.device('cuda'))
        load_checkpoints(CNN_tmp, i, train_epoch)
        CNN_tmp = CNN_tmp.to(torch.device('cpu'))
        CNNs.append(CNN_tmp)
    return CNNs

def load_HIS7_ground_AAVs(n_models = 6, train_epoch=300):
    def load_checkpoints(CNN, model_id, epoch):
        checkpoints = torch.load(
        os.path.join('HIS7_data/ground_model/'+'CNN' + str(model_id) + \
        '_epoch_{}'.format(epoch)))  
        CNN.load_state_dict(checkpoints['CNN_state_dict'])
        return
    seq_len = len(WT_HIS7)
    ## load pretrained CNN into CPU, inference by CPU
    CNNs = []
    for i in range(n_models):
        CNN_tmp = CNN_ground(
            seq_len = seq_len, hidden_fc=128, hidden_conv=12, n_chars = 20)
        CNN_tmp = CNN_tmp.to(torch.device('cuda'))
        load_checkpoints(CNN_tmp, i, train_epoch)
        CNN_tmp = CNN_tmp.to(torch.device('cpu'))
        CNNs.append(CNN_tmp)
    return CNNs


def ensemble_infer(models, Xt, return_var = False):
    """Modified from CbAS's get_balaji_predictions() function."""
    M = len(models)
    N = Xt.shape[0]
    means = np.zeros((M, N))
    variances = np.zeros((M, N))
    for m in range(M):
        y_pred = models[m](Xt).detach().cpu()
        means[m, :] = y_pred[:, 0]
        variances[m, :] = np.log(1+np.exp(y_pred[:, 1])) + 1e-6
    mu_star = np.mean(means, axis=0)
    var_star = (1/M) * (np.sum(variances, axis=0) + np.sum(means**2, axis=0)) - mu_star**2
    if return_var:
        return mu_star, var_star
    else:
        return mu_star
    

    