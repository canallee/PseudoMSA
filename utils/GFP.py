# modified base on code for the paper 
# 'Conditioning by adaptive sampling for robust design'
# https://github.com/dhbrookes/CbAS 

import numpy as np
import torch
from utils.dataloader import AA_IDX

def one_hot_encode_aa(aa_str):
    """Returns a one hot encoded amino acid sequence"""
    M = len(aa_str)
    aa_arr = np.zeros((M, 20), dtype=int)
    for i in range(M):
        aa_arr[i, AA_IDX[aa_str[i]]] = 1
    return aa_arr


def get_gfp_X_y_aa(data_df, large_only=False, large_threshold = 1.0,
                   ignore_stops=True, return_str=False):
    """
    Converts the raw GFP data to a set of X and y values that are ready to use
    in a model
    """
    if large_only:
        min_fitness = data_df['medianBrightness'].min()
        mean_fitness = data_df['medianBrightness'].mean()
        idx = data_df.loc[
            ((data_df['medianBrightness']-min_fitness)> 
             large_threshold * (mean_fitness-min_fitness))].index
    else:
        idx = data_df.index
    data_df = data_df.loc[idx]
    
    if ignore_stops:
        idx = data_df.loc[~data_df['aaSequence'].str.contains('!')].index
    data_df = data_df.loc[idx]
    seqs = data_df['aaSequence']
        
    M = len(seqs[0])
    N = len(seqs)
    X = np.zeros((N, M, 20))
    j = 0
    for i in idx:
        X[j] = one_hot_encode_aa(seqs[i])
        j += 1
    y = np.array(data_df['medianBrightness'][idx])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    if return_str:
        seqs = list(data_df['aaSequence'])
        seqs = [seq.upper() for seq in seqs]
        return X, y, seqs
    else:
        return X, y