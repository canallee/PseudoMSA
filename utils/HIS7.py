import numpy as np
import pandas as pd
import torch
import pandas
from utils.pesudo_MSA import WT_HIS7
from utils.dataloader import AA_IDX

def one_hot_encode_aa(aa_str):
    """Returns a one hot encoded amino acid sequence"""
    M = len(aa_str)
    aa_arr = np.zeros((M, 20), dtype=int)
    for i in range(M):
        aa_arr[i, AA_IDX[aa_str[i]]] = 1
    return aa_arr

def clean_HIS7_data():
    df = pd.read_csv('HIS7_data/his7.tsv', sep='\t')
    mutation_lst = df['mutation'].tolist()
    fitness_score = df['score'].tolist()
    aaSequence = []; edit_distance = []
    for mutations in mutation_lst:
        mut_lst = mutations.split(";")
        mutant_seq = WT_HIS7
        for mut in mut_lst:
            old_aa, new_aa = mut[0], mut[-1]
            locus = int(mut[1:-1]) - 1
            assert WT_HIS7[locus] == old_aa, 'wrong data'+mut
            mutant_seq = mutant_seq[:locus] + new_aa + mutant_seq[locus+1:]
        aaSequence.append(mutant_seq)
        edit_distance.append(len(mut_lst))
    out_df = pd.DataFrame(list(zip(mutation_lst,aaSequence,edit_distance,fitness_score)),
                        columns = ['mutation_lst', 'aaSequence', 
                                    'edit_distance', 'fitness_score'])
    out_df.to_csv('HIS7_data/his7.csv')    

def get_HIS7_X_y_aa(data_df, large_only=False, large_threshold = 1.0,
                   ignore_stops=True, return_str=False):
    """
    Converts the raw GFP data to a set of X and y values that are ready to use
    in a model
    """
    if large_only:
        min_fitness = data_df['fitness_score'].min()
        mean_fitness = data_df['fitness_score'].mean()
        idx = data_df.loc[
            ((data_df['fitness_score']-min_fitness)> 
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
        X[j] = one_hot_encode_aa(seqs[i].lower())
        j += 1
    y = np.array(data_df['fitness_score'][idx])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    if return_str:
        seqs = list(data_df['aaSequence'])
        seqs = [seq.upper() for seq in seqs]
        return X, y, seqs
    else:
        return X, y