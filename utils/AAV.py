import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from utils.dataloader import AA_up, decode_one_seq


AA_with_mask = AA_up+['<mask>']
AA_idx = {AA_up[i]:i for i in range(len(AA_up))}

def one_hot_encode_AAV(aa_str):
    """Returns a one hot encoded amino acid sequence"""
    M = len(aa_str)
    aa_arr = np.zeros((M, 21), dtype=int)
    for i in range(M):
        aa_arr[i, AA_idx[aa_str[i]]] = 1
    return aa_arr

def decode_aav_mutation_seq(seq):
    # seq is numerical instead of one-hot
    mut_seq = seq[:28]
    insert_seq = seq[28:]
    mut_AA_seq = decode_one_seq(mut_seq)
    insert_AA_seq = decode_one_seq(insert_seq).lower()
    aa_seq = ''
    # if there is insertion before AAV region
    if insert_seq[0] <= 19:
        aa_seq += insert_AA_seq[0]
    for i in range(len(mut_seq)):
        if insert_seq[i+1] <= 19:
            aa_seq += mut_AA_seq[i] + insert_AA_seq[i+1]
        else:
            aa_seq += mut_AA_seq[i]
    return aa_seq

def one_hot_aav_mutation_seq(seq, placeholder_token='-'):
    # The WT sequence length for targeted AAV region is 28
    # n_substitution is therefore 28 and n_insertion is 29
    # since the token for no insert is '-'
    # encoding is 57x21
    mut_seq = ''
    insert_seq = ''
    i = 0
    if seq[i].islower():
        insert_seq += seq[i].upper()
        i += 1
    else:
        insert_seq += placeholder_token
    while i < len(seq):
        if i < len(seq) - 1 and seq[i + 1].islower():
            mut_seq += seq[i]
            insert_seq += seq[i+1].upper()
            i += 2
        else:
            mut_seq += seq[i]
            insert_seq += placeholder_token
            i += 1
    expand_seq = mut_seq+insert_seq        
    return one_hot_encode_AAV(expand_seq) 

def tokenize_mutation_seq(seq, placeholder_token='_'):
    tokens = []
    i = 0
    if seq[i].islower():
        tokens.append((placeholder_token, seq[i].upper()))
        i += 1
    else:
        tokens.append((placeholder_token, placeholder_token))

    while i < len(seq):
        if i < len(seq) - 1 and seq[i + 1].islower():
            tokens.append((seq[i], seq[i+1].upper()))
            i += 2
        else:
            tokens.append((seq[i], placeholder_token))
            i += 1
    return tokens

def detokenize_mutation_seq(tokens, upper=False):
    assert len(tokens) == 29, \
        'AAV should have 29 pairs of subs and insert, but currently have '+str(len(tokens))
    def parse_no_gap(token):
        if token == '_':
            return ''
        else:
            return token
    aa_str = ''
    for i in range(len(tokens)):
        if i == 0:
            if upper:
                aa_str += parse_no_gap(tokens[i][1])
            else:
                # no substitution, only (potentially) insertion
                aa_str += parse_no_gap(tokens[i][1].lower())
        else:
            aa_str += parse_no_gap(tokens[i][0])
            if upper:
                aa_str += parse_no_gap(tokens[i][1])
            else:
                aa_str += parse_no_gap(tokens[i][1].lower())
    return aa_str

def fill_mask(masked_seq, aa, mask_site):
    i,j = mask_site
    if j == 0:
        masked_seq = masked_seq.replace('<mask>', '_')
    else:
        masked_seq = masked_seq.replace('<mask>', '')
    masked_tokens = tokenize_mutation_seq(masked_seq)
    token = list(masked_tokens[i])
    if j == 0:
        token[j] = aa.upper()
    else:
        token[j] = aa.lower()
    masked_tokens[i] = tuple(token)
    return detokenize_mutation_seq(masked_tokens)

def mask_substitutes(start_tokens):
    mutant_seqs_list = []
    mutant_seqs_upper_list = []
    mask_site_list = []
    WT_AA_list = []
    for i in range(len(start_tokens)):
        for j in range(2):
            token_to_mutate = start_tokens.copy()
            if (i, j) != (0, 0) and start_tokens[i][j] != '_':
                token = list(token_to_mutate[i])
                WT_AA_list.append(token[j])
                token[j] = '<mask>'
                token_to_mutate[i] = tuple(token)
                #print(token_to_mutate)
                mutant_seqs_list.append(
                    detokenize_mutation_seq(token_to_mutate))
                mutant_seqs_upper_list.append(
                    detokenize_mutation_seq(token_to_mutate, upper=True))
                mask_site_list.append((i, j))
    return mutant_seqs_list, mutant_seqs_upper_list, mask_site_list, WT_AA_list

def mask_inserts(start_tokens):
    mutant_seqs_list = []
    mutant_seqs_upper_list = []
    mask_site_list = []
    for i in range(len(start_tokens)):
        for j in range(2):
            token_to_mutate = start_tokens.copy()
            if (i, j) != (0, 0) and start_tokens[i][j] == '_':
                token = list(token_to_mutate[i])
                token[j] = '<mask>'
                token_to_mutate[i] = tuple(token)
                mutant_seqs_list.append(
                    detokenize_mutation_seq(token_to_mutate))
                mutant_seqs_upper_list.append(
                    detokenize_mutation_seq(token_to_mutate, upper=True))
                mask_site_list.append((i, j))
    return mutant_seqs_list, mutant_seqs_upper_list, mask_site_list

def get_AAV_X_y_aa(data_df, large_only=True, 
                   large_threshold = 1.0, return_str=False):
    if large_only:
        min_fitness = data_df['viable_score'].min()
        mean_fitness = data_df['viable_score'].mean()
        idx = data_df.loc[
            ((data_df['viable_score'] - min_fitness) > 
             large_threshold * (mean_fitness-min_fitness))].index        
    else:
        idx = data_df.index
    data_df = data_df.loc[idx]
    seqs = data_df['aaSequence']
    N = len(seqs)
    X = np.zeros((N, 28+29, 21))
    j = 0
    for i in idx:
        X[j] = one_hot_aav_mutation_seq(seqs[i])
        j += 1
    y = np.array(data_df['viable_score'][idx])
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    if return_str:
        seqs = list(data_df['aaSequence'])
        return X, y, seqs
    else:
        return X, y
    
    
class AAV_dataset(torch.utils.data.Dataset):
    '''
        Loading AAV target sequence variants
    '''
    def __init__(self, fasta, device):
        self.records = list(SeqIO.parse(fasta, "fasta"))
        self.length = len(self.records)
        self.seqs = [str(self.records[i].seq) for i in range(self.length)]
        # pre encoding to save time
        self.seqs_onehot = [torch.tensor(one_hot_aav_mutation_seq(seq)
                                         , dtype=torch.float32)
                            for seq in self.seqs]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return  self.seqs_onehot[index]  
 
 