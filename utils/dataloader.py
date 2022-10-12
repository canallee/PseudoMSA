import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm


AA = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v']
AA_IDX = {AA[i]:i for i in range(len(AA))}
AA_up = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']


def onehot_encoder(seq, no_pad=False):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
    if no_pad:
        amino_acids = amino_acids[:20]
    aa_to_int = dict((a, i) for i, a in enumerate(amino_acids))
    aa_to_int['X'] = 20
    aa_to_int['B'] = 20
    l = len(seq)
    onehot = torch.zeros((l, len(amino_acids)))
    for i, char in enumerate(seq):
        onehot[i][aa_to_int[char]] = 1
    return onehot

def decode_one_seq_pMSA(in_seq):
    '''
    require: not one-hot encode, but argmax after softmax. Only take in one sequence of length L
    '''
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '<mask>']
    seq = ''
    for AA_idx in in_seq:
        seq += amino_acids[AA_idx]
    return seq

def decode_one_seq(in_seq):
    '''
    require: not one-hot encode, but argmax after softmax. Only take in one sequence of length L
    '''
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
    seq = ''
    for AA_idx in in_seq:
        seq += amino_acids[AA_idx]
    return seq

class MSA_dataset_from_general(torch.utils.data.Dataset):
    '''
        Loading dropped MSA sequences specific to the WT sequence 
        from the fasta file for the general MSA fasta file (no columns dropped)
    '''
    def __init__(self, fasta, device):
        self.records = list(SeqIO.parse(fasta, "fasta"))
        self.length = len(self.records)
        self.seqs = [str(self.records[i].seq.upper()) for i in range(self.length)]
        self.valid_pos = [i for i, char in enumerate(self.seqs[0]) if char != '-']
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        seq = ''.join(np.array(list(self.seqs[index]))[self.valid_pos])
        print(seq)
        return onehot_encoder(seq).to(self.device)


class MSA_dataset(torch.utils.data.Dataset):
    '''
        Loading dropped MSA sequences specific to the WT sequence 
        from the fasta file for cleaned MSA fasta file specific to WT
    '''
    def __init__(self, fasta, device, pseudo_MSA=False):
        self.records = list(SeqIO.parse(fasta, "fasta"))
        self.length = len(self.records)
        self.seqs = [str(self.records[i].seq.upper()) for i in range(self.length)]
        # pre encoding to save time
        if pseudo_MSA:
            self.seqs_onehot = [
                onehot_encoder(seq, no_pad=True).to(device) for seq in tqdm(self.seqs)]
        else:
            self.seqs_onehot = [onehot_encoder(seq).to(device) for seq in tqdm(self.seqs)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return  self.seqs_onehot[index]
    
    
class MSA_dataset_faster(torch.utils.data.Dataset):
    '''
        Loading dropped MSA sequences specific to the WT sequence 
        from the fasta file for cleaned MSA fasta file specific to WT
    '''
    def __init__(self, seqs_onehot):
        self.length = len(seqs_onehot)
        self.seqs_onehot = seqs_onehot

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return  self.seqs_onehot[index]
 
 
if __name__ == '__main__':
    params = {
        'batch_size': 16,
        'shuffle': False,
    }
    train_data = MSA_dataset('MSA_data/bvmo_MSA_gap_0_65.fasta')
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    for batch, data in enumerate(train_loader):
        print(batch)
        break
    
    