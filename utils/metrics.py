import os
import itertools
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics.pairwise import rbf_kernel
import torch
from tqdm import tqdm
import numpy as np
import time
from Bio import SeqIO
import matplotlib.pyplot as plt
from utils.dataloader import onehot_encoder

##
# From paper Conditional generative modeling for de novo protein
# design with hierarchical functions
# github: https://github.com/timkucera/proteogan


PATH = os.path.dirname(os.path.realpath(__file__))
terms = ['A']
amino_acid_alphabet = 'ARNDCEQGHILKMFPSTWYV-'


def make_kmer_trie(k):
    '''
    For efficient lookup of k-mers.
    '''
    kmers = [''.join(i)
             for i in itertools.product(amino_acid_alphabet, repeat=k)]
    # print(len(amino_acid_alphabet))
    kmer_trie = {}
    for i, kmer in enumerate(kmers):
        tmp_trie = kmer_trie
        for aa in kmer:
            if aa not in tmp_trie:
                tmp_trie[aa] = {}
            if 'kmers' not in tmp_trie[aa]:
                tmp_trie[aa]['kmers'] = []
            tmp_trie[aa]['kmers'].append(i)
            tmp_trie = tmp_trie[aa]
    return kmer_trie


three_mer_trie = make_kmer_trie(3)


def spectrum_map(sequences, k=3, mode='count', normalize=True, progress=False):
    '''
    Maps a set of sequences to k-mer vector representation.
    '''

    if isinstance(sequences, str):
        sequences = [sequences]
    if k == 3:
        trie = three_mer_trie
    else:
        trie = make_kmer_trie(k)

    def matches(substring):
        d = trie
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['kmers']

    def map(sequence):
        vector = np.zeros(len(amino_acid_alphabet)**k)
        for i in range(len((sequence))-k+1):
            for j in matches(sequence[i:i+k]):
                if mode == 'count':
                    vector[j] += 1
                elif mode == 'indicate':
                    vector[j] = 1
        feat = np.array(vector)
        if normalize:
            norm = np.sqrt(np.dot(feat, feat))
            if norm != 0:
                feat /= norm
        return feat

    it = tqdm(sequences) if progress else sequences
    return np.array([map(seq) for seq in it], dtype=np.float32)


def mmd(seq1=None, seq2=None, emb1=None, emb2=None, mean1=None, mean2=None,
        embedding='spectrum', kernel='linear', kernel_args={}, return_pvalue=False,
        progress=False, **kwargs):
    '''
    Calculates MMD between two sets of sequences. 
    Optionally takes embeddings or mean embeddings of sequences if 
    these have been precomputed for efficiency. 
    If <return_pvalue> is true, a Monte-Carlo estimate (1000 iterations) 
    of the p-value is returned. Note that this is compute-intensive and only 
    implemented for the linear kernel.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map

    if mean1 is None and emb1 is None:
        emb1 = embed(seq1, progress=progress, **kwargs)
    if mean2 is None and emb2 is None:
        emb2 = embed(seq2, progress=progress, **kwargs)

    if not mean1 is None and not mean2 is None:
        MMD = np.sqrt(np.dot(mean1, mean1) +
                      np.dot(mean2, mean2) - 2*np.dot(mean1, mean2))
        return MMD

    if kernel == 'linear':
        x = np.mean(emb1, axis=0)
        y = np.mean(emb2, axis=0)
        MMD = np.sqrt(np.dot(x, x) + np.dot(y, y) - 2*np.dot(x, y))
    elif kernel == 'gaussian':
        x = np.array(emb1)
        y = np.array(emb2)
        m = x.shape[0]
        n = y.shape[0]
        Kxx = rbf_kernel(x, x, **kernel_args)  # .numpy()
        Kxy = rbf_kernel(x, y, **kernel_args)  # .numpy()
        Kyy = rbf_kernel(y, y, **kernel_args)  # .numpy()
        MMD = np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )

    if return_pvalue:
        agg = np.concatenate((emb1, emb2), axis=0)
        mmds = []
        it = tqdm(range(1000)) if progress else range(1000)
        for i in it:
            np.random.shuffle(agg)
            _emb1 = agg[:m]
            _emb2 = agg[m:]
            mmds.append(mmd(emb1=_emb1, emb2=_emb2,
                        kernel=kernel, kernel_args=kernel_args))
        rank = float(sum([x <= MMD for x in mmds]))+1
        pval = (1000+1-rank)/(1000+1)
        return MMD, pval
    else:
        return MMD


def entropy(seq1=None, seq2=None, emb1=None, emb2=None, embedding='spectrum', **kwargs):
    '''
    Calculates the average entropy over embedding dimensions between two sets 
    of sequences. Optionally takes embeddings of sequences if these have been 
    precomputed for efficiency.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map

    if not seq1 is None:
        emb1 = embed(seq1, **kwargs)
    if not seq2 is None:
        emb2 = embed(seq2, **kwargs)

    lo = np.min(np.vstack((emb1, emb2)), axis=0)
    hi = np.max(np.vstack((emb1, emb2)), axis=0)

    def _entropy(emb):
        res = 0
        for i, col in enumerate(emb.T):
            hist, _ = np.histogram(col, bins=1000, range=(lo[i], hi[i]))
            res += scipy_entropy(hist, base=2)
        res = res / emb.shape[1]
        return res

    return _entropy(emb2)-_entropy(emb1)


def distance(seq1=None, seq2=None, emb1=None, emb2=None, embedding='spectrum', **kwargs):
    '''
    Calculates the average pairwise distance between two sets 
    of sequences in mapping space. Optionally takes embeddings of sequences 
    if these have been precomputed for efficiency.
    '''

    if embedding == 'spectrum':
        embed = spectrum_map

    if not seq1 is None:
        emb1 = embed(seq1, **kwargs)
    if not seq2 is None:
        emb2 = embed(seq2, **kwargs)

    def _distance(emb):
        res = np.sum(emb ** 2, axis=1, keepdims=True) + \
            np.sum(emb ** 2, axis=1, keepdims=True).T - 2 * np.dot(emb, emb.T)
        res = np.mean(res)
        return res

    return _distance(emb2)-_distance(emb1)


def pearson_cor(seq1=None, seq2=None, random_n=2000, gap_below=0.75):
    # randomly pick 1000 sequences from both sequences
    # to calculate person's correlation score
    # Only positions with a number of gaps below 75% are represented
    # positions above 75% gap receive a score of zero in output array
    seq_len = len(seq1[0])
    cor_array = torch.zeros(seq_len)
    seq1_size = len(seq1)
    seq2_size = len(seq2)
    min_size = min(seq1_size, seq2_size)
    # can't sample more than min size of two seqences
    if random_n > min_size:
        random_n = min_size
    for i in range(seq_len):
        list1 = [seq[i] for seq in seq1]
        list2 = [seq[i] for seq in seq2]
        n_gap = list2.count('-')
        if n_gap >= gap_below * min_size:
            cor_array[i] = 0
        else:
            list1 = np.random.choice(list1, size=random_n, replace=False)
            list2 = np.random.choice(list2, size=random_n, replace=False)
            
            
            list1 = onehot_encoder(list1).argmax(dim=-1).sort()[0]
            list2 = onehot_encoder(list2).argmax(dim=-1).sort()[0]
            
            cor_array[i] = np.corrcoef(list1, list2)[0, 1]
    # calculate mean and std after removing zeros
    nonzero_idx = torch.nonzero(cor_array)
    nonzero_cor = cor_array[nonzero_idx]
    mean_cor = torch.mean(nonzero_cor)
    std_cor = torch.std(nonzero_cor)
    return cor_array, mean_cor, std_cor


def plot_gap_dist(seqs=None, run_name=None, record_gap_threshold=95, epoch=0):
    seq_len = len(seqs[0])
    gap_bin = np.zeros(seq_len)
    for seq in seqs:
        gap_indices = [pos for pos, char in enumerate(seq) if char == '-']
        gap_bin[gap_indices] += 1
    gap_bin = gap_bin/len(seqs)
    percentage_gap = 100*gap_bin
    x = np.array(range(len(seqs[0])))
    plt.figure()
    plt.plot(x, percentage_gap)
    plt.suptitle("Percentage of gap per locus")
    plt.xlabel("Locus")
    plt.ylabel("% of gap")
    # show loci with %gap >= threshold%
    threshold = 95
    indices = np.where(percentage_gap >= threshold)[0].tolist()
    plt.title("Locus with %gap >= " + str(threshold) +
              '% is:' + str(len(indices)) + '/' + str(len(seqs[0])))
    #plt.close()
    # plt.show()
    plt.savefig('out/'+run_name+'/plot_out/{}.png'.format(str(epoch).zfill(3)),
                bbox_inches='tight')
    plt.close()
    print("# ================== Sampling", len(seqs),
          "sequences at epoch",
          epoch, "================== #")
    print("The loci with percentage of gap >= %d%% in the cleaned MSA: \n" %
          threshold, indices)
    print("Their percentage gap are:\n", np.around(
        percentage_gap[indices], decimals=3).tolist())
    print("# ====================================================== #")
    return


if __name__ == '__main__':
    records1 = list(SeqIO.parse('MSA_data/bvmo_MSA_gap_0_6.fasta', "fasta"))
    seqs1 = [str(records1[i].seq.upper()) for i in range(len(records1))]
    records2 = list(SeqIO.parse('MSA_data/bvmo_MSA_gap_0_65.fasta', "fasta"))
    seqs2 = [str(records2[i].seq.upper()) for i in range(len(records2))]

    seqs1 = seqs1[:2000]
    #seqs1 = ['AAAAAAAAAAAAAAAACCCCCCCCCCCCCCCAAAAAAAAAAAAAAAA-------AAAAAAA'] * 1000
    start_time = time.time()
    MMD = mmd(seq1=seqs1, seq2=seqs2)
    entropy = abs(entropy(seq1=seqs1, seq2=seqs2))
    distance = abs(distance(seq1=seqs1, seq2=seqs2))
    cor_array, mean_cor, std_cor = pearson_cor(
        seq1=seqs1, seq2=seqs2, random_n=1000)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("MMD(lower), abs_D_entropy(lower), abs_D_distance(lower):")
    print(MMD, entropy, distance)
    print("Pearson's correlation, [mean, std]:", [mean_cor, std_cor])
    plot_gap_dist(seqs=seqs1, run_name='tmp', epoch=0)
    plot_gap_dist(seqs=seqs2, run_name='tmp', epoch=1)
    # print(cor_array)
