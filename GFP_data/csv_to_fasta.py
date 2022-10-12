import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.dataloader import *
from utils.GFP import *
from utils.torch_utils import *
from Groundtruth_model.CNN import *
from Groundtruth_model.ensemble import *
import warnings
warnings.filterwarnings("ignore")
from Oracle_model.oracle_from_CbAS import *

df = pd.read_csv('GFP_data/gfp_data.csv')
N_mutations = df["numAAMutations"].tolist()
X, _, X_seq = get_gfp_X_y_aa(df, large_only=True, ignore_stops=True, return_str=True)
unique_seq_idx = get_unique_X(X, return_idx=True)
unique_X_seq = get_unique_X(X)
# get number of AA mutation distribution
N_mutations_unique = [N_mutations[i] for i in unique_seq_idx]
unique_X_AA_seq = [X_seq[i] for i in unique_seq_idx]

fasta_name = "GFP_data/gfp_mutational.fasta"
outfile = open(fasta_name, 'w')
for i, seq in enumerate(unique_X_AA_seq):
    seq_id = 'GFP_seq' + str(i) + \
        '_nAAMutation' + str(N_mutations_unique[i])
    outfile.write('>' + seq_id + '\n')
    outfile.write(seq + '\n')
outfile.close()

# dedup and reformat fasta:
records = list(SeqIO.parse(fasta_name, "fasta"))
os.remove(fasta_name)

print("Total number of pseudo MSA sequences:", len(records))
seqs = [str(record.seq.upper()) for record in records]
seq_set = set()
for seq in seqs:
    seq_set.add(seq)
print("Number of nondup sequences:", len(seq_set))
output_records = []
for record in records:
    seq = record.seq
    if seq in seq_set:
        seq_set.remove(seq)
        output_records.append(record)
with open(fasta_name, 'w') as output_handle:
    SeqIO.write(output_records, output_handle, 'fasta')