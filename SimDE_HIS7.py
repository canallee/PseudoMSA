import argparse
import os
import warnings
from Bio import SeqIO
from utils.dataloader import *
from utils.pesudo_MSA import *
from transformers import ESMForMaskedLM, ESMTokenizer, pipeline
from tqdm import tqdm

def warn(*args, **kwargs):
    pass

warnings.warn = warn
def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--max_edit_distance', type=int, default=30,
                        help='maximum edit distance to evolve')
    parser.add_argument('-M', '--min_mutant', type=int, default=1000,
                        help='minimum number of mutant for one distance')
    parser.add_argument('-r', '--random_n', type=int, default=128,
                        help='number of random single AA mutant to mask')
    parser.add_argument('-id', '--gen_id', type=int, default=0,
                        help='data generation id')
    parser.add_argument('-n', '--name_protein', type=str, 
                        default='HIS7', help='name of the protein')
    args = parser.parse_args()
    return args


def main():
    args = eval_parse()
    # User: set starting wild type sequences here!
    WT_AA_seq = WT_HIS7
    fasta_name = args.name_protein + '_pseudo_MSA_editDist' + \
        str(args.max_edit_distance) + '_' + str(args.gen_id) + '.fasta'
    fasta_name = args.name_protein + '_data/pseudo_MSA/' + fasta_name
    print("Arguments used: ", args)
    print("######### In-silico directed evolution begin #########")
    print("Generating data for fasta file:", fasta_name)
    # set parameters
    min_mutant = args.min_mutant
    random_n = args.random_n
    max_edit_distance = args.max_edit_distance
    # import ESM-1b
    tokenizer = ESMTokenizer.from_pretrained(
        "facebook/esm-1b", do_lower_case=False)
    model = ESMForMaskedLM.from_pretrained("facebook/esm-1b")
    pipe = pipeline('fill-mask', model=model, tokenizer=tokenizer,
                    device=0, top_k=5)
    pseudo_mutant_by_edit_dist = []
    percent_improved_by_edit_dist = []

    # edit_distance = 1; iterate through all possible single mutants for WT seq
    pseudo_mutant_ed1, percent_improved_ed1 = \
        generate_pseudo_mutant(WT_AA_seq, pipe, random_n=len(WT_AA_seq))

    # add mutants with edit distance 1
    pseudo_mutant_by_edit_dist.append(pseudo_mutant_ed1)
    percent_improved_by_edit_dist.append(percent_improved_ed1)
    # set mutants from edit distance 1 to previous round
    pseudo_mutant_ed_prev = pseudo_mutant_ed1
    percent_improved_ed_prev = percent_improved_ed1

    for _ in tqdm(range(max_edit_distance-1)):
        pseudo_mutant_ed_i, percent_improved_ed_i = in_silico_single_mutant_DE(
            pseudo_mutant_ed_prev, percent_improved_ed_prev,
            pipe, min_mutant, random_n)
        pseudo_mutant_by_edit_dist.append(pseudo_mutant_ed_i)
        percent_improved_by_edit_dist.append(percent_improved_ed_i)
        # update previous round
        pseudo_mutant_ed_prev = pseudo_mutant_ed_i
        percent_improved_ed_prev = percent_improved_ed_i

    # write fasta:
    outfile = open(fasta_name, 'w')

    for edit_distance in range(1, max_edit_distance+1):
        p_mutant_ed_i = pseudo_mutant_by_edit_dist[edit_distance-1]
        p_imprv_ed_i = percent_improved_by_edit_dist[edit_distance-1]
        for i, seq in enumerate(p_mutant_ed_i):
            improve_score = p_imprv_ed_i[i]
            seq_id = 'EditDist_'+str(edit_distance) + '_seq' + \
                str(i)+' | score='+str(float('%.4g' % improve_score))
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


if __name__ == '__main__':
    main()
