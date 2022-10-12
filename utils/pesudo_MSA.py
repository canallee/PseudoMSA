from re import L
import torch
import random
import numpy as np
from Bio import SeqIO
from utils.dataloader import onehot_encoder, decode_one_seq_pMSA
from utils.AAV import *

# Wildtype Amino Acid sequences
WT_GFP = 'SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
WT_AAV2 = 'MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL'
WT_HIS7 = 'MTEQKALVKRITNETKIQIAISLKGGPLAIEHSIFPEKEAEAVAEQATQSQVINVHTGIGFLDHMIHALAKHSGWSLIVECIGDLHIDDHHTTEDCGIALGQAFKEALGAVRGVKRFGSGFAPLDEALSRAVVDLSNRPYAVVELGLQREKVGDLSCEMIPHFLESFAEASRITLHVDCLRGKNDHHRSESAFKALAVAIREATSPNGTNDVPSTKGVLM'


# Region of mutation for AAV (561-588 one-based)
AAV_START = 560
AAV_END = 588
AAV2_head = WT_AAV2[:AAV_START]
AAV2_tail = WT_AAV2[AAV_END:]
AAV2_target = WT_AAV2[AAV_START:AAV_END]

def compute_edit_distance(seq_a, seq_b):
    return sum(seq_a[i] != seq_b[i] for i in range(len(seq_a)))


def print_different_AA(seq_a, seq_b):
    for i in range(len(seq_a)):
        if seq_a[i] != seq_b[i]:
            print('diff at idx', i, '; AA:', seq_a[i], seq_b[i])
    print('===================================================')
    return sum(seq_a[i] != seq_b[i] for i in range(len(seq_a)))


def generate_pseudo_mutant(start_seq, pipe, random_n=64):
    '''
    Generate new mutant determined esm-1b fill mask score. For every possible 
    locus in the starting sequence, the AA is replaced by <mask> and esm-1b will
    infer the possible AA for the <mask>. If the filled AA is not the AA in the 
    starting sequence, then a new, predicted-higher-fitness mutant is found. 
    Conceptually, this produces single-mutation homologies for the start_seq
    Parameter:
        start_seq: start sequence
        pipe: fill mask pipeline from huggingface
        random_n: number of single mutants to be inferred
    Return:
        pseudo_mutant: list of mutants with higher ESM-1b likelihood than start_seq
        percent_improved: percentage improvement of the score for 1st over the 2nd
        AA, for the mutants in the pseudo_mutant
    '''
    start_seq_onehot = onehot_encoder(start_seq).argmax(dim=1)
    repeated = start_seq_onehot.expand(
        len(start_seq_onehot), len(start_seq_onehot)).clone()
    # mask 0th AA for 0th seq, 1th AA for 1th seq...
    masked_single_mutants = repeated.fill_diagonal_(20)
    single_masks = [decode_one_seq_pMSA(mutant)
                    for mutant in masked_single_mutants]
    single_masks = list(np.random.choice(
        single_masks, size=random_n, replace=False))
    unmasked = pipe(single_masks, batch_size=32)
    pseudo_mutant = []
    percent_improved = []
    for i in range(len(unmasked)):
        mutant_AA_seq = unmasked[i][0]['sequence'].replace(' ', '')
        score_0 = unmasked[i][0]['score']
        score_1 = unmasked[i][1]['score']
        percent_score_improved = (score_0 - score_1)/score_1
        if mutant_AA_seq != start_seq:
            pseudo_mutant.append(mutant_AA_seq)
            percent_improved.append(percent_score_improved)
    return pseudo_mutant, percent_improved


def generate_pseudo_mutant_w_insert(start_seq, pipe, p_insert=0.05, random_n=10,
                                    insert_improve_thres=0.5, context=0):
    '''
    Performs same task as generate_pseudo_mutant(), excepts that insertions are allowed
    in between WT amino acid loci. This is done by randomly generate an insertion mutant
    with p_insert every substitution. p_insert is 0.05 by default, base on the fact that
    measured frequencies of non-conservative insertion is 0.05 (N de la Chaux). Although 
    this code only mutates 561-588 region, inference is done based on the entire AAV2 
    sequence. 
    '''
    start_tokens = tokenize_mutation_seq(start_seq)
    subs, subs_ups, subs_sites, WT_AAs = mask_substitutes(start_tokens)
    insert, insert_ups, insert_sites = mask_inserts(start_tokens)
    chosen_inserts = [set(), set(), set()]
    if p_insert < 1.0:
        for i in range(len(subs)):
            toss_a_coin = random.random()
            if toss_a_coin < p_insert:
                # add an insert for selection
                which_insert = random.randint(0, len(insert)-1)
                #print(toss_a_coin, which_insert)
                chosen_inserts[0].add(insert[which_insert])
                chosen_inserts[1].add(insert_ups[which_insert])
                chosen_inserts[2].add(insert_sites[which_insert])
    else:
        for i in range(57 - len(subs)):
            chosen_inserts[0].add(insert[i])
            chosen_inserts[1].add(insert_ups[i])
            chosen_inserts[2].add(insert_sites[i])
            
    random_idx = np.random.choice(len(subs), size=random_n, replace=False)        
    mutant = list(chosen_inserts[0]) + [subs[i] for i in random_idx]
    mutant_ups = list(chosen_inserts[1]) + [subs_ups[i] for i in random_idx]
    mutant_sites = list(chosen_inserts[2]) + [subs_sites[i] for i in random_idx]
    
    # concat to make full length AAV sequence
    AAV2_head = WT_AAV2[AAV_START-context:AAV_START]
    AAV2_tail = WT_AAV2[AAV_END:AAV_END+context]   
    mutant_ups = [AAV2_head+mutant_up+AAV2_tail for mutant_up in mutant_ups]
    #print(len(mutant_ups[0]))
    unmasked = pipe(mutant_ups, batch_size=32)
    # collecting viable mutants
    pseudo_mutant = []
    percent_improved = []
    n_inserts = 0
    for i in range(len(unmasked)):
        mutant_to_fill_mask = mutant[i]
        mask_site = mutant_sites[i]
        first_choice_aa = unmasked[i][0]['token_str']
        # obtain the aav region after filling the mask site
        score_0 = unmasked[i][0]['score']
        score_1 = unmasked[i][1]['score']
        percent_score_improved = (score_0 - score_1)/score_1
        # if it's insert, accept the insertion if percent_score_improved
        # is greater than insert_improve_threshold
        if i in range(len(chosen_inserts[0])):
            mutant_new_seq = fill_mask(
                mutant[i], first_choice_aa, mutant_sites[i])
            if percent_score_improved > insert_improve_thres:
                pseudo_mutant.append(mutant_new_seq)
                percent_improved.append(percent_score_improved)
                n_inserts += 1
        # if substitution, do it normally
        else:
            mutant_new_seq = fill_mask(
                mutant[i], first_choice_aa, mutant_sites[i])
            wt_AA = WT_AAs[i - len(chosen_inserts[0])]
            if first_choice_aa != wt_AA:
                pseudo_mutant.append(mutant_new_seq)
                percent_improved.append(percent_score_improved)
    return pseudo_mutant, percent_improved, n_inserts


def in_silico_single_mutant_DE(
        pseudo_mutant_ed_prev, percent_improved_ed_prev, pipe,
        min_mutant=1000, random_n=32):
    '''
    Performing a round of in silico directed evolution to mutate a single AA, 
    with esm-1b likelihood being the in silico fitness score. Instead of taking 
    a single starting sequence as in the case of the first round (where start_seq
    is the WT sequence), here a list of possible starting mutant is taken as input
    (pseudo_mutant_ed_prev from previous round with edit distance i-1).
    A start_seq is chosen randomly (without replacement) from pseudo_mutant_ed_prev 
    with probability weighted by percent_improved_ed_prev, and random_n of its single 
    mutant is masked for inference by esm-1b. This process is repeated until the number
    of collected pseduo mutant with edit distance i is at least min_mutant
    Parameter:
        pseudo_mutant_ed_prev: pseudo homolgy with edit distance i-1 from previous
            round of in silicon single mutant DE
        percent_improved_ed_prev: precentage of score improvement of first choice AA
            over second choice AA for mutants in pseudo_mutant_ed_prev
        pipe: fill mask pipeline from huggingface
        min_mutant: minimum number of mutants to collect for edit distance i
        random_n: number of single AA mutant to fill mask for each start mutant
    Return:
        pseudo_mutant_ed_i: pseudo homolgy with edit distance i
        percent_improved_ed_i: precentage of score improvement for pseudo_mutant_ed_i
    '''
    pseudo_mutant_ed_i = []
    percent_improved_ed_i = []
    # normalize score to probability to weight samples
    sample_prob = percent_improved_ed_prev / np.sum(percent_improved_ed_prev)

    while len(pseudo_mutant_ed_i) < min_mutant \
        and len(pseudo_mutant_ed_prev) > 0:
        # select a mutant from previous edit distance for single mutations,
        # weighted by the sampling probability
        start_seq_idx = np.random.choice(
            len(pseudo_mutant_ed_prev), p=sample_prob)
        start_seq = pseudo_mutant_ed_prev[start_seq_idx]

        pseudo_mutant, percent_improved = \
            generate_pseudo_mutant(start_seq, pipe, random_n=random_n)
        pseudo_mutant_ed_i += pseudo_mutant
        percent_improved_ed_i += percent_improved
        #print(len(pseudo_mutant), len(percent_improved))
        # remove the sequence and its score, recalculate probability
        pseudo_mutant_ed_prev = pseudo_mutant_ed_prev[:start_seq_idx] + \
            pseudo_mutant_ed_prev[start_seq_idx+1:]
        percent_improved_ed_prev = percent_improved_ed_prev[:start_seq_idx] + \
            percent_improved_ed_prev[start_seq_idx+1:]
        sample_prob = percent_improved_ed_prev / \
            np.sum(percent_improved_ed_prev)
    return pseudo_mutant_ed_i, percent_improved_ed_i

def in_silico_single_mutant_DE_w_insert(
        pseudo_mutant_ed_prev, percent_improved_ed_prev, pipe, random_n=10,
        min_mutant=1000, p_insert=0.05, insert_improve_thres=0.1, context=0):
    '''
    Performing a round of in silico directed evolution to mutate a single AA.
    This version allows for single insertion between consecutive AAs
    '''
    
    pseudo_mutant_ed_i = []
    percent_improved_ed_i = []
    # normalize score to probability to weight samples
    sample_prob = percent_improved_ed_prev / np.sum(percent_improved_ed_prev)

    while len(pseudo_mutant_ed_i) < min_mutant \
        and len(pseudo_mutant_ed_prev) > 0:
        # select a mutant from previous edit distance for single mutations,
        # weighted by the sampling probability
        start_seq_idx = np.random.choice(
            len(pseudo_mutant_ed_prev), p=sample_prob)
        start_seq = pseudo_mutant_ed_prev[start_seq_idx]

        pseudo_mutant, percent_improved, n_inserts = \
            generate_pseudo_mutant_w_insert(
                start_seq, pipe, p_insert = p_insert, context=context,
                insert_improve_thres=insert_improve_thres, random_n=random_n)
        pseudo_mutant_ed_i += pseudo_mutant
        percent_improved_ed_i += percent_improved
        #print(len(pseudo_mutant), len(percent_improved))
        # remove the sequence and its score, recalculate probability
        pseudo_mutant_ed_prev = pseudo_mutant_ed_prev[:start_seq_idx] + \
            pseudo_mutant_ed_prev[start_seq_idx+1:]
        percent_improved_ed_prev = percent_improved_ed_prev[:start_seq_idx] + \
            percent_improved_ed_prev[start_seq_idx+1:]
        sample_prob = percent_improved_ed_prev / \
            np.sum(percent_improved_ed_prev)    
    return pseudo_mutant_ed_i, percent_improved_ed_i, n_inserts