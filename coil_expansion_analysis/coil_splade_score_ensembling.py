import os
from tqdm.auto import tqdm
from collections import defaultdict

from numpy import interp

coil_scores_path   = '/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.0008_d0.0006_expansion_weight0_tokens_weight1_expansion-norm_max_expansion_weight_not_learned/rank.txt'
splade_scores_path = '/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.0008_d0.0006_expansion_weight1_tokens_weight0_expansion-norm_max_expansion_weight_not_learned/rank.txt'

coil_scores = defaultdict(dict)
splade_scores = defaultdict(dict)

with open(coil_scores_path, 'r') as fi:
    for line in tqdm(fi):
        qid, docid, score = line.strip().split('\t')
        coil_scores[qid][docid] = float(score)

with open(splade_scores_path, 'r') as fi:
    for line in tqdm(fi):
        qid, docid, score = line.strip().split('\t')
        splade_scores[qid][docid] = float(score)

base_output_path = '/bos/tmp15/adityasv/COILv2-vkeshava/interpolation_splade_coil'
for interpolation in tqdm(range(1,10)):
    with open(os.path.join(base_output_path, f'interpolation-{interpolation}-rank.txt'), 'w') as frank:
        interpolation *= 0.1
        for qid in coil_scores.keys():
            doc_ids = set(coil_scores[qid].keys()).union(set(splade_scores[qid].keys()))
            results = []
            for docid in doc_ids:
                score = 0
                score += (1-interpolation)*splade_scores[qid].get(docid, 0)
                score += interpolation*coil_scores[qid].get(docid, 0)
                results.append((score, docid))
            
            results = sorted(results, reverse=True)[:1000]
            for score, docid in results:
                frank.write('\t'.join([qid, docid, str(score)]) + '\n')