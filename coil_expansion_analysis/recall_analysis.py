from collections import defaultdict
import sys
sys.path.append("/home/adityasv/COILv2/SparseRetrieval/COIL")

import json
import torch
import os
from torch import nn

from tqdm.auto import tqdm
from beir.datasets.data_loader import GenericDataLoader
from transformers import PreTrainedModel
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import  MaskedLMOutput
from arguments import ModelArguments, DataArguments, TrainingArguments

from modeling import COILWithExpansion

from torch import Tensor
from typing import Dict, List, Tuple, Iterable
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

import numpy as np

model_path = "/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.0008_d0.0006_expansion_weight1_tokens_weight1_expansion-norm_max_expansion_weight_not_learned_init_distilbert_base/"

data_args, model_args, train_args = torch.load(os.path.join(model_path, "args.pt"))
config = AutoConfig.from_pretrained(
        model_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
model = COILWithExpansion.from_pretrained(model_args, data_args, train_args, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2tok = {v:k for k,v in tokenizer.vocab.items()}

if torch.cuda.is_available():
    model.to('cuda')
    model.eval()

def get_scores(query, doc):
    doc = [doc.get("title") + " " + doc.get("text")]
    query = [query]
    doc = tokenizer(doc, return_tensors='pt')
    query = tokenizer(query, return_tensors='pt')

    if torch.cuda.is_available():
        doc = {k: v.cuda() for k,v in doc.items()}
        query = {k: v.cuda() for k,v in query.items()}

    with torch.no_grad():
        scores, tok_scores, expansion_scores, qry_expansion, doc_expansion, qry_logits, doc_logits = model(query, doc, return_activations=True)
    
    return scores, tok_scores, expansion_scores, qry_expansion, doc_expansion 

data_path = "/bos/tmp15/adityasv/DomainAdaptation/MSMARCO/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

retrieved_results_path = os.path.join(model_path, "rank.txt")

not_retrieved_docs_path = os.path.join(model_path, "not_retrieved_docs.tsv")
not_retrieved_docs = defaultdict(set)
with open(not_retrieved_docs_path, 'r') as fi:
    for idx, line in tqdm(enumerate(fi)):
        if idx == 0:
            continue
        line = line.strip().split('\t')
        query_id, rel_doc_id = line
        not_retrieved_docs[query_id].add(rel_doc_id)

with open('retrieval_analysis.tsv', 'w') as fo:
    fo.write('\t'.join(['qid', 'docid', 'tok_scores', 'expansion_scores', 'is_retrieved']) + '\n')
    for qid in tqdm(qrels.keys()):
        for docid, score in qrels[qid].items():
            query = queries[qid]
            doc = corpus[docid]
            scores, tok_scores, expansion_scores, qry_expansion, doc_expansion = get_scores(query, doc)
            if qid not in not_retrieved_docs or docid not in not_retrieved_docs[qid]:
                fo.write('\t'.join([qid, docid, str(tok_scores.item()), str(expansion_scores.item()), 'retrieved'])+'\n')
            else:
                fo.write('\t'.join([qid, docid, str(tok_scores.item()), str(expansion_scores.item()), 'not_retrieved'])+'\n')