# %%
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

# %%
model_path = "/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_cls_flops_q0.0008_d0.0006_expansion_weight1_tokens_weight1_expansion-norm_cls_expansion_weight_not_learned_init_distilbert_base/checkpoint-60000/"
# model_path = "/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.0008_d0.0006_expansion_weight1_tokens_weight0_expansion-norm_max_expansion_weight_not_learned/"

data_args, model_args, train_args = torch.load(os.path.join(model_path, "args.pt"))
model_args.expansion_normalization = 'cls'

# temp_path = 'bert-base-uncased'
# config = AutoConfig.from_pretrained(
#         temp_path,
#         num_labels=1,
#         cache_dir=model_args.cache_dir,
#         output_hidden_states=True
#     )
# model = COILWithExpansion.from_pretrained(model_args, data_args, train_args, temp_path, config=config)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

config = AutoConfig.from_pretrained(
        model_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
model = COILWithExpansion.from_pretrained(model_args, data_args, train_args, model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

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


# %%
data_path = "/bos/tmp15/adityasv/DomainAdaptation/MSMARCO/msmarco"
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

# %%
retrieved_results_path = os.path.join(model_path, "rank.txt")

retrieval_results = {}
with open(retrieved_results_path, 'r') as fi:
    for line in tqdm(fi): 
        query_id, doc_id, score = line.strip().split('\t')
        score = float(score)

        if query_id not in retrieval_results:
            retrieval_results[query_id] = {doc_id: score}
        else:
            retrieval_results[query_id][doc_id] = score

# %%
not_retrieved_docs_path = os.path.join(model_path,"not_retrieved_docs.tsv")

not_retrieved_docs = {}
with open(not_retrieved_docs_path, 'r') as fi:
    for idx, line in tqdm(enumerate(fi)):
        if idx == 0:
            continue
        line = line.strip().split('\t')
        query_id, rel_doc_id = line
        if query_id not in not_retrieved_docs:
            not_retrieved_docs[query_id] = []
        not_retrieved_docs[query_id].append(rel_doc_id)

# %%

out_data = []
with open('analysis.tsv', 'w') as fo:
    for query_idx, query_id in tqdm(enumerate(not_retrieved_docs.keys())):

        if query_idx == 200:
            break

        retrieved_docs = list(sorted([i for i in retrieval_results[query_id].items()], key=lambda x: x[1], reverse=True))[:2]
        
        rel_doc = json.dumps(corpus[not_retrieved_docs[query_id][0]])
        query = queries[query_id]

        fo.write(query+'\n')
        fo.write(rel_doc+'\n')
        fo.write('####\n')

        for retrieved_doc, score in retrieved_docs:
            doc = corpus[retrieved_doc]

            scores, tok_scores, expansion_scores, qry_expansion, doc_expansion = get_scores(query, doc)
            doc = json.dumps(doc)
            score = str(scores)
            tok_scores = str(tok_scores)
            expansion_score = str(expansion_scores)
            
            qry_expansion = torch.nonzero(qry_expansion)[:,1].cpu().numpy().tolist()
            doc_expansion = torch.nonzero(doc_expansion)[:,1].cpu().numpy().tolist()

            qry_expansion = " ## ".join([id2tok[i] for i in qry_expansion])
            doc_expansion = " ## ".join([id2tok[i] for i in doc_expansion])

            # qry_expansion = "##".join([id2tok[i] for i in qry_expansion])
            # doc_expansion = "##".join([id2tok[i] for i in doc_expansion])

            fo.write("\t".join([doc, score, tok_scores, expansion_score, qry_expansion, doc_expansion]) + "\n")
        
        fo.write("*"*50 + "\n")

# %%

expansion_scores_list = []
tok_scores_list = []
query_num_expansions = []
doc_num_expansions = []

for query_idx, query_id in tqdm(enumerate(retrieval_results.keys())):
    if query_idx == 1000:
        break
    retrieved_docs = list(sorted([i for i in retrieval_results[query_id].items()], key=lambda x: x[1], reverse=True))[:5]
    
    query = queries[query_id]

    for idx, item in enumerate(retrieved_docs):
        retrieved_doc, score = item
        doc = corpus[retrieved_doc]

        scores, tok_scores, expansion_score, qry_expansion, doc_expansion = get_scores(query, doc)

        expansion_scores_list.append(expansion_score.squeeze().detach().cpu().numpy().item())
        tok_scores_list.append(tok_scores.squeeze().detach().cpu().numpy().item())
        doc_num_expansions.append(torch.nonzero(doc_expansion).shape[0])
        if idx  == 0:
            query_num_expansions.append(torch.nonzero(qry_expansion).shape[0])

fig, axs = plt.subplots(4,1)
fig.tight_layout(h_pad=3 )

axs[0].hist(expansion_scores_list, label='expansion scores')
axs[0].set_title('expansion scores')

axs[1].hist(tok_scores_list, label='token scores')
axs[1].set_title('token scores')

axs[2].hist(query_num_expansions, label= "num query expansions")
axs[2].set_title("num query expansions")

axs[3].hist(doc_num_expansions, label= "num doc expansions")
axs[3].set_title("num doc expansions")

# base_model_name = model_path.split('/')[-1]
# base_model_name = base_model_name.replace("/", "")
# print(base_model_name)
model_path = os.path.dirname(model_path)
base_model_name = os.path.basename(model_path)

print('base_model_name', base_model_name)
plt.savefig(f"{base_model_name}.png")

expansion_scores_list = np.array(expansion_scores_list)
tok_scores_list = np.array(tok_scores_list)
query_num_expansions = np.array(query_num_expansions)
doc_num_expansions = np.array(doc_num_expansions)

print(f"expansion scores avg, std {np.mean(expansion_scores_list)}, {np.std(expansion_scores_list)}")
print(f"tok scores avg, std {np.mean(tok_scores_list)}, {np.std(tok_scores_list)}")
print(f"query expansions avg, std {np.mean(query_num_expansions)}, {np.std(query_num_expansions)}")
print(f"doc expansions avg, std {np.mean(doc_num_expansions)}, {np.std(doc_num_expansions)}")