# %%
import sys
sys.path.append("/home/adityasv/COILv2/SparseRetrieval/COIL")

import json
import torch
import os
from torch import nn
import numpy as np

from tqdm.auto import tqdm
from transformers import PreTrainedModel
from transformers import DataCollatorWithPadding
from beir.datasets.data_loader import GenericDataLoader
from transformers.modeling_outputs import  MaskedLMOutput
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, BertTokenizer

from modeling import COILWithExpansion, COIL
from arguments import ModelArguments, DataArguments, TrainingArguments
from marco_datasets import GroupedMarcoTrainDataset, MarcoPredDataset, MarcoEncodeDataset

from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Iterable
from torch.cuda.amp import autocast

import pdb
# %%
model_path = "/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.08_d0.06_expansion_weight0.1/"

# model_path = "/bos/tmp3/vkeshava/data/output/model_coil_tok_bert_nod2q"

data_args, model_args, train_args = torch.load(os.path.join(model_path, 'args.pt'))

config = AutoConfig.from_pretrained(
        model_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )

config.output_hidden_states= True
model = COILWithExpansion.from_pretrained(model_args, data_args, train_args, model_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir)

base_implementation = COIL.from_pretrained(model_args, data_args, train_args, model_path, add_pooling_layer=False)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

id2tok = {v:k for k,v in tokenizer.vocab.items()}

if torch.cuda.is_available():
    model.to('cuda')
    base_implementation.to('cuda')

model.eval()
base_implementation.eval()

def get_scores(query, doc, model):
    
    doc = [doc.get("title") + " " + doc.get("text")]
    query = [query]
    doc = tokenizer(doc, return_tensors='pt', add_special_tokens=True)
    query = tokenizer(query, return_tensors='pt', add_special_tokens=True)

    if torch.cuda.is_available():
        doc = {k: v.cuda() for k,v in doc.items()}
        query = {k: v.cuda() for k,v in query.items()}

    with torch.no_grad():
        scores, qry_expansion, doc_expansion = model(query, doc, return_activations=True)
    return scores, None, None, qry_expansion, doc_expansion

def get_encodings(query, doc, model):
    
    doc = [doc.get("title") + " " + doc.get("text")]
    query = [query]
    doc = tokenizer(doc, return_tensors='pt')
    query = tokenizer(query, return_tensors='pt')

    if torch.cuda.is_available():
        doc = {k: v.cuda() for k,v in doc.items()}
        query = {k: v.cuda() for k,v in query.items()}

    with torch.no_grad():
        qcls, qtok = model.encode(**query)
        dcls, dtok = model.encode(**doc)
    return qcls, qtok, dcls, dtok

# %%
# data_path = "/bos/tmp15/adityasv/DomainAdaptation/MSMARCO/msmarco"
# corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

# # test_qid = list(qrels.keys())[0]
# # test_docid = list(qrels[test_qid].keys())[0]

# test_qid = "1048585"
# # test_docid = "5979392"
# test_docid = "7187158"

# test_query = queries[test_qid]
# # test_doc = corpus[test_docid]
# test_doc = {"title": "", "text": "the presence of communication amid scientific minds was equally important to the success of the manhattan project as scientific intellect was. the only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant ; hundreds of thousands of innocent lives obliterated."}

# # pdb.set_trace()
# base_scores, tok_scores, expansion_scores, qry_expansion, doc_expansion = get_scores(test_query, test_doc, base_implementation)
# scores, tok_scores, expansion_scores, qry_expansion, doc_expansion = get_scores(test_query, test_doc, model)

# print(scores, base_scores)

# qcls, qtok, dcls, dtok = get_encodings(test_query, test_doc, model)
# base_qcls, base_qtok, base_dcls, base_dtok = get_encodings(test_query, test_doc, base_implementation)

# print(torch.all(qtok==base_qtok), torch.all(dtok==base_dtok))
# print(torch.all(qcls==qry_expansion))


# test_cls = np.load("/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_q0.08_d0.06_expansion_weight0.1/msmarco-passage-encoding/split00/temp.npy")
# dcls = np.squeeze(dcls.cpu().numpy())
# print(np.count_nonzero(test_cls), np.count_nonzero(dcls))
# print(np.all(dcls == np.squeeze(test_cls)))


# %% 
from torch import tensor

input_ids = tensor([[  101,  1996,  3739,  1997,  4807, 13463,  4045,  9273,  2001,  8053, \
          2590,  2000,  1996,  3112,  1997,  1996,  7128,  2622,  2004,  4045, \
         24823,  2001,  1012,  1996,  2069,  6112,  5689,  2058,  1996,  8052, \
          6344,  1997,  1996,  9593,  6950,  1998,  6145,  2003,  2054,  2037, \
          3112,  5621,  3214,  1025,  5606,  1997,  5190,  1997,  7036,  3268, \
         27885, 22779,  9250,  1012,   102,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0], \
        [  101,  1996,  7128,  2622,  1998,  2049,  9593,  5968,  3271,  3288, \
          2019,  2203,  2000,  2088,  2162,  2462,  1012,  2049,  8027,  1997, \
          9379,  3594,  1997,  9593,  2943,  4247,  2000,  2031,  2019,  4254, \
          2006,  2381,  1998,  2671,  1012,   102,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0, \
             0,     0,     0,     0,     0,     0,     0,     0]], \
       device='cpu')
attention_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
         1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0], \
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
         0, 0, 0, 0, 0, 0, 0, 0]], device='cpu')


encode_dataset = MarcoEncodeDataset(
            "/bos/tmp3/vkeshava/data/corpus/split00", tokenizer, p_max_len=data_args.p_max_len
        )
encode_loader = DataLoader(
    encode_dataset,
    batch_size=128,
    collate_fn=DataCollatorWithPadding(
        tokenizer,
        max_length=data_args.p_max_len,
        padding='max_length'
    ),
    shuffle=False,
    drop_last=False,
    num_workers=1,
)

model.cpu()
batch = next(iter(encode_loader))
batch = {k:v.cpu() for k,v in batch.items()}

pdb.set_trace()
cls, reps = model.encode(**batch)

print(cls.shape, torch.count_nonzero(cls[0]))