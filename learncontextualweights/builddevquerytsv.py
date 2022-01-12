'''
The learned MLP is used to get the term weights for the dev query
The updated dev query is constructed
'''
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--range', type=int, required=True)
parser.add_argument('--quantization', type=int, required=True)
args = parser.parse_args()


# default network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(32, 1)   # hidden layer
        self.output = torch.nn.Linear(1, 32)   # output layer

    def forward(self, x):
        x = self.hidden(x.double())    # activation function for hidden layer
        #x = F.relu(self.hidden(x.double()))    # activation function for hidden layer
        x = self.output(x)            # output
        return x

# Path to the learned MLP
nmlp = torch.load('/bos/tmp16/vkeshava/coil/data/contextaulweightmodel/split12')
nmlp.eval()


tok_all_ids = np.load(f'{args.input}/tok_pids.npy')
tok_all_reps = np.load(f'{args.input}/tok_reps.npy')
with open(f'{args.input}/offsets.pkl', 'rb') as pf:
    offset_dict = pickle.load(pf)

scores = []
psg_tok_weight_dict = {}
for tok_id in tqdm(offset_dict):
    start_idx = offset_dict[tok_id][0]
    end_idx = start_idx + offset_dict[tok_id][1]
    for i in range(start_idx, end_idx):
        cur_tok_id = tok_id
        cur_tok_psg = tok_all_ids[i]
        cur_tok_weightV = np.zeros(32)
        cur_tok_weightV = np.add(cur_tok_weightV, tok_all_reps[i])
        cur_tok_weight = nmlp.hidden(torch.tensor(cur_tok_weightV))
        cur_tok_weight = 0 if cur_tok_weight < 0 else float(cur_tok_weight)
        cur_tok_weight = int(np.ceil(cur_tok_weight/args.range * (2**args.quantization)))
        scores.append(cur_tok_weight)
        if cur_tok_psg not in psg_tok_weight_dict:
            psg_tok_weight_dict[cur_tok_psg] = {}
        if cur_tok_id not in psg_tok_weight_dict[cur_tok_psg]:
            psg_tok_weight_dict[cur_tok_psg][cur_tok_id] = cur_tok_weight
        else:
            print(psg_tok_weight_dict[cur_tok_psg][cur_tok_id])
            print(cur_tok_weight)
            psg_tok_weight_dict[cur_tok_psg][cur_tok_id] += cur_tok_weight

with open(args.output, 'w') as f:
    for pid in sorted(list(set(tok_all_ids))):
        query = []
        values = max(list(psg_tok_weight_dict[pid].values()))
        for tok in psg_tok_weight_dict[pid]:
            freq = psg_tok_weight_dict[pid][tok]
            query.extend([str(tok)] * freq)
        query = " ".join(query)
        if len(query) > 0:
            f.write(f'{pid}\t{query}\n')
        else:
            f.write(f'{pid}\t102')
