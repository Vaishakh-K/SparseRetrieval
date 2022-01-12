'''
This file contains code for training the MLP
The terms weights from COIL are used to build the document vector
Pair-wise training is done using hard negatives
Cosine similarity measure is used
'''
import argparse
import json
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
parser.add_argument('--modelpath', type=str, required=True)
parser.add_argument('--range', type=int, required=True)
parser.add_argument('--quantization', type=int, required=True)
args = parser.parse_args()

print("Starting...")


def readTSVFile(tsvFilePath):
    tsvFile = open(tsvFilePath)
    tsvFile_readTSV = csv.reader(tsvFile, delimiter=' ')
    return tsvFile_readTSV


def readFileToNumpyArray(inputFile):
    numpyArray = np.loadtxt(inputFile, delimiter=' ', dtype=str)
    return numpyArray


print("Reading qrelsFile file")
qrelsFile = readFileToNumpyArray('/bos/tmp16/vkeshava/coil/data/msmarco-doctrain-qrels.tsv')
print("qrelsFile len", len(qrelsFile))

print("Reading doctrain file")
doctrain = readFileToNumpyArray('/bos/tmp16/vkeshava/coil/data/msmarco-doctrain-top100')
print("doctrain len", len(doctrain))

print ("Build Train Quey Vector ")
# Read the trainig query vector
qry_all_ids = np.load('/bos/tmp16/vkeshava/coil/data/encodedTrainQry/tok_pids.npy')
qry_all_reps = np.load('/bos/tmp16/vkeshava/coil/data/encodedTrainQry/tok_reps.npy')
with open('/bos/tmp16/vkeshava/coil/data/encodedTrainQry/offsets.pkl', 'rb') as pf:
    qry_offset_dict = pickle.load(pf)

# Build contextualizedQryVector
contextualizedQryVector = {}
for q_tok_id in qry_offset_dict:
    start_idx = qry_offset_dict[q_tok_id][0]
    end_idx = start_idx + qry_offset_dict[q_tok_id][1] - 1

    for i in range(start_idx, end_idx):
        cur_tok_id = str(q_tok_id)
        cur_tok_psg = str(qry_all_ids[i])
        cur_tok_weight = qry_all_reps[i]

        if cur_tok_psg not in contextualizedQryVector:
            contextualizedQryVector[cur_tok_psg] = np.zeros(32)
        contextualizedQryVector[cur_tok_psg] = np.add(contextualizedQryVector[cur_tok_psg], cur_tok_weight)

QryKeys = contextualizedQryVector.keys()
print("len(QryKeys)", len(QryKeys))


# default network
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = torch.nn.Linear(32, 1)  # hidden layer
        self.output = torch.nn.Linear(1, 32)  # output layer

    def forward(self, x):
        x = self.hidden(x.double())  # activation function for hidden layer
        # x = F.relu(self.hidden(x.double()))    # activation function for hidden layer
        x = self.output(x)  # output
        return x


print ("Initialize the MLP")
# Initialize the MLP
mlp = MLP()
mlp = mlp.double()
LR = 0.01
BATCH_SIZE = 16
EPOCH = 1
# Define the loss function and optimizer
loss_function = nn.MarginRankingLoss()
opt_Adam = torch.optim.Adam(mlp.parameters(), lr=LR, betas=(0.9, 0.99))


# Function to get BM25 hard negatives
def getNegativeDocIdForQry(qryId, posDocId, DocKeys):
    didx = np.where(doctrain[:, 0] == qryId)
    for i in didx[0]:
        docId = doctrain[i][2][1:]
        if docId in DocKeys and docId != posDocId:
            return docId


def train(QryKeys, DocKeys, qrelsFile, contextualizedQryVector, contextualizedDocVector):
    # Find the key in contextual Doc vector whose query vector we possess
    count = 0
    gradAcc = 16
    for i in range(len(DocKeys)):
        docId = 'D' + list(DocKeys)[i]
        # Go through qrels file and find entry with this docId.
        # Gives qry for which the docId is positive result
        idx = np.where(qrelsFile[:, 2] == docId)
        if len(idx[0]) > 0:
            qidx = idx[0][0]
            qryIdInQrels = qrelsFile[qidx][0]
            if qryIdInQrels in contextualizedQryVector:
                qryId = qryIdInQrels
                posdocId = docId[1:]
                negdocId = getNegativeDocIdForQry(qryId, posdocId, DocKeys)
                if negdocId is None:
                    continue
                # print ("DocId", docId[1:])
                print (qryId, posdocId, negdocId)

                # Build training examples
                queryVector = torch.tensor(contextualizedQryVector[qryId])
                positiveDocVector = torch.tensor(contextualizedDocVector[posdocId])
                negativeDocVector = torch.tensor(contextualizedDocVector[negdocId])

                weightedPosDocVector = mlp(positiveDocVector)
                weightedNegDocVector = mlp(negativeDocVector)

                cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                posSimScore = cos(queryVector, weightedPosDocVector)
                negSimScore = cos(queryVector, weightedNegDocVector)
                # print (posSimScore, negSimScore)

                posSimScore = posSimScore.reshape(1, -1)
                negSimScore = negSimScore.reshape(1, -1)

                loss = loss_function(posSimScore, negSimScore, torch.tensor([1]))
                print (loss)
                if gradAcc > 1:
                    loss = loss / gradAcc
                loss.backward()  # backpropagation, compute gradients

                if (count + 1) % gradAcc == 0:
                    opt_Adam.step()
                    opt_Adam.zero_grad()  # clear gradients for next train
                    count = 0

                count += 1


print ("Build Doc Vector for training ")
# Build Doc Vector for training
for filename in sorted(os.listdir(args.input)):
    tok_all_ids = np.load(f'{args.input}/{filename}/tok_pids.npy')
    tok_all_reps = np.load(f'{args.input}/{filename}/tok_reps.npy')
    with open(f'{args.input}/{filename}/offsets.pkl', 'rb') as pf:
        offset_dict = pickle.load(pf)

    contextualizedDocVector = {}
    for tok_id in offset_dict:
        start_idx = offset_dict[tok_id][0]
        end_idx = start_idx + offset_dict[tok_id][1]

        for i in range(start_idx, end_idx):
            cur_tok_id = str(tok_id)
            cur_tok_psg = str(tok_all_ids[i])  # docId
            cur_tok_weight = tok_all_reps[i]  # coil weight vector for token

            if cur_tok_psg not in contextualizedDocVector:
                contextualizedDocVector[cur_tok_psg] = np.zeros(32)
            contextualizedDocVector[cur_tok_psg] = np.add(contextualizedDocVector[cur_tok_psg], cur_tok_weight)

    # Model Training
    DocKeys = contextualizedDocVector.keys()
    print("len(DocKeys)", len(DocKeys))
    print("Start training ", filename)
    train(QryKeys, DocKeys, qrelsFile, contextualizedQryVector, contextualizedDocVector)

    # Save model
    print ("Saving model")
    torch.save(mlp, f'{args.modelpath}/{filename}')

print("Compute weights and write results")
for filename in sorted(os.listdir(args.input)):
    tok_all_ids = np.load(f'{args.input}/{filename}/tok_pids.npy')
    tok_all_reps = np.load(f'{args.input}/{filename}/tok_reps.npy')
    with open(f'{args.input}/{filename}/offsets.pkl', 'rb') as pf:
        offset_dict = pickle.load(pf)

    scores = []
    psg_tok_weight_dict = {}
    for tok_id in tqdm(offset_dict):
        start_idx = offset_dict[tok_id][0]
        end_idx = start_idx + offset_dict[tok_id][1]
        for i in range(start_idx, end_idx):
            cur_tok_id = str(tok_id)
            cur_tok_psg = str(tok_all_ids[i])
            cur_tok_weightV = np.zeros(32)
            cur_tok_weightV = np.add(cur_tok_weightV, tok_all_reps[i])
            cur_tok_weight = mlp.hidden(torch.tensor(cur_tok_weightV))
            cur_tok_weight = 0 if cur_tok_weight < 0 else float(cur_tok_weight)
            scores.append(cur_tok_weight)
            cur_tok_weight = int(np.ceil(cur_tok_weight / args.range * (2 ** args.quantization)))
            if cur_tok_psg not in psg_tok_weight_dict:
                psg_tok_weight_dict[cur_tok_psg] = {}
            if cur_tok_id not in psg_tok_weight_dict[cur_tok_psg]:
                psg_tok_weight_dict[cur_tok_psg][cur_tok_id] = cur_tok_weight
            elif psg_tok_weight_dict[cur_tok_psg][cur_tok_id] < cur_tok_weight:
                psg_tok_weight_dict[cur_tok_psg][cur_tok_id] = cur_tok_weight

    with open(f'{args.output}/{filename}', 'w') as f:
        for pid in sorted(list(set(tok_all_ids))):
            f.write(json.dumps({'id': str(pid), 'contents': '', 'vector': psg_tok_weight_dict[str(pid)]}) + '\n')

print("done")
