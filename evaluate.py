import numpy as np
import torch
import os
import time
from utils.quadtree import *
from utils.read_points import *
from models.models import nndp
from utils.point_shift import *
from utils.retrieve import *
import math
import argparse
import dgl
import torch


parser = argparse.ArgumentParser()

parser.add_argument('--test_size', type=int, default=100, help='number of nets')
parser.add_argument('--batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--test_data', type=str, default='', help='test data')
parser.add_argument('--decomps', type=int, default=8, help='point decompos')
parser.add_argument('--net', type=int, default=500, help='net')
parser.add_argument('--run_optimal', type=str, default='true', help='run GeoSteiner to generate optimal RSMT')

args = parser.parse_args()

device = torch.device("cuda:0")

base_dir = 'save/'
ckp_dir = base_dir + 'nnsteiner500.pt'

checkpoint = torch.load(ckp_dir)
model = nndp(args.net, device)
model.load_state_dict(checkpoint['nnsteiner_dict'])
model.eval()
evaluator = Evaluator()

test_cases = []
files = os.listdir(args.test_data)
for file in files:
    points = read_points(file)
    graphs = process_points(points)[2]
    test_cases.append(graphs)

num_batches = (args.test_size + args.batch_size - 1) // args.batch_size

all_lengths = []
all_outputs = []
for b in range(num_batches):
    test_batch = test_cases[b * args.batch_size: (b + 1) * args.batch_size]
    iter_lengths = [1e9 for i in range(len(test_batch))]
    iter_outputs = [[] for i in range(len(test_batch))]
    for t in range(args.decompos):
        decomp_batch = decompos_inputs(test_batch, t)
        with torch.no_grad():
            outpus, _ = model(decomp_batch, t)
        outputs = outputs.cpu().detach().numpy()
        lengths = evaluator.eval_batch(decomp_batch, outputs, args.degree)
        if t >= 4:
            outputs = np.flip(outputs, 1)
        for i in range(len(test_batch)):
            if lengths[i] < iter_lengths[i]:
                iter_lengths[i] = lengths[i]
                iter_outputs[i] = outputs[i]

    all_lengths.append(iter_lengths)
    all_outputs.append(iter_outputs)
all_lengths = np.concatenate(all_lengths, 0)
all_outputs = np.concatenate(all_outputs, 0)

print('Mean length', all_lengths.mean())