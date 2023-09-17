from negative_sampler import RandomNegativeSampler, BatchRandomNegativeSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import time
import pickle
import numpy as np
import argparse
import random
from LDTGN import LDTGN
import torch
from edgebank import EdgeBank

parser = argparse.ArgumentParser(description='Train model for transductive testing.')
parser.add_argument('-d', '--data', type=str, default='wikipedia', help='name of the network dataset.')
parser.add_argument('--n_runs', type=int, default=5, help='number of runs.')
parser.add_argument('-l', '--learn', type=bool, default=False, help='Online Learning mode.')
parser.add_argument('-b', '--batch_size', type=int, default=200, help="Batch size.")
parser.add_argument('-i', '--inductive', type=bool, default=False, help='test inductive')
parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs in the online-learning scenario.')

args = parser.parse_args()

print(args)

DATASET = args.data
N_RUNS = args.n_runs
BATCH_SIZE = args.batch_size
LEARN = args.learn
INDUCTIVE = args.inductive
EPOCHS = args.epochs

with open(f"data/ml_{DATASET}.csv") as f:
    dataset = f.readlines()
    for i in range(1, len(dataset)):
        splitted_line = dataset[i].split(",")
        dataset[i] = [int(splitted_line[i]) - 1 for i in range(1, 3)] + [float(splitted_line[3])]

test_data_original = dataset[int(0.85 * len(dataset)):]


def make_data_inductive(run):
    rnd = random.Random(run)
    all_nodes = set()
    for pe in (dataset[1:]):
        s, d, _ = pe
        all_nodes.add(s)
        all_nodes.add(d)
    test_nodes = set()
    for pe in test_data_original:
        s, d, _ = pe
        test_nodes.add(s)
        test_nodes.add(d)
    test_nodes = list(test_nodes)
    new_test_nodes = set(rnd.sample(test_nodes, int(0.1 * len(all_nodes))))
    new_test_data = []
    for pe in test_data_original:
        s, d, t = pe
        if s in new_test_nodes or d in new_test_nodes:
            new_test_data.append((s, d, t))
    return new_test_data

def test_edgebank(model: EdgeBank, data, sampler):
    preds = []
    for pe in data:
        ne = sampler.sample()
        preds.append(model.predict(ne))
        preds.append(model.predict(pe,update=True))
    return [-1,1]*len(data),preds

def test_ldtgn(model:LDTGN, data, sampler):
    print("Start testing: ")
    if LEARN:
        model.train()
    else:
        model.eval()

    y = torch.Tensor([1, -1] * BATCH_SIZE).to(torch.device(model.device))
    updates_batch = []
    inputs_batch = []
    preds = []
    for pe in data:
        ne = sampler.sample()
        inputs_batch.append([pe, ne])
        updates_batch.append(pe)
        if len(updates_batch) == BATCH_SIZE:
            x, t, pred = model.forward(updates_batch, inputs_batch)
            preds.append(pred.reshape(-1))
            if LEARN:
                model.backward(y, pred)
                for _ in range(EPOCHS):
                    pred = model.forward_prediction_module(x, t)
                    model.backward(y, pred)
            updates_batch = []
            inputs_batch = []

    if len(updates_batch) > 0:
        x,t, pred = model.forward(updates_batch, inputs_batch)
        preds.append(pred.reshape(-1))
        if LEARN:
            y = torch.Tensor([1, -1] * len(updates_batch)).to(torch.device(model.device))
            model.backward(y, pred)
            for _ in range(EPOCHS):
                pred = model.forward_prediction_module(x,t)
                model.backward(y, pred)
    preds = torch.cat(preds).to("cpu")
    preds[preds < 0] = -1
    preds[preds >= 0] = 1

    return [1, -1] * len(data), preds.tolist()


scores = []
running_times = []
for i in range(N_RUNS):
    with open(f'trained_ol_{DATASET}_{i}.pkl', 'rb') as f:
        model = pickle.load(f)
        if type(model) == LDTGN:
            test = test_ldtgn
        else:
            test = test_edgebank
    if INDUCTIVE:
        test_data = make_data_inductive(i)
        test_sources = list(set([s for s, _, _ in test_data]))
        test_dests = list(set([d for _, d, _ in test_data]))
    else:
        test_data = test_data_original
        test_sources = list(set([s for s, _, _ in dataset[1:]]))
        test_dests = list(set([d for _, d, _ in dataset[1:]]))

    sampler = RandomNegativeSampler(test_data, test_sources, test_dests)
    start = time.time()
    labels, pred_labels = test(model, test_data, sampler)
    end = time.time()
    score = average_precision_score(labels, pred_labels)
    scores.append(score)
    running_times.append(end - start)
    print(f"Test AP: {score}")
    print(f"running time: {end - start}")
scores = np.array(scores) * 100
running_times = np.array(running_times)
print(f"average AP score: {'{:.2f}'.format(scores.mean())}±{'{:.2f}'.format(scores.std())}")
print(f"average running time: {'{:.2f}'.format(running_times.mean())}")
print(f"average throughput: {'{:.2f}'.format(len(test_data)/running_times.mean())}")
