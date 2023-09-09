from edgebank import EdgeBankUnlimited, EdgeBankTW, EdgeBankThresh,EdgeBank
from negative_sampler import RandomNegativeSampler
from sklearn.metrics import average_precision_score
import time
import pickle
import tqdm
import argparse
import numpy as np
import torch

import random
from LDTGN import LDTGN

parser = argparse.ArgumentParser(description='Train model for transductive/inductive testing.')
parser.add_argument('-d', '--data', type=str, default='wikipedia', help='name of the network dataset.')
parser.add_argument('--n_runs', type=int, default=5, help='number of runs.')
parser.add_argument('-m', '--model', type=str, default='LDTGN', help='model type')

# online model parameters
parser.add_argument('-n', '--normalization', default='dynamic', help='normalization type of online model')
parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int, default=200, help='batch size')
parser.add_argument('-us', '--use_split', type=bool, default=False, help='use split or not')
parser.add_argument('-i', '--inductive', type=bool, default=False, help='train inductive')
parser.add_argument('-l', '--learn', type=bool, default=False, help='Online Learning mode.')
parser.add_argument('-dv', '--device', type=str, default='cpu', help='Device of the model')

# edgebank parameters
parser.add_argument('-t', '--type', type=str, default='unlimited', help='type of edgebank')
parser.add_argument('-w', '--window', type=int, default=1000, help='window size of edgebank')
parser.add_argument('-th', '--threshold', type=float, default=2, help='threshold of edgebank')

args = parser.parse_args()

print(args)

DATASET = args.data
N_RUNS = args.n_runs
MODEL_TYPE = args.model
NORMALIZATION = args.normalization
FEATURES = args.features
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
USE_SPLIT = args.use_split
EDGE_BANK_TYPE = args.type
WINDOW_SIZE = args.window
THRESHOLD = args.threshold
LEARN = args.learn
DEVICE = args.device

assert MODEL_TYPE == 'LDTGN' or MODEL_TYPE == 'EdgeBank'
assert NORMALIZATION == 'static' or NORMALIZATION == 'dynamic'
assert DEVICE == 'cpu' or 'cuda:' in DEVICE
assert type(USE_SPLIT) is bool

assert EDGE_BANK_TYPE == 'unlimited' or EDGE_BANK_TYPE == 'tw' or EDGE_BANK_TYPE == 'thresh'

with open(f"../data/ml_{DATASET}.csv") as f:
    dataset = f.readlines()
    for i in range(1, len(dataset)):
        splitted_line = dataset[i].split(",")
        dataset[i] = [int(splitted_line[i]) - 1 for i in range(1, 3)] + [float(splitted_line[3])]
    dataset = dataset[1:]

training_data_original = dataset[:int(0.7 * len(dataset))]
validation_data_original = dataset[int(0.7 * len(dataset)):int(0.85 * len(dataset))]


def make_data_inductive(run):
    rnd = random.Random(run)
    all_nodes = set()
    for pe in (dataset[1:]):
        s, d, _ = pe
        all_nodes.add(s)
        all_nodes.add(d)
    test_nodes = set()
    for pe in (dataset[int(0.85 * len(dataset)):]):
        s, d, _ = pe
        test_nodes.add(s)
        test_nodes.add(d)
    test_nodes = list(test_nodes)
    new_test_nodes = set(rnd.sample(test_nodes, int(0.1 * len(all_nodes))))
    new_training_data = []
    for pe in training_data_original:
        s, d, t = pe
        if s not in new_test_nodes and d not in new_test_nodes:
            new_training_data.append((s, d, t))
    new_validation_data = []
    for pe in validation_data_original:
        s, d, t = pe
        if s in new_test_nodes or d in new_test_nodes:
            new_validation_data.append((s, d, t))
    return new_training_data, new_validation_data


def train_ldtgn(model: LDTGN, data, sampler):
    print("Start training: ")
    model.train()
    updates_batch = []
    inputs_batch = []
    y = torch.Tensor([1, -1] * BATCH_SIZE).to(DEVICE)
    for pe in tqdm.tqdm(data):
        ne = sampler.sample()
        inputs_batch.append([pe, ne])
        updates_batch.append(pe)

        if len(updates_batch) == BATCH_SIZE:
            x, t, preds = model.forward(updates_batch, inputs_batch)
            model.backward(y, preds)
            for _ in range(EPOCHS):
                preds = model.forward_prediction_module(x, t)
                model.backward(y, preds)
            updates_batch = []
            inputs_batch = []
    if len(updates_batch) > 0:
        y = torch.Tensor([1, -1] * len(updates_batch)).to(DEVICE)
        x, t, preds = model.forward(updates_batch, inputs_batch)
        model.backward(y, preds)
        for _ in range(EPOCHS):
            preds = model.forward_prediction_module(x, t)
            model.backward(y, preds)


def valid_ldtgn(model: LDTGN, data, sampler):
    print("Start validating: ")
    if LEARN:
        model.train()
    else:
        model.eval()

    y = torch.Tensor([1, -1] * BATCH_SIZE).to(torch.device(DEVICE))
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
            y = torch.Tensor([1, -1] * len(updates_batch)).to(torch.device(DEVICE))
            model.backward(y, pred)
            for _ in range(EPOCHS):
                pred = model.forward_prediction_module(x,t)
                model.backward(y, pred)
    preds = torch.cat(preds).to("cpu")
    preds[preds < 0] = -1
    preds[preds >= 0] = 1

    return [1, -1] * len(data), preds.tolist()


def train_edgebank(model: EdgeBank, data, sampler):
    preds = []
    for pe in data:
        sampler.sample()
        preds.append(model.predict(pe,update=True))

def valid_edgebank(model: EdgeBank, data, sampler):
    preds = []
    for pe in data:
        ne = sampler.sample()
        preds.append(model.predict(ne))
        preds.append(model.predict(pe,update=True))
    return [-1,1]*len(data),preds


scores = []

for i in range(N_RUNS):
    if MODEL_TYPE == 'LDTGN':
        model = LDTGN(
            normalization=NORMALIZATION,
            device=DEVICE,
            use_split=USE_SPLIT
        )
        train = train_ldtgn
        valid = valid_ldtgn
    else:
        train=train_edgebank
        valid=valid_edgebank
        if EDGE_BANK_TYPE == 'unlimited':
            model = EdgeBankUnlimited()
        elif EDGE_BANK_TYPE == 'tw':
            model = EdgeBankTW(tw=WINDOW_SIZE)
        else:
            model = EdgeBankThresh(threshold=THRESHOLD)
    if INDUCTIVE:
        training_data, validation_data = make_data_inductive(i)
        validation_sources = list(set([s for s, _, _ in validation_data]))
        validation_destinations = list(set([d for _, d, _ in validation_data]))
    else:
        training_data, validation_data = training_data_original, validation_data_original
        validation_sources = list(set([s for s, _, _ in training_data + validation_data]))
        validation_destinations = list(set([d for _, d, _ in training_data + validation_data]))
    train_sources = list(set([s for s, _, _ in training_data]))
    train_destinations = list(set([d for _, d, _ in training_data]))

    sampler_train = RandomNegativeSampler(training_data, train_sources, train_destinations)
    sampler_validation = RandomNegativeSampler(validation_data, validation_sources, validation_destinations)
    start = time.time()
    train(model, training_data, sampler_train)
    labels, pred_labels = valid(model, validation_data, sampler_validation)
    end = time.time()
    score = average_precision_score(labels, pred_labels)
    scores.append(score)
    print(f"running time: {end - start}")
    print(f"Average precision: {score}")
    fileObj = open(f'trained_ol_{DATASET}_{i}.pkl', 'wb')
    pickle.dump(model, fileObj)
    fileObj.close()

scores = np.array(scores) * 100
print(f"average AP score: {'{:.2f}'.format(scores.mean())}±{'{:.2f}'.format(scores.std())}")
