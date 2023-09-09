from sortedcontainers import SortedDict, SortedList, SortedSet
import torch
from torch import nn
import math


class PredictionModule(nn.Module):
    def __init__(self, use_split=False, device="cpu", normalization="static"):
        super(PredictionModule, self).__init__()
        self.device = torch.device(device)
        self.lin_seen_edge = torch.nn.Linear(3, 1).to(self.device)
        self.lin_seen_edge.weight.data.fill_(0.)
        self.lin_seen_edge.bias.data.fill_(0.)
        self.lin_unseen_edge = torch.nn.Linear(2, 1).to(self.device)
        self.lin_unseen_edge.weight.data.fill_(0.)
        self.lin_unseen_edge.bias.data.fill_(0.)
        self.opt = torch.optim.SGD(self.parameters(), lr=0.0001)
        self.use_split = use_split
        self.normalization = normalization
        self.log_C = 15

        self.iter = 0

    def forward(self, x, t):
        x = x.to(self.device)
        x = self.normalize(x, t)
        y = -torch.ones(x.shape[0], 1, device=self.device)

        if not self.use_split:
            y[(x[:, 1] < 1) + (x[:, 2] < 1)] = self.lin_seen_edge(x[(x[:, 1] < 1) + (x[:, 2] < 1)])
        else:
            seen_edges_idx = (x[:, 0] < 1)
            y[seen_edges_idx] = self.lin_seen_edge(x[seen_edges_idx])
            unseen_edges_idx = (x[:, 0] == 1) * ((x[:, 1] < 1) + (x[:, 2] < 1))
            y[unseen_edges_idx] = self.lin_unseen_edge(x[unseen_edges_idx][:, 1:])
        return y

    def backward(self, y, pred):
        y = y.reshape(-1).to(self.device)
        pred = pred.reshape(-1).to(self.device)
        loss = torch.mean((y - pred) ** 2)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def normalize(self, x, t):
        if self.normalization == 'static':
            return torch.log(1 + x) / self.log_C
        else:
            return x / t.unsqueeze(0).T


class MemoryModule:
    def __init__(self, device="cpu", normalization="static"):
        self.device = torch.device(device)
        self.adj_s2d = []
        self.adj_d2s = []
        self.iteration = 1
        self.node_iterations = []
        self.normalization = normalization
        self.C = math.e ** 15

    def __are_new__(self, edge):
        s, d, t = edge
        s_new = len(self.adj_s2d[s]) == 0 if len(self.adj_s2d) > s else True
        d_new = len(self.adj_d2s[d]) == 0 if len(self.adj_d2s) > d else True
        return s_new, d_new

    def get_state(self, edge):
        s, d, t = edge
        s_new, d_new = self.__are_new__(edge)
        ret_val = None
        if s_new and d_new:
            ret_val = [self.iteration if self.normalization == "dynamic" else self.C - 1,
                       self.iteration if self.normalization == "dynamic" else self.C - 1,
                       self.iteration if self.normalization == "dynamic" else self.C - 1]
        if s_new and not d_new:
            ret_val = [self.iteration if self.normalization == "dynamic" else self.C - 1
                , self.iteration if self.normalization == "dynamic" else self.C - 1
                , self.iteration - self.node_iterations[d]]
        if d_new and not s_new:
            ret_val = [self.iteration if self.normalization == "dynamic" else self.C - 1,
                       self.iteration - self.node_iterations[s],
                       self.iteration if self.normalization == "dynamic" else self.C - 1]
        if not s_new and not d_new:
            ret_val = [(self.iteration - self.adj_s2d[s][d]) if d in self.adj_s2d[s] else (
                self.iteration if self.normalization == "dynamic" else self.C - 1),
                       self.iteration - self.node_iterations[s],
                       self.iteration - self.node_iterations[d]]
        return torch.Tensor(ret_val).to(self.device), self.iteration

    def update(self, edge):
        s, d, t = edge
        if len(self.adj_s2d) <= s:
            self.adj_s2d += [SortedDict() for _ in range(s - len(self.adj_s2d) + 1)]
        if len(self.adj_d2s) <= d:
            self.adj_d2s += [SortedDict() for _ in range(d - len(self.adj_d2s) + 1)]
        if len(self.node_iterations) <= max(s, d):
            self.node_iterations += [0 for _ in range(max(s, d) - len(self.node_iterations) + 1)]
        self.adj_s2d[s][d] = self.iteration
        self.adj_d2s[d][s] = self.iteration
        self.node_iterations[s] = self.iteration
        self.node_iterations[d] = self.iteration
        self.iteration += 1


class LDTGN(nn.Module):
    def __init__(self, normalization="static", device="cpu", use_split=False):
        super(LDTGN, self, ).__init__()
        self.prediction_module = PredictionModule(normalization=normalization, device=device, use_split=use_split)
        self.memory_module = MemoryModule(normalization=normalization, device=device)
        self.device = torch.device(device)

    def forward(self, updates, inputs):
        # one update and then a list of inputs
        x = []
        t = []
        for upd, inp in zip(updates, inputs):
            for i in inp:
                state, time = self.memory_module.get_state(i)
                x.append(state)
                t.append(time)
            self.memory_module.update(upd)
        x = torch.stack(x)
        t = torch.tensor(t, device=self.device)
        return x, t, self.prediction_module.forward(x, t)

    def forward_prediction_module(self, x, t):
        return self.prediction_module.forward(x, t)

    def backward(self, y, pred):
        self.prediction_module.train()
        return self.prediction_module.backward(y, pred)

    def eval(self):
        self.prediction_module.eval()

    def train(self, mode: bool = True):
        self.prediction_module.train()
