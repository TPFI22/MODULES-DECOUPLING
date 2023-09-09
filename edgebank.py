from sortedcontainers import SortedDict
from abc import ABC, abstractmethod


class EdgeBank(ABC):
    def __init__(self):
        self.adj_s2d = []
        self.iteration = 0

    def __update__(self, edge):
        s, d, _ = edge
        if len(self.adj_s2d) <= s:
            self.adj_s2d = self.adj_s2d + [SortedDict([]) for _ in range(s - len(self.adj_s2d))] + [SortedDict([])]
        if d not in self.adj_s2d[s]:
            self.adj_s2d[s][d] = (0, 0)
        self.iteration += 1
        self.adj_s2d[s][d] = (self.iteration, self.adj_s2d[s][d][1] + 1)

    @abstractmethod
    def predict(self, edge, update=False):
        raise NotImplementedError()


class EdgeBankUnlimited(EdgeBank):
    def __init__(self):
        super().__init__()

    def predict(self, edge, update=False):
        s, d, _ = edge
        pred = 1 if s < len(self.adj_s2d) and d in self.adj_s2d[s] else -1
        if update:
            self.__update__(edge)
        return pred


class EdgeBankTW(EdgeBank):
    def __init__(self, tw=100000):
        super().__init__()
        self.tw = tw

    def predict(self, edge, update=False):
        s, d, _ = edge
        pred = 1 if s < len(self.adj_s2d) and d in self.adj_s2d[s] and self.iteration - self.adj_s2d[s][d][
            0] < self.tw else -1
        if update:
            self.__update__(edge)
        return pred


class EdgeBankThresh(EdgeBank):
    def __init__(self, threshold=2):
        super().__init__()
        self.threshold = threshold

    def predict(self, edge, update=False):
        s, d, _ = edge
        pred = 1 if s < len(self.adj_s2d) and d in self.adj_s2d[s] and self.iteration - self.adj_s2d[s][d][
            1] >= self.threshold else -1
        if update:
            self.__update__(edge)
        return pred
