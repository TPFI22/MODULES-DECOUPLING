import random


class RandomNegativeSampler:
    def __init__(self, data, sources, destinations, index=-1):
        self.edges = set()
        self.sources = sources
        self.destinations = destinations
        self.data = data
        self.index = index

    def sample(self):
        ns_idx = random.randint(0, len(self.sources) - 1)
        nd_idx = random.randint(0, len(self.destinations) - 1)
        self.index += 1
        while self.sources[ns_idx] == self.data[self.index][0] and self.destinations[nd_idx] == self.data[self.index][
            1]:
            ns_idx = random.randint(0, len(self.sources) - 1)
            nd_idx = random.randint(0, len(self.destinations) - 1)
        return self.sources[ns_idx], self.destinations[nd_idx], self.data[self.index][2]


class RandomNegativeSamplerWindow:
    def __init__(self, data, window=200):
        self.edges = set()
        self.index = 0
        self.window = window
        self.data = data

    def sample(self):
        sources = list(set([s for s, _, _ in self.data[self.index:self.index + window]]))
        destinations = list(set([d for _, d, _ in self.data[self.index:self.index + window]]))
        ns = sources[random.randint(0, len(self.sources))]
        nd = destinations[random.randint(0, len(self.destinations))]
        while ns == self.data[self.index][0] and nd == self.data[self.index][1]:
            ns = sources[random.randint(0, len(self.sources))]
            nd = destinations[random.randint(0, len(self.destinations))]
        self.index += 1
        return ns, nd, self.data[self.index[2]]


class BatchRandomNegativeSampler:
    def __init__(self, data):
        self.data = data
        self.curr_neg_edges = []
        self.starting_index = 0
        self.batch_size = 200

    def sample(self, data):
        if len(self.curr_neg_edges) > 0:
            edge2return = self.curr_neg_edges[0]
            self.curr_neg_edges.pop(0)
            return edge2return

        all_sources = set()
        all_destinations = set()
        data = self.data[self.starting_index:self.starting_index + self.batch_size]
        for s, d, t in data:
            all_sources.add(s)
            all_destinations.add(d)

        all_sources = list(all_sources)
        all_destinations = list(all_destinations)

        neg_sources = random.choices(all_sources, k=len(data))
        neg_destinations = random.choices(all_destinations, k=len(data))

        neg_edges = []
        for i in range(len(data)):
            neg_edges.append((neg_sources[i], neg_destinations[i], data[i][2]))

        self.starting_index += self.batch_size
        self.curr_neg_edges = neg_edges
        edge2return = self.curr_neg_edges[0]
        self.curr_neg_edges.pop(0)

        return edge2return

    def update(self, pe):
        pass
