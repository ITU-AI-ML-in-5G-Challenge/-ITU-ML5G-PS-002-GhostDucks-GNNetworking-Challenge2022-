import random
import numpy as np
import networkx as nx


def sample(val):
    if callable(val):
        return val()
    else:
        return val


class Const:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


class RepeatNTimes:
    def __init__(self, values, num_samples_per_value):
        self.values = values
        self.num_samples_per_value = num_samples_per_value
        self.i = 0

    def __call__(self):
        ival = (self.i // self.num_samples_per_value) % len(self.values)
        self.i += 1
        return self.values[ival]


class WeightedChoice:
    def __init__(self, vals, weights=None):
        self.values = [class_from_cfg(v) for v in vals]
        self.weights = weights

    def __call__(self):
        return sample(random.choices(self.values, self.weights)[0])


class IntRangeChoice:
    def __init__(self, a, b, step=None):
        self.min = a
        self.max = b
        self.step = step

    def __call__(self):
        if self.step is None:
            x = random.randint(self.min, self.max)
        else:
            n = (self.max - self.min) // self.step
            x = self.min + random.randint(0, n) * self.step

        return x


def random_with_sum(n: int, s: float = 1):
    """ gen n random numbers that sum to s and minimum is eps """
    x = np.random.rand(n-1)
    x = np.diff(np.concatenate(([0], np.sort(x), [1])))
    return x * s


def random_prob(n: int, prec: int = None, eps: float = 0.0):
    probs = random_with_sum(n, 1 - n * eps) + eps
    if prec is not None:
        probs = probs.round(prec)

    s2 = probs.sum()
    if s2 > 1:
        probs[probs.argmax()] -= s2 - 1
    elif s2 < 1:
        probs[probs.argmin()] += 1 - s2
    return list(probs)


def random_pct(n, eps=0):
    pct = random_with_sum(n, 100 - eps*n).astype(int) + int(eps)
    pct[pct.argmin()] += 100 - sum(pct)
    return pct


class RandomPct:
    def __init__(self, n):
        self.n = n

    def __call__(self):
        return random_pct(self.n, eps=1)


class RandomPktSizeDist:
    def __init__(self, min_size, max_size, min_peaks, max_peaks):
        self.min_size = min_size
        self.max_size = max_size
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks

    def __call__(self):
        npeaks = random.randint(self.min_peaks, self.max_peaks)
        sizes = [random.randint(self.min_size, self.max_size) for _ in range(npeaks)]
        # we run the sampling several times to ensure that ==1 check will be true in the docker
        while True:
            probs = random_prob(npeaks, eps=0.01)
            if sum([float(f'{x}') for x in probs]) == 1:
                break

        ret = {'sizes': sizes, 'probs': probs}
        return ret


class ShortestPathRouting:
    def __init__(self, weights=None):
        self.weights = weights

    def __call__(self, G, pairs=None):
        if self.weights is None:
            weight_fn = None
        elif self.weights == 'bandwidth':
            weight_fn = lambda n1, n2, edge: 1 / edge['bandwidth']
        else:
            raise NotImplemented()

        paths = nx.shortest_path(G, weight=weight_fn)
        return paths


class RandomPathRouting:
    def __init__(self, path_sampler='uniform'):
        self.path_sampler = path_sampler

    def __call__(self, G, pairs=None):
        ret = {}
        if not pairs:
            pairs = [(n1, n2) for n1 in G.nodes for n2 in G.nodes]

        for n1, n2 in pairs:
            if n1 == n2:
                p = [n1]
            else:
                paths = list(nx.all_simple_paths(G, n1, n2))
                if self.path_sampler == 'uniform':
                    p = random.choice(paths)
                elif self.path_sampler == 'by_length':
                    lengths = np.array([len(r) for r in paths])
                    cumlen = lengths.cumsum()
                    s = np.random.rand() * lengths.sum()
                    ix = cumlen.searchsorted(s)
                    p = paths[ix]
                elif self.path_sampler == 'longest_path':
                    lengths = np.array([len(r) for r in paths])
                    # rr = np.where(lengths == lengths.max())
                    # print(rr)
                    imax = np.where(lengths == lengths.max())[0]
                    ix = np.random.choice(imax, 1)[0]
                    p = paths[ix]
                else:
                    raise RuntimeError('bad config')

            ret.setdefault(n1, {})[n2] = p

        return ret


class MultipleRoutingsMix:
    def __init__(self, vals, weights=None):
        self.route_generators = [class_from_cfg(v) for v in vals]
        self.weights = weights

    def __call__(self, G):
        n = G.number_of_nodes()
        # decide which route taken from which route generator
        choice = np.random.choice(len(self.route_generators), p=self.weights, size=(n, n))

        routes = []
        for i, gen in enumerate(self.route_generators):
            pairs = np.stack(np.where((choice == i))).T     # array of src dest pairs for selected generator
            routes.append(gen(G, pairs))

        # mix routes
        ret = {n1: {n2: routes[choice[n1, n2]][n1][n2] for n2 in range(n)} for n1 in range(n)}
        return ret


class LinkBWFromTraffic:
    def __init__(self, vals, assignment):
        self.vals = vals
        self.assignment = assignment
        self.default = min(vals)

    def __call__(self, *args, **kwargs):
        return self.default


def class_from_cfg(cfg, **kwargs):
    if not hasattr(cfg, 'type'):
        return Const(cfg)

    elif cfg.type in (None, 'none'):
        return None

    cls = globals()[cfg.type]
    args = {k: v for k, v in cfg.items() if k != 'type'}
    args.update(kwargs)
    obj = cls(**args)
    return obj

