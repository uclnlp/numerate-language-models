import numpy as np
from collections import OrderedDict
from pprint import pprint

import matplotlib.pyplot as plt

class Aggregator(object):

    def __init__(self):
        self.nom = []
        self.denom = []

    def aggregate(self, nom, denom):
        self.nom.append(nom)
        self.denom.append(denom)

    def result(self):
        nom = np.sum(self.nom)
        denom = np.sum(self.denom)
        return nom/denom


class Averager(object):

    def __init__(self):
        self.nom = []

    def aggregate(self, nom):
        self.nom.append(nom)

    def result(self):
        return np.mean(self.nom)

############################################

class Modes(object):

    def __init__(self):
        self.modes = []
        self.values = []
        self.rounds = []

    def aggregate(self, modes, values, rounds):
        self.modes.extend(modes)
        self.values.extend(values)
        self.rounds.extend(rounds)

    def print_result(self, min_occurrences=5):

        num2modes = {}
        for m, v, r in zip(self.modes, self.values, self.rounds):
            r = int(np.round(r, 0))
            num = np.round(v, r)
            num = ("{:."+str(r)+"f}").format(num)
            m = np.asarray(m)
            if num in num2modes:
                num2modes[num] += m
            else:
                num2modes[num] = m

        num2modes = {k: (v/np.sum(v)).tolist() for k,v in num2modes.items() if np.sum(v) >= min_occurrences}
        modes = []
        for iii in range(3):
            modes.append(sorted(num2modes.items(), key=lambda x: x[1][iii], reverse=True))

        for iii, mode in enumerate(modes):
            print('outmode', iii)
            pprint(mode[:50])

############################################

class Numeric(object):

    def __init__(self):
        self.preds = []
        self.golds = []
        self.partition = []

    def aggregate(self, preds, golds, is_iv):
        self.preds.extend(preds)
        self.golds.extend(golds)
        self.partition.extend(is_iv)

    def result(self):
        partition = np.asarray(self.partition)
        _preds = np.asarray(self.preds)
        _golds = np.asarray(self.golds)

        _preds = {
            'tot': _preds,
            'iv': _preds[partition == 0],
            'oov': _preds[partition == 1]
        }
        _golds = {
            'tot': _golds,
            'iv': _golds[partition == 0],
            'oov': _golds[partition == 1]
        }
        results = OrderedDict()
        for k in ['tot', 'iv', 'oov']:
            preds = _preds[k]
            golds = _golds[k]
            results['n_' + k] = float(preds.shape[0])
            results['pred_std_' + k] = np.std(preds)
            #results['pred_std_' + k] = np.sqrt(np.mean([x for x in abs(preds - preds.mean())**2 if np.isfinite(x)]))
            #
            ae = np.abs(preds - golds)
            ape = get_ape(preds, golds)
            sape = get_sape(preds, golds)
            mse = np.mean(np.nan_to_num(np.square(ae)))
            #mse = np.mean([x for x in np.square(ae).tolist() if np.isfinite(x)])
            '''
            sae = np.square(ae)
            i_inf = np.isinf(sae)
            sae[i_inf] = 0.0
            mse = np.mean(sae) + np.nan_to_num(np.inf)*(np.sum(i_inf)/float(preds.shape[0]))
            '''
            #
            #results['mse_' + k]= mse,
            results['rmse_' + k] = np.sqrt(mse)
            results['mae_' + k] = np.mean(ae)
            results['medae_' + k] = np.median(ae)
            results['mape_' + k] = np.mean(ape) * 100.0
            results['smape_' + k] = np.mean(sape) * 100.0
            results['mdape_' + k] = np.median(ape) * 100.0
            results['Q_' + k] = np.mean(get_accuracy_ratio(preds, golds)) * 100.0
            results['overestimate_'+k] = np.mean(preds > golds) * 100.0
            #results['smedape_' + k] = np.median(sape) * 100.0
            #
            '''
            hist, bins = np.histogram(np.log10(np.maximum(ape * 10, 1e-18)),
                                      bins=np.log10(np.asarray([1e-16] + np.logspace(-10, 10, 21).tolist())),
                                      normed=True)
            print(bins)
            print(hist)
            plt.bar(np.arange(len(hist.tolist())), hist*100)
            plt.title('%.2f' % (np.mean(ape) * 100.0))
            plt.show()
            '''

        return results

def get_ape(preds, golds):
    indices = np.nonzero(golds)
    golds = golds[indices]
    preds = preds[indices]
    return np.abs(golds - preds)/np.abs(golds)


def get_accuracy_ratio(preds, golds):
    indices = np.nonzero(golds)
    golds = golds[indices]
    preds = preds[indices]
    return np.abs(preds)/np.abs(golds)


def get_sape(preds, golds):
    """
    smape(x,y) = |x-y| / (|x|+|y|)  (=0 for x=y=0)
    dissimilarity metric in [0,1] (0: similar, 1: dissimilar)
    for const a!=0:
    b->+-inf => metric->1, i.e. extreme numbers are maximally dissimilar
    b=0 => metric=1 [for a=0, b=0 => metric=0, i.e. 0 is only similar to itself]
    sign(a)!=sign(b) => metric=1, i.e. numbers with opposite signs are always dissimilar
    b->a => metric->0, i.e. any number is similar to itself
    Sample values for ratio=a/b :
    ratio<0   => metric=1, maximum dissimilarity
    ratio->0  => metric->1, maximum dissimilarity
    ratio=1/4 => metric=0.6
    ratio=1/3 => metric=0.5, the halfway point
    ratio=1/2 => metric=0.3333333
    ratio=2/3 => metric=0.2
    ratio=1   => metric=0.0, maximum similarity
    then same values for inverse ratios, e.g. ratio=4 => metric=0.6
    """
    nom = np.abs(preds-golds)
    denom = np.abs(preds)+np.abs(golds)
    denom[denom == 0] = 1.0  # pred = gold = 0 => sape=0.0
    return nom/denom


#####

class Progress(object):
    def __init__(self, time, total_instances):
        self.start_time = time
        self.total_instances = total_instances
        self.seen_instances = 0.0
        self.seen_tokens = 0.0

    def update(self, n_tokens, n_instances):
        self.seen_tokens += n_tokens
        self.seen_instances += n_instances

    def result(self, time):
        if self.total_instances:
            spent_time = time - self.start_time
            token_rate = self.seen_tokens / spent_time
            secs_remaining = (self.total_instances - self.seen_instances) * spent_time / self.seen_instances
        else:
            token_rate = 0.0
            secs_remaining = 0.0
        return token_rate, secs_remaining
