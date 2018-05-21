from collections import defaultdict
import numpy as np
import re
from pprint import pprint


def get_class_counts(n_vocab, word2counts, token_to_index):
    # gather counts
    index2counts = defaultdict(int)
    for w in word2counts.keys():
        index2counts[token_to_index(w)] += 1
    # populate array
    class_counts = [0] * n_vocab
    for idx, counts in index2counts.items():
        class_counts[idx] += counts
    return class_counts


def get_unigram_counts(n_vocab, word2counts, token_to_index):
    unigram_counts = [0] * n_vocab
    for w, counts in word2counts.items():
        unigram_counts[token_to_index(w)] += counts
    return unigram_counts


def get_word2counts(tokens):
    word2counts = defaultdict(int)
    for token in tokens:
        word2counts[token] += 1
    return word2counts


def to_float(word, allow_inf=False):
    num = None
    try:
        num = float(word.replace(',', ''))
        if not allow_inf and not np.isfinite(num):
            num = None
    except ValueError:
        pass
    return num


def build_normalise_token(word2index):
    def normalise_token(token):
        if token in word2index:
            return token
        else:
            num = to_float(token)
            if num is None:
                return "<UNK>"
            else:
                return "<UNK_NUM>"
            return token
    return normalise_token


def build_token_to_index(word2index, normalise_token):
    def token_to_index(w):
        try:
            return word2index[w]
        except KeyError:
            return word2index[normalise_token(w)]  # normalise token and try again
    return token_to_index


def print_float_analysis(word2count):
    num2count = defaultdict(int)
    for word, count in word2count.items():
        num = to_float(word)
        if num is not None:
            num2count[num] += count
    nums_counts = sorted(num2count.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    #print(nums_counts)
    nums, counts = zip(*nums_counts)
    nums = np.asarray(nums)
    counts = np.asarray(counts)
    n_nums = counts.sum()
    mean = (nums*counts).sum()/n_nums
    # find median
    sorted_nums_counts = sorted(num2count.items(), key=lambda x: x[0])
    sorted_nums, sorted_counts = zip(*sorted_nums_counts)
    median = weighted_median(np.asarray(sorted_nums), np.asarray(sorted_counts))

    print('# unique nums ', len(num2count))
    print('# total nums ', n_nums)
    print('min num   \t', np.min(nums))
    print('mode num  \t', nums[0], '(seen ', counts[0], ' times)')
    print('median num\t', median)
    print('mean num  \t', mean)
    print('max num   \t', np.max(nums))


def weighted_median(sorted_nums, sorted_counts):
    total_count = np.sum(sorted_counts)
    cumsum_mask = np.cumsum(sorted_counts) >= total_count/2.0
    cumsum_mask_reverse = np.cumsum(sorted_counts[::-1])[::-1] >= total_count/2.0
    median_mask = cumsum_mask * cumsum_mask_reverse
    median = np.sum(median_mask * sorted_nums/np.sum(median_mask))
    return median
