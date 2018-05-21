import tensorflow as tf
import numpy as np
from collections import defaultdict
import json
import spacy
from sklearn.manifold import TSNE
from src.number_cloze import reader, ploty
from src.preproc import vocabulary
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt
from time import strftime, gmtime
import random
import os
import sklearn
from pprint import pprint


RNG = np.random.RandomState(1214)
tf.set_random_seed(1234)
np.random.seed(5678)
random.seed(4321)


def _token_gen(get_inst_gen):
    for instance in get_inst_gen():
        for token in instance['tokens']:
            yield token
        #for token in instance['enc_tokens']:
        #    yield token


def stitch_vocab(partitions):
    partition_sizes = []
    word2index = {}
    for part in partitions:
        partition_sizes.append(0)
        for w in part:
            if w not in word2index:
                word2index[w] = len(word2index)
                partition_sizes[-1] += 1
    return word2index, partition_sizes


def inspect_vocab(word2counts, word2index):
    #print('word2counts:')
    #print(sorted(word2counts.items(), key=lambda x: (x[1], x[0]), reverse=True))
    #print('word2index:')
    #print(sorted(word2index.items(), key=lambda x: x[1]))
    print('Total tokens:', sum(word2counts.values()))
    print('#unique words:', len(word2counts))
    print('vocab size:', len(word2index))
    vocabulary.print_float_analysis(word2counts)
    #print('UNK_NUM intervals:')
    #print(intervals)


def get_embedding_matrix(model, word2index, token_to_index):
    hidden_size = model.vector_size
    pretrained = RNG.rand(len(word2index), hidden_size)
    for w in word2index.keys():
        if w in model.wv:
            pretrained[token_to_index(w), :] = model.wv[w]
    return pretrained


def get_glove_matrix(hidden_size, word2index, token_to_index,
                     glove_file='../../wordvecs/glove.6B.{hidden}d.txt'
                     ):
    glove_file = glove_file.format(hidden=hidden_size)
    pretrained = None
    if os.path.isfile(glove_file):
        # load vectors
        nlp = spacy.load('en', parser=False, tagger=False, entity=False)
        nlp.vocab.load_vectors(open(glove_file, 'r', encoding='utf-8'))
        #
        num_emb_pairs = []
        # initialise matrix
        random_init = RNG.rand(len(word2index), hidden_size)
        pretrained = np.zeros((len(word2index), hidden_size))
        counts = np.zeros(len(word2index))
        # populate
        for lex in nlp.vocab:
            w = lex.lower_
            if lex.has_vector:
                idx = token_to_index(w)
                counts[idx] += 1.0
                pretrained[idx, :] = lex.vector
                # embeddings of nums
                num = vocabulary.to_float(w)
                if num is not None and num >= 0.0:
                    num_emb_pairs.append((num, lex.vector))
        # init unseen words to random embeddings
        pretrained[counts == 0.0] = random_init[counts == 0.0]
        # average embeddings of classes
        pretrained = pretrained/np.maximum(counts, 1.0)[:, None]
    return pretrained


def normalise(get_instance_gen, normalise_token):

    def new_inst_gen():
        for instance in get_instance_gen():
            new_inst = {}
            new_inst['ids'] = instance['ids']
            new_inst['tokens'] = ["<SOS>"] + [normalise_token(w) for w in instance['tokens']] + ["<EOS>"]
            new_inst['numbers'] = [''] + instance['numbers'] + ['']
            new_inst['enc_tokens'] = ["<SOS>"] + [normalise_token(w) for w in instance['enc_tokens']] + ["<EOS>"]
            new_inst['enc_numbers'] = [''] + instance['enc_numbers'] + ['']
            yield new_inst

    return new_inst_gen


def extract_top_words(word2counts, max_vocab_size):
    iv_words, iv_nums = [], []
    for word, count in sorted(word2counts.items(), key=lambda x: (x[1], x[0]), reverse=True)[:max_vocab_size]:
        num = vocabulary.to_float(word)
        if num is None:
            iv_words.append(word)
        else:
            iv_nums.append(word)
    iv_nums = sorted(iv_nums, key=lambda x: float(x.replace(',', '')))
    return iv_words, iv_nums


def get_all_nums(instance_gen):
    all_nums = []
    for instance in instance_gen():
        for number in instance['numbers']:
            if number != '':
                all_nums.append(number)
    return all_nums


def build_pretrained(pretrained, unigrams, n_vocab_words):
    unigrams = np.asarray(unigrams)

    pre_words = pretrained.T[:, :n_vocab_words]  # hv
    pre_nums = pretrained.T[:, n_vocab_words:]  # hv

    support_words = unigrams[:n_vocab_words]
    pre_w = np.sum(pre_words * support_words[:, None].T/support_words.sum(), axis=-1, keepdims=True)
    support_nums = unigrams[n_vocab_words:]
    pre_n = np.sum(pre_nums * support_nums[:, None].T/support_nums.sum(), axis=-1, keepdims=True)
    pre_class = np.concatenate([pre_w, pre_n], axis=-1)  # (target=1 for "is number")

    pre2embs = {'class': pre_class,
                'words': pre_words,
                'nums': pre_nums}
    return pre2embs


def main(
        #root_dir='../../experiments/clinical',
        #max_vocab_size=1000,
        root_dir='../../experiments/arxmliv',
        max_vocab_size=5000,
        hidden_sizes=[50, 100, 200, 300],
):
    timestamp = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    input_data = os.path.join(root_dir, 'data/{fold}.joined')
    output_data = os.path.join(root_dir, 'data/{fold}.strings_to_floats')
    vocab_dir = os.path.join(root_dir, 'vocab')

    print('Timestamp:', timestamp)
    print('Loading dataset: ', input_data)
    get_instance_gen = reader.load_data(path_to_data=input_data)

    print('Collecting vocabulary...')
    word2counts = vocabulary.get_word2counts(_token_gen(get_instance_gen['train']))
    iv_words, iv_nums = extract_top_words(word2counts, max_vocab_size)
    print('Collecting numbers...')
    iv_nums = [x.replace(',', '') for x in iv_nums]
    all_nums = get_all_nums(get_instance_gen['train'])
    #print(sorted(oov_floats))
    print('Stitching vocabulary...')
    vocab_partitions = [
        ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + iv_words,
        iv_nums + ["<UNK_NUM>"],
    ]
    word2index, partition_sizes = stitch_vocab(vocab_partitions)
    inspect_vocab(word2counts, word2index)
    normalise_token = vocabulary.build_normalise_token(word2index)
    token_to_index = vocabulary.build_token_to_index(word2index, normalise_token)
    print('Calculating counts...')
    adjusted_counts = vocabulary.get_class_counts(len(word2index), word2counts, token_to_index)
    unigram_counts = vocabulary.get_unigram_counts(len(word2index), word2counts, token_to_index)
    assert(sum(1 for x in unigram_counts if x == 0) <= 4)  # only few words can have zero count (e.g. <PAD>)

    print('Normalising dataset...')
    for fold in ['train', 'dev', 'test']:
        new_inst_gen = normalise(get_instance_gen[fold], normalise_token)
        reader.write_instances(output_data.format(fold=fold), new_inst_gen)


    hidden2glove = {}
    for hidden in hidden_sizes:
        print('Extracting embeddings from glove (d={hidden})...'.format(hidden=hidden))
        hidden2glove[hidden] = get_glove_matrix(hidden, word2index, token_to_index)
        #ploty.plot_tsne(hidden2glove[hidden], list(zip(*sorted(word2index.items(), key=lambda x: x[1])))[0])
    #
    if vocab_dir:
        if not os.path.exists(vocab_dir):
            os.makedirs(vocab_dir)
        # Vocab
        output_file = os.path.join(vocab_dir, 'vocab.json')
        print('Saving vocab to:', output_file)
        store = {'word2counts': word2counts,
                 'word2index': word2index,
                 'unigram_counts': unigram_counts,
                 'adjusted_counts': adjusted_counts,
                 'iv_nums': iv_nums,
                 'all_nums': all_nums,
                 'partition_sizes': partition_sizes}
        #print(store)
        with open(output_file, 'w') as fout:
            json.dump(store, fout)
        # Glove
        for hidden, embs in sorted(hidden2glove.items(), key=lambda x: x[0]):
            if embs is not None:
                pretrained = build_pretrained(embs, unigram_counts, partition_sizes[0])
                for name, e in pretrained.items():
                    output_file = os.path.join(vocab_dir, 'glove_{name}_{hidden}.npy'.format(name=name, hidden=hidden))
                    print('Saving embeddings to:', output_file)
                    np.save(output_file, e)




if __name__ == "__main__":
    main()
    print('Done!')
