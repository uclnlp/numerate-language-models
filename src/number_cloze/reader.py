from pprint import pprint
from collections import defaultdict
import numpy as np
import os
from src.preproc import vocabulary

RNG = np.random.RandomState(12145)
np.random.seed(56785)


def analyse_dataset(get_inst_gen):
    n_instances = 0
    n_tokens = 0
    n_words = 0
    n_nums = 0
    all_nums = []
    max_len = 0
    for instance in get_inst_gen():
        nums = [x for x in instance['numbers'] if x != '']
        tokens = instance['tokens']
        max_len = max(max_len, len(tokens))
        n_instances += 1
        n_tokens += len(tokens)
        n_nums += len(nums)
        n_words += len(tokens) - len(nums)
        all_nums.extend(nums)
    all_floats = [float(x) for x in all_nums]
    print('#instances: %.0f' % n_instances)
    print('max length: %.0f' % max_len)
    print('#tokens:    %.0f' % n_tokens)
    print('#tokens/instance: %.2f' % (n_tokens/n_instances))
    print('#words/instance:  %.2f' % (n_words/n_instances))
    print('#nums/instance:   %.2f' % (n_nums/n_instances))
    print('perc.words: %.2f' % (n_words/n_tokens*100.0))
    print('perc.nums:  %.2f' % (n_nums/n_tokens*100.0))
    print('min:    %.2f' % min(all_floats))
    print('median: %.2f' % np.median(all_floats))
    print('mean:   %.2f' % np.mean(all_floats))
    print('max:    %.2f' % max(all_floats))
    return n_instances


def load_data(path_to_data='', max_length=None, max_lines=None):
    folds = ['train', 'dev', 'test']
    if max_lines is None:
        max_lines = {}
    if max_length is None:
        max_length = {}
    for fold in folds:
        if fold not in max_lines:
            max_lines[fold] = None
        if fold not in max_length:
            max_length[fold] = None
    #
    get_instance_gen = {}
    for fold in folds:
        get_instance_gen[fold] = get_instance_gen_builder(path_to_data.format(fold=fold),
                                                          max_length=max_length[fold],
                                                          max_lines=max_lines[fold])
    #for instance in get_instance_gen['train'](): pprint(instance)
    return get_instance_gen


def get_instance_gen_builder(filename, max_lines=None, max_length=None):

    def _instance_generator():
        with open(filename, 'r', encoding='utf-8') as fin:
            for line_num, line in enumerate(fin):
                if max_lines is not None and line_num >= max_lines:
                    break
                fields = line.split('\t')
                ids = fields[0].strip()
                tokens = fields[1].strip().split('|')
                numbers = fields[2].strip().split('|')
                r = []
                for num in numbers:
                    p = num.split('.')
                    i = len(p[0])
                    d = len(p[1]) if len(p) > 1 else 0
                    r.append((i, d))  # str to num of digits
                #values = [float(num) if num != '' else None for num in numbers]  # str to float/None
                context_tokens = fields[3].strip().split('|')
                context_numbers = fields[4].strip().split('|')
                #context_values = [float(num) if num != '' else None for num in context_numbers]
                # truncate to maximum length
                if max_length is not None and len(tokens) >= max_length:
                    tokens = tokens[:max_length]
                    numbers = numbers[:max_length]
                #context_toks, context_nums, context_is_nums, toks, nums, is_nums = toks, nums, is_nums, context_toks, context_nums, context_is_nums # FIXME kb completion
                #context_toks, context_nums, context_is_nums = toks, nums, is_nums  # FIXME autoencoder
                instance = {
                    'ids': ids,
                    'enc_tokens': context_tokens,
                    #'enc_values': context_values,
                    'enc_numbers': context_numbers,
                    'tokens': tokens,
                    #'values': values,
                    'numbers': numbers,
                    'round': r
                }
                assert(len(instance['tokens']) == len(instance['numbers']))#==len(instance['values']))
                assert(len(instance['enc_tokens']) == len(instance['enc_numbers']))#==len(instance['enc_values']))
                yield instance
    return _instance_generator


def write_instances(filename, instance_gen):

    with open(filename, 'w', encoding='utf-8') as fout:
        for instance in instance_gen():
            fout.write(str(instance['ids']))
            fout.write('\t')
            fout.write('|'.join(instance['tokens']))
            fout.write('\t')
            fout.write('|'.join(instance['numbers']))
            fout.write('\t')
            fout.write('|'.join(instance['enc_tokens']))
            fout.write('\t')
            fout.write('|'.join(instance['enc_numbers']))
            fout.write('\n')


def which_interval(x, intervals):
    for iii, (low, high) in enumerate(intervals):
        if low <= x <= high:
            return iii
    return iii


def prepare_batch(batch,  # list of instances
                  token_to_index,
                  include_enc=False,
                  ):
    # get matrix dimensions
    float2index = {}
    batch_floats_round = []
    batch_floats_value = []
    batch_floats_ids = []
    batch_size = len(batch)

    dec_max_len, enc_max_len = 0, 0
    for instance in batch:
        # collect lengths
        dec_max_len = max(dec_max_len, len(instance['tokens']))
        enc_max_len = max(enc_max_len, len(instance['enc_tokens']))
        # collect all numbers in batch
        for num, (_, round_digit) in zip(instance['numbers'], instance['round']):
            if num != '':
                idx = token_to_index(num)  # symbolic index
                #print(num, idx)
                num = float(num)
                if num not in float2index:
                    float2index[num] = len(float2index)  # local index
                    batch_floats_value.append(num)
                    batch_floats_ids.append(idx)
                    batch_floats_round.append(int(round_digit))
                    #print(num, round_digit, num - 0.5 * (10.0 ** (-round_digit)), num + 0.5 * (10.0 ** ( -round_digit)))
        if include_enc:
            for num in instance['enc_numbers']:
                if num != '':
                    idx = token_to_index(num)  # symbolic index
                    num = float(num)
                    if num not in float2index:
                        float2index[num] = len(float2index)  # local index
                        batch_floats_value.append(num)
                        batch_floats_ids.append(idx)
                        batch_floats_round.append(0)
    #print('abcd')
    # local float vocab
    #print(list(zip(batch_floats_value, batch_floats_ids)))
    #print(min(batch_floats_value), max(batch_floats_value))
    #print(len(batch_floats), sorted(np.asarray(batch_floats, dtype='float32').tolist()))
    #print(sorted(list(zip(batch_floats, batch_floats_ids)), key=lambda x: (x[1], x[0])))
    assert(len(batch_floats_value) > 0)
    # initialise matrices
    matrices = {
        'batch_floats_round': np.asarray(batch_floats_round, dtype='int32'),  # values
        'batch_floats_value': np.asarray(batch_floats_value, dtype='float32'),  # values
        'batch_floats_ids': np.asarray(batch_floats_ids, dtype='int32'),  # nums/interval ids
        'tokens': np.zeros([batch_size, dec_max_len], dtype='int32'),
        'numbers': np.zeros([batch_size, dec_max_len], dtype='int32'),
    }
    if include_enc:
        matrices['enc_tokens'] = np.zeros([batch_size, enc_max_len], dtype='int32')
        matrices['enc_numbers'] = np.zeros([batch_size, enc_max_len], dtype='int32')

    # populate matrices
    for idx, instance in enumerate(batch):
        for pos, token in enumerate(instance['tokens']):
            matrices['tokens'][idx, pos] = token_to_index(token)
            if instance['numbers'][pos] != '':
                num = float(instance['numbers'][pos])
                #num = np.round(num, n_decimals)
                matrices['numbers'][idx, pos] = float2index[num]
        if include_enc:
            for pos, token in enumerate(instance['enc_tokens']):
                matrices['enc_tokens'][idx, pos] = token_to_index(token)
                if instance['enc_numbers'][pos] != '':
                    num = float(instance['enc_numbers'][pos])
                    #num = np.round(num, n_decimals)
                    matrices['enc_numbers'][idx, pos] = float2index[num]
    #pprint(matrices)
    return matrices


def get_batch_gen_builder(get_instance_gen, batch_size=64, prepare=None):
    def _batch_generator():
        batch = []
        for instance in get_instance_gen():
            if len(batch) == batch_size:
                if prepare:
                    batch = prepare(batch)
                yield batch
                batch = []
            batch.append(instance)
        if batch:
            if prepare:
                batch = prepare(batch)
            yield batch
    return _batch_generator


def build_to_feed_dict(placeholders):
    def _to_feed_dict(batch):
        _batch = {placeholders[k]: v for k, v in batch.items()}
        return _batch
    return _to_feed_dict


def main():
    get_inst_gen = get_instance_gen_builder('../../experiments/clinical/data/train_bucketed.strings_to_floats')
    #get_inst_gen = get_instance_gen_builder('../../experiments/arxmliv/data/train_bucketed.strings_to_floats')
    counts = defaultdict(int)
    for instance in get_inst_gen():
        for num, (_, d) in zip(instance['numbers'], instance['round']):
            if num != '':
                #print(num, d)
                counts[d] += 1
    print(counts)

if __name__ == '__main__':
    main()
