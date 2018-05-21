'''
Filter out unwanted instances
Sort (by length)
Shuffle buckets (bucket size = multiple of batch size)
'''
from __future__ import print_function
import os
from pprint import pprint
import numpy as np

RNG = np.random.RandomState(1234)


def get_buckets(max_index,
                bucket_size=32,
                shuffle=True):
    buckets = []
    bucket = []
    for idx in range(max_index):
        bucket.append(idx)
        if len(bucket) == bucket_size:
            buckets.append(bucket)
            bucket = []
    if shuffle:
        RNG.shuffle(buckets)
    if bucket:
        # last bucket
        buckets.append(bucket)
    return buckets


def read_lines(input_file):
    n_all_instances = 0
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = []
        for line in fin:
            n_all_instances += 1
            lines.append(line)
    return lines, n_all_instances


def write_buckets(output_file, lines, buckets):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for bucket in buckets:
            for idx in bucket:
                fout.write(lines[idx])


def main(
        #root_dir='../../experiments/clinical/',
        root_dir='../../experiments/arxmliv/',
):
    input_pattern = os.path.join(root_dir, 'data/{fold}.strings_to_floats')
    output_pattern = os.path.join(root_dir, 'data/{fold}_bucketed.strings_to_floats')
    print('Input file :', input_pattern)
    print('Output file:', output_pattern)
    for fold in ['train', 'test', 'dev']:
        with open(input_pattern.format(fold=fold), 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        print(fold)
        print('#total:', len(lines))
        print('#kept:', len(lines))
        lines = sorted(lines, key=lambda x: len(x.strip('\n').split('\t')[4].split('|')))
        buckets = get_buckets(len(lines), bucket_size=512, shuffle=True)
        # print(buckets)
        write_buckets(output_pattern.format(fold=fold), lines, buckets)


if __name__ == '__main__':
    main()
