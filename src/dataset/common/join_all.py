from __future__ import print_function
import os
import json
from pprint import pprint
import time


def parse_context_line(line):
    #print(line)
    _fields = line.split('\t')
    row = int(_fields[0])
    col = int(_fields[1])
    toks = _fields[4].strip().split('|')
    nums = _fields[5].strip().split('|')
    assert(len(toks) == len(nums))
    return col, toks, nums


def process_context(txt):
    tokens, numbers = [], []
    for line in txt.strip('\n').split('\n')[1:]:
        col, toks, nums = parse_context_line(line)
        tokens.extend(toks)
        numbers.extend(nums)
        tokens.append('_col_%d_' % col)
        numbers.append('')
    return tokens, numbers



def get_instance_gen_builder(fold_file, id2context):
    def _instance_gen():
        with open(fold_file, 'r', encoding='utf-8') as fin:
            fin.readline()  # skip header
            for line in fin:
                fields = line.split('\t')
                idx = fields[0].strip()
                context = fields[2].strip()
                tokens = fields[4].strip().split('|')
                numbers = fields[5].strip().split('|')
                context_id = context.split('/')[-1].replace('.annotated', '')  # TODO
                context_tokens, context_numbers = process_context(id2context[context_id])
                instance = {
                    'ids': idx,
                    'tokens': tokens,
                    'numbers': numbers,
                    'context_tokens': context_tokens,
                    'context_numbers': context_numbers,
                }
                assert(len(instance['tokens']) == len(instance['numbers']))
                assert(len(instance['context_tokens']) == len(instance['context_numbers']))
                yield instance
    return _instance_gen


def main(
        #root_dir='../../../experiments/clinical/'
        root_dir='../../../experiments/arxmliv/'
):
    path_to_annotated = os.path.join(root_dir, 'data/{fold}.annotated')
    path_to_context = os.path.join(root_dir, 'data/tables_annotated.json')
    output_file_pattern = os.path.join(root_dir, 'data/{fold}.joined')
    print('Input files:')
    print('\t', path_to_annotated)
    print('\t', path_to_context)
    print('Output files:')
    print('\t', output_file_pattern)

    # read input data
    with open(path_to_context, 'r') as fin:
        id2context = json.load(fin)

    for fold in ['train', 'test', 'dev']:
        instance_gen = get_instance_gen_builder(path_to_annotated.format(fold=fold), id2context)
        output_file = output_file_pattern.format(fold=fold)
        with open(output_file, 'w', encoding='utf-8') as fout:
            for instance in instance_gen():
                #if fold == 'train' and len(instance['tokens']) > 200:  # skip too long lines
                #    continue
                fout.write(instance['ids'])
                fout.write('\t')
                fout.write('|'.join(instance['tokens']))
                fout.write('\t')
                fout.write('|'.join(instance['numbers']))
                fout.write('\t')
                fout.write('|'.join(instance['context_tokens']))
                fout.write('\t')
                fout.write('|'.join(instance['context_numbers']))
                fout.write('\n')
    print('Done!')

if __name__ == '__main__':
    main()


