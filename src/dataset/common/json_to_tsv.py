"""
Convert json to tsv file:
id, utterance, context, targetValue
"""

from __future__ import print_function
import os
import json
from pprint import pprint

# ------------------------------------------------------------------------------------------------------------


def get_utterance(datapoint):
    text_fields = [
        'report',  # for clinical
        'paragraph'  # for arxmliv
    ]
    for field in text_fields:
        if field in datapoint:
            return datapoint[field]


def main(
        #rood_dir='../../../experiments/clinical',
        rood_dir='../../../experiments/arxmliv',
        ):
    input_file_pattern = os.path.join(rood_dir, 'data/{fold}.json')
    output_file_pattern = os.path.join(rood_dir, 'data/{fold}.tsv')
    print('Input files:')
    print('\t', input_file_pattern)
    print('Output files:')
    print('\t', output_file_pattern)

    folds = ['train', 'dev', 'test']
    for fold in folds:
        input_file_name = input_file_pattern.format(fold=fold)
        with open(input_file_name, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        output_file_name = output_file_pattern.format(fold=fold)
        with open(output_file_name, 'w', encoding='utf-8') as fout:
            titles = ['id', 'utterance', 'context', 'targetValue']
            line = '\t'.join(titles)
            fout.write(line)
            fout.write('\n')
            for datapoint in data:
                utterance = get_utterance(datapoint)
                idx = datapoint['id']
                target_value = idx
                table_id = datapoint['table_id']
                fields = [
                    idx,
                    utterance,
                    table_id,
                    target_value
                ]
                line = '\t'.join(fields)
                fout.write(line)
                fout.write('\n')

if __name__ == '__main__':
    main()
    print('Done!')
