

from __future__ import division, absolute_import, print_function, unicode_literals
import os
import numpy as np
import json
from pprint import pprint

SEED = 101


def record_gen(path_to_file, max_records=None):
    with open(path_to_file, 'r', encoding='utf-8') as fin:
        headers = fin.readline().strip('\n').split('\t')
        headers = [h.lower().replace(' ', '_') for h in headers]
        for n_lines, line in enumerate(fin):
            if max_records is not None and n_lines >= max_records:
                break
            fields = line.strip('\n').split('\t')
            record = dict(zip(headers, fields))
            record['id'] = str(n_lines)
            yield record


def analyse_record(record):
    local_table = {}
    local_texts = {}
    text_fields = {'id', 'rid', 'history', 'report'}
    exclude_fields = {'requester', 'myocarditis', 'scan_name', 'sarcoid',
                      'finalised_by', 'analyser', 'bicuspid_ao_valve',
                      'apical_cavity_obliteration', 'hcm', 'study_date'}
    for k, v in record.items():
        if k in exclude_fields:
            continue
        if k in text_fields:
            local_texts[k] = v
        else:
            local_table[k] = v
    return local_texts, local_table

# ------------------------------------------------------------------------------------------------------------


def main(input_file='../../../data/clinical/good_edited/good_edited.csv',
         output_dir='../../../experiments/clinical'):
    output_dir = os.path.join(output_dir, 'data')
    output_fold_file_pattern = os.path.join(output_dir, '{fold}.json')
    output_tables_file = os.path.join(output_dir, 'tables.json')
    print('Input files:')
    print('\t', input_file)
    print('Output files:')
    print('\t', output_fold_file_pattern)
    print('\t', output_tables_file)

    # set up fold splitting
    ratios = {'train': 0.7, 'test': 0.2, 'dev': 0.1}
    rng_split = np.random.RandomState(SEED)
    folds, ratios = zip(*ratios.items())
    fold2data = {fold: [] for fold in folds}
    id2table = {}

    for record in record_gen(input_file, max_records=None):
        local_texts, local_table = analyse_record(record)
        #print(record)
        table_id = 'kb_' + local_texts['id']
        local_texts['table_id'] = table_id
        id2table[table_id] = local_table
        this_fold = folds[rng_split.choice(len(folds), p=ratios)]
        fold2data[this_fold].append(local_texts)
        print(record['id'], '->', this_fold)

    print('Writing files...')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # text
    for fold in folds:
        n_data = len(fold2data[fold])
        print('{num} texts in {fold}'.format(num=n_data, fold=fold))
        with open(output_fold_file_pattern.format(fold=fold), 'w') as fout:
            json.dump(fold2data[fold], fout)
    # tables
    with open(output_tables_file, 'w') as fout:
        print('{num} tables'.format(num=len(id2table)))
        json.dump(id2table, fout)


if __name__ == '__main__':
    main()
    print('Done!')
