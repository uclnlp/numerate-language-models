from __future__ import print_function
import json
from pprint import pprint
import numpy as np
import os
from src.preproc.nlp_proc import NLPAnalyser

# ------------------
# Table preprocessing
# ------------------

def process_table_xml(table_xml, preprocess):

    # extract label
    label = ''
    # extract caption
    caption = ''

    prop_val = sorted(table_xml.items(), key=lambda x:x[0])
    prop_val = [(k, v) for k, v in prop_val if v]
    # extract headers
    headers = []
    for e_header in prop_val[0]:
        header = preprocess(e_header)
        headers.append(header)
    # extract rows
    rows = []
    for e_row in prop_val[1:]:
        row = []
        for e_cell in e_row:
            cell = preprocess(e_cell)
            row.append(cell)
        rows.append(row)
    extracted = {
        'label': label,
        'caption': caption,
        'headers': headers,
        'rows': rows,
    }
    return extracted


def main(root_dir='../../../experiments/clinical'):
    input_file_name = os.path.join(root_dir, 'data/tables.json')
    output_file_name = os.path.join(root_dir, 'data/tables_processed.json')
    print('Input files:')
    print('\t', input_file_name)
    print('Output files:')
    print('\t', output_file_name)
    print('Reading tables...')
    with open(input_file_name, 'r') as fin:
        id2table = json.load(fin)
    print('Processing tables...')
    process_text = NLPAnalyser().process
    id2processed = {}
    for table_id, table_xml in id2table.items():
        # process table
        processed_table = process_table_xml(table_xml, process_text)
        print(processed_table)
        id2processed[table_id] = processed_table
    print('Writing tables...')
    # write to file
    with open(output_file_name, 'w') as fout:
        json.dump(id2processed, fout)


if __name__ == '__main__':
    main()
    print('Done!')
