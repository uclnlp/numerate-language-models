from __future__ import print_function
import os
import json
from src.preproc.nlp_proc import NLPAnalyser
from pprint import pprint
import time


def gen_lines_from_row(idx_row, cells):
    for idx_col, cell in enumerate(cells):
        content = cell['text']
        tokens = cell['tokens']
        ner_values = cell['nerValues']
        line_id = content.strip().lower().replace(r'\s+', '_')
        if not line_id:
            line_id = 'null'
        if idx_row < 0:
            line_id = 'fb:row.row.%s' % line_id
        else:
            line_id = 'fb:cell.%s' % line_id

        nums = []
        for value in ner_values:
            if value != '':
                nums.append(value)
        fields = [str(idx_row),
                  str(idx_col),
                  line_id,
                  content,
                  '|'.join(tokens),
                  '|'.join(ner_values),
                  ]
        yield '\t'.join(fields)


def main(
        #root_dir='../../../experiments/clinical',
        root_dir='../../../experiments/arxmliv',
):
    input_file = os.path.join(root_dir, 'data/tables_processed.json')
    output_file = os.path.join(root_dir, 'data/tables_annotated.json')
    print('Input file:', input_file)
    print('Output file:', output_file)
    # read input data
    with open(input_file, 'r') as fin:
        id2tables = json.load(fin)
    titles = ['row', 'col', 'id', 'content',
              # ANNOTATIONS
              'tokens',
              'nerValues',]
    id2annotated = {}
    for table_id, table in id2tables.items():
        # Process TABLES
        # gather table
        tbl = '\t'.join(titles)
        tbl += '\n'
        # table headers
        headers_found = True
        for line in gen_lines_from_row(-1, table['headers']):
            tbl += line
            tbl += '\n'
        # table body
        for idx_row, row in enumerate(table['rows']):
            for line in gen_lines_from_row(idx_row, row):
                tbl += line
                tbl += '\n'
        #if not headers_found:
        #    continue
        id2annotated[table_id] = tbl
    # write to file
    with open(output_file, 'w') as fout:
        json.dump(id2annotated, fout)

if __name__ == '__main__':
    main()
