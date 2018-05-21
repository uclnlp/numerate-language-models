from __future__ import print_function
from lxml import etree
import json
from pprint import pprint
import numpy as np
import os
from src.dataset.latexml import util_lxml
from src.preproc.nlp_proc import NLPAnalyser


NS = {'ns': "http://dlmf.nist.gov/LaTeXML"}

# ------------------
# Table preprocessing
# ------------------

def process_table_xml(table_xml, parser, preprocess):
    # parse xml
    table = etree.fromstring(table_xml, parser=parser)
    # extract label
    label = table.attrib.get('labels', '')  # e.g. 'LABEL:table:...'
    frefnum = table.attrib.get('frefnum', '')  # e.g. 'Table 1'
    # extract caption
    caption = ''
    for e_caption in table.xpath('./ns:caption', namespaces=NS):
        caption = util_lxml.get_only_text(e_caption)
    # extract headers
    headers = []
    for e_header in table.xpath('./ns:tabular/ns:thead/ns:tr/ns:td', namespaces=NS):
        header = util_lxml.get_only_text(e_header)
        header = header.replace('|', '_')
        header = preprocess(header)
        headers.append(header)
    # extract rows
    rows = []
    for e_row in table.xpath('./ns:tabular/ns:tbody/ns:tr', namespaces=NS):
        row = []
        for e_cell in e_row.xpath('./ns:td', namespaces=NS):
            border = e_cell.attrib.get('border', '')  # 'l r t b'
            cell = util_lxml.get_only_text(e_cell)
            cell = cell.replace('|', '_')
            cell = preprocess(cell)
            row.append(cell)
        rows.append(row)
    extracted = {
        #'xml': table_xml,
        'label': label,
        'caption': caption,
        'headers': headers,
        'rows': rows,
    }
    return extracted


def main(root_dir='../../../experiments/arxmliv'):
    input_file_name = os.path.join(root_dir, 'data/tables.json')
    output_file_name = os.path.join(root_dir, 'data/tables_processed.json')
    print('Input files:')
    print('\t', input_file_name)
    print('Output files:')
    print('\t', output_file_name)

    parser = etree.XMLParser(encoding='utf-8',)
    with open(input_file_name, 'r') as fin:
        id2table = json.load(fin)
    process_text = NLPAnalyser().process
    id2processed = {}
    for table_id, table_xml in id2table.items():
        # process table
        processed_table = process_table_xml(table_xml, parser, process_text)
        id2processed[table_id] = processed_table
    # write to file
    with open(output_file_name, 'w') as fout:
        json.dump(id2processed, fout)

if __name__ == '__main__':
    main()
    print('Done!')
