"""
Convert xml paragraphs to text
"""
from __future__ import print_function
import os
from lxml import etree
import json
from pprint import pprint
from src.dataset.latexml import util_lxml

NS = {'ns': "http://dlmf.nist.gov/LaTeXML"}


# ------------------
# Text preprocessing
# ------------------

def process_paragraph_xml(paragraph_xml, parser):
    # parse xml
    paragraph = etree.fromstring(paragraph_xml, parser=parser)
    # FIXME method='text' only gets text (and tail) of xml elements (does not include tags, e.g. labelrefs, bibrefs)
    paragraph = util_lxml.get_only_text(paragraph).strip()
    return paragraph


# ------------------------------------------------------------------------------------------------------------

def main(root_dir='../../../experiments/arxmliv'):
    input_file_pattern = os.path.join(root_dir, 'data/{fold}_xml.json')
    output_file_pattern = os.path.join(root_dir, 'data/{fold}.json')
    print('Input files:')
    print('\t', input_file_pattern)
    print('Output files:')
    print('\t', output_file_pattern)

    parser = etree.XMLParser(encoding='utf-8',)
    folds = ['train', 'dev', 'test']
    #
    for fold in folds:
        # read
        input_file_name = input_file_pattern.format(fold=fold)
        with open(input_file_name, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        # process
        for idx, datapoint in enumerate(data):
            paragraph_xml = datapoint['paragraph']
            # process paragraph
            paragraph_str = process_paragraph_xml(paragraph_xml, parser)
            #print(paragraph_str)
            data[idx]['paragraph'] = paragraph_str
        # write
        output_file_name = output_file_pattern.format(fold=fold)
        with open(output_file_name, 'w', encoding='utf-8') as fout:
            json.dump(data, fout)


if __name__ == '__main__':
    main()
    print('Done!')
