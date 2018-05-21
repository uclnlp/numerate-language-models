"""
Read latexml papers
Extract paragraphs (as unparsed xml) with table mentions
Extract tables (as unparsed xml)
Split paragrahps in train/dev/test
Write paragraphs and tables in json
"""
from __future__ import division, absolute_import, print_function, unicode_literals
import os
from lxml import etree
import numpy as np
import re
import json
from pprint import pprint
from xml.sax.saxutils import escape
from collections import defaultdict
from src.dataset.latexml import util_lxml

NS = {'ns': "http://dlmf.nist.gov/LaTeXML"}


def get_domain_ids(path, max_domains=None):
    domain_ids = []
    for domain_id in os.listdir(path):  # e.g. 'path/1206'
        if max_domains is not None and len(domain_ids) >= max_domains:
            break
        domain_ids.append(domain_id)
    return domain_ids


def get_paths_to_xmls(data_dir, domain_id, max_projects=None):
    path_to_domain = os.path.join(data_dir, domain_id)
    proj_id2path_to_xml = {}   # e.g. 'path/1206/1206.6660/filename.xml'
    for proj_id in sorted(os.listdir(path_to_domain)):
        if max_projects is not None and len(proj_id2path_to_xml) >= max_projects:
            break
        path_to_xml = os.path.join(data_dir,
                                   domain_id,
                                   proj_id,
                                   '{proj_id}.noparse.xml'.format(proj_id=proj_id))
        if os.path.isfile(path_to_xml):
            proj_id2path_to_xml[proj_id] = path_to_xml
    return proj_id2path_to_xml


def split_folds(n_data):
    #n_data = len(alist)
    n_train = int(np.floor(0.7*n_data))
    n_test = int(np.floor(0.2*n_data))
    n_dev = n_data - n_train - n_test
    #
    fold_ids = [0]*n_train + [1]*n_dev + [2]*n_test
    return fold_ids


##############

def escape_text(matchobj):
    pre = matchobj.group(1)
    txt = matchobj.group(2)
    post = matchobj.group(3)
    escaped = escape(txt)
    #if txt != escaped: print(matchobj.group(0), txt, escaped)
    #print(matchobj.group(0), txt, escaped)
    return pre + escaped + post


def parse_xml(xml_path, parser, fix=True):
    if not fix:
        doc = etree.parse(xml_path, parser=parser)
    else:
        # read xml
        with open(xml_path, 'r', encoding='utf-8') as xml_file:
            xml_string = xml_file.readlines()
            xml_string = ''.join(xml_string)
        # fix xml string
        #print(xml_string)
        xml_string = re.sub(r'(=")(.*?)(")', escape_text, xml_string, 0, re.IGNORECASE | re.DOTALL)
        xml_string = re.sub(r'(>)(.+?)(<)', escape_text, xml_string, 0, re.IGNORECASE | re.DOTALL)
        #print(xml_string)
        # parse xml string
        doc = etree.fromstring(xml_string.encode('utf-8'), parser=parser)
    return doc



def get_tables(doc):
    tables = []
    for table in doc.xpath('//ns:table', namespaces=NS):
        # check if table should be included
        label = table.attrib.get('labels', '')  # e.g. 'LABEL:table:...'
        if label:
            # get text
            xml_string = util_lxml.get_xml_string(table)
            table = {
                'xml': xml_string,
                'label': label,
            }
            tables.append(table)
    return tables


def get_paragraphs(doc):
    paragraphs = []
    for pnode in doc.xpath('//ns:p[ns:ref[@labelref]]', namespaces=NS):
        # get xml string
        xml_string = util_lxml.get_xml_string(pnode)
        # get refs
        refs = set()
        for ref in pnode.xpath('./ns:ref[@labelref]', namespaces=NS):
            ref = ref.attrib['labelref']
            refs.add(ref)
        paragraph = {
            'xml': xml_string,
            'labelrefs': refs,
        }
        paragraphs.append(paragraph)
    return paragraphs


def analyse_paper(path_to_xml, parser):
    local_paragraphs = []
    # parse xml
    doc = parse_xml(path_to_xml, parser=parser, fix=True)
    local_tables = get_tables(doc)
    if local_tables:
        local_paragraphs = get_paragraphs(doc)
    return local_paragraphs, local_tables

# ------------------------------------------------------------------------------------------------------------


def main(input_dir='../../../arxmliv/',
         output_dir='../../../experiments/arxmliv'):
    output_dir = os.path.join(output_dir, 'data')
    output_fold_file_pattern = os.path.join(output_dir, '{fold}_xml.json')
    output_tables_file = os.path.join(output_dir, 'tables.json')
    print('Input files:')
    print('\t', input_dir)
    print('Output files:')
    print('\t', output_fold_file_pattern)
    print('\t', output_tables_file)

    parser = etree.XMLParser(encoding='utf-8',
                             remove_comments=True,
                             recover=True)  # FIXME original xml contains unescaped characters, nodes might be dropped!
    #
    folds = ['train', 'dev', 'test']
    fold2data = {fold: [] for fold in folds}
    id2table = {}
    used_table_ids = set()
    for domain_id in get_domain_ids(input_dir):#, max_domains=1):    # FIXME max_domains
        proj_id2path_to_xml = get_paths_to_xmls(input_dir, domain_id)#, max_projects=2000)  # FIXME max_projects
        fold_ids = split_folds(len(proj_id2path_to_xml))
        for pair_id, (proj_id, path_to_xml) in enumerate(proj_id2path_to_xml.items()):
            fold = folds[fold_ids[pair_id]]
            print(path_to_xml, '->', fold)
            #
            local_paragraphs, local_tables = analyse_paper(path_to_xml, parser)
            print('Found tables:', len(local_tables))
            print('Found texts:', len(local_paragraphs))
            #
            label2table_id = {}
            for table in local_tables:
                label = table['label']
                table_id = '_'.join([domain_id, proj_id, label.replace(':', '.').replace('/', '.')])
                id2table[table_id] = table['xml']
                label2table_id[label] = table_id
            for pid, paragraph in enumerate(local_paragraphs):
                paragraph_id = '_'.join([domain_id, proj_id, str(pid)])
                table_ids = [label2table_id[ref] for ref in paragraph['labelrefs'] if ref in label2table_id]
                if len(table_ids) == 1:
                    table_id = table_ids[0]
                    used_table_ids.add(table_id)
                    data = {
                        'id': paragraph_id,
                        'paragraph': paragraph['xml'],
                        'table_id': table_id,
                    }
                    fold2data[fold].append(data)
    # keep only referenced tables
    id2table = {idx: table for idx, table in id2table.items() if idx in used_table_ids}
    #
    print('Writing files...')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # text
    for fold in folds:
        n_data = len(fold2data[fold])
        print('{num} texts in {fold}'.format(num=n_data, fold=fold))
        with open(output_fold_file_pattern.format(fold=fold), 'w', encoding='utf-8') as fout:
            json.dump(fold2data[fold], fout)
    # tables
    with open(output_tables_file, 'w', encoding='utf-8') as fout:
        print('{num} tables'.format(num=len(id2table)))
        json.dump(id2table, fout)


if __name__ == '__main__':
    main()
    print('Done!')
