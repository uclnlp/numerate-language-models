from __future__ import print_function
import os
from src.preproc.nlp_proc import NLPAnalyser
from pprint import pprint


def write_annotated_file_from_tsv(tsv_file, annotated_path):
    nlp = NLPAnalyser()
    with open(tsv_file, 'r', encoding='utf-8') as fin:
        fin.readline()  # skip headers
        with open(annotated_path, 'w', encoding='utf-8') as fout:
            titles = ['id',
                      'utterance',
                      'context',
                      'target',
                      # ANNOTATIONS
                      'tokens',
                      'nerValues',]
            line = '\t'.join(titles)
            fout.write(line)
            fout.write('\n')
            for line in fin:
                fields_in = line.strip().split('\t')
                utterance = fields_in[1].replace('|', ' _pipe_ ')
                target_value = fields_in[3]
                # NLP processing
                processed = nlp.process(utterance)
                tokens = processed['tokens']
                ner_values = processed['nerValues']
                assert(len(tokens) == len(ner_values))
                # write line
                fields = [
                    fields_in[0],  # id
                    utterance,
                    fields_in[2],  # context
                    target_value,
                    # annotations
                    '|'.join(tokens),
                    '|'.join(ner_values),
                ]
                #print(fields)
                line = '\t'.join(fields)
                fout.write(line)
                fout.write('\n')


def main(
        #root_dir='../../../experiments/clinical',
        root_dir='../../../experiments/arxmliv',
        ):
    input_pattern = os.path.join(root_dir, 'data/{fold}.tsv')
    output_pattern = os.path.join(root_dir, 'data/{fold}.annotated')
    print('Input files:')
    print('\t', input_pattern)
    print('Output files:')
    print('\t', output_pattern)

    folds = ['train', 'dev', 'test']
    for fold in folds:
        tsv_file = input_pattern.format(fold=fold)
        annotated_path = output_pattern.format(fold=fold)
        write_annotated_file_from_tsv(tsv_file, annotated_path)

    '''
    tsv_file = os.path.join('../../out/wiki/annotated/data', 'test_corrupted2.tsv')
    annotated_path = os.path.join('../../out/wiki/annotated/data', 'test_corrupted.annotated')
    write_annotated_file_from_tsv(tsv_file, annotated_path)
    '''
    print('Done!')

if __name__ == '__main__':
    main()
