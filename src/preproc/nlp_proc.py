from __future__ import print_function
import os
import re
import nltk
from pprint import pprint

RE_NUMBER = re.compile(r"(\d+(,\d{3})*(\.\d+)?)", re.UNICODE)
RE_TOKEN = re.compile(r"((\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))", re.UNICODE)


def normalise_number(token):
    num_str = ''
    m = RE_NUMBER.match(token)
    if m:
        try:
            # remove thousands separator
            num_str = m.group(1).replace(',', '')
            # remove leading zeros
            num_str = num_str.lstrip('0')
            if num_str == '':
                num_str = '0'
            else:
                if num_str[0] == '.':
                    num_str = '0' + num_str
            float(num_str)
        except ValueError:
            num_str = ''
    return num_str


def custom_tokenise(text):
    tokens = []
    for token in RE_TOKEN.findall(text):
        if isinstance(token, tuple):
            token = token[0]
        tokens.append(token)
    return tokens


def custom_ner_tag(tokens):
    value_tag_pairs = []
    for token in tokens:
        num_str = normalise_number(token)
        if num_str:
            pair = (num_str, 'NUMBER')
        else:
            pair = ('', 'O')
        value_tag_pairs.append(pair)
    return value_tag_pairs


class NLPAnalyser(object):
    def __init__(self,
                 fast=True,
                 tolower=True,
                 standard_numbers=True,
                 path_to_stanford='../../stanford',
                 ):
        self.tolower = tolower
        self.standard_numbers = standard_numbers
        if not path_to_stanford or fast:
            self.tokenise = custom_tokenise
            self.ner_tag = custom_ner_tag
        else:
            path_to_postagger_jar = 'stanford-postagger-full-2016-10-31/stanford-postagger.jar'
            path_to_postagger_jar = os.path.join(path_to_stanford, path_to_postagger_jar)
            path_to_ner_jar = 'stanford-ner-2016-10-31/stanford-ner.jar'
            path_to_ner_jar = os.path.join(path_to_stanford, path_to_ner_jar)
            path_to_ner_model = 'stanford-ner-2016-10-31/classifiers/english.conll.4class.distsim.crf.ser.gz'
            path_to_ner_model = os.path.join(path_to_stanford, path_to_ner_model)
            self.tokenise = nltk.tokenize.StanfordTokenizer(path_to_postagger_jar).tokenize
            self.ner_tag = nltk.tag.StanfordNERTagger(path_to_ner_model, path_to_ner_jar).tag
        self.sent_segment = nltk.tokenize.PunktSentenceTokenizer().tokenize

    def process(self, text):
        # tokenise
        tokens = self.tokenise(text)
        # NER tagging
        ner = self.ner_tag(tokens)
        # normalise
        norm_tokens = []
        ner_values, ner_tags = [], []
        for token, (value, tag) in zip(tokens, ner):
            if self.tolower:
                token = token.lower()
            if self.standard_numbers:
                num = normalise_number(token)
                if num:
                    token = num
            norm_tokens.append(token)
            ner_values.append(value)
            ner_tags.append(tag)
        #
        processed = {
            'text': text,
            'tokens': norm_tokens,
            'nerTags': ner_tags,
            'nerValues': ner_values
        }
        return processed


def main():
    nlp = NLPAnalyser()
    text = 'hello! This is a test that should be 99.8% precise! Or not? hmm...'
    sentences = nlp.sent_segment(text)
    print('\n'.join(sentences))
    print('-----')
    for sentence in sentences:
        pprint(nlp.process(sentence))

if __name__ == '__main__':
    main()
