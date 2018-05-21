Numerate Language Models
========================

Numeracy is the ability to deal with numbers and numerals.
This project investigates various strategies for language models to predict numerals.
Models are trained and tested on a clinical and a scientific dataset.

## Reference
Georgios Spithourakis and Sebastian Riedel. Numeracy for Language Models: Evaluating and Improving their Ability to Predict Numbers, ACL 2018


Dependencies
==============
pip install beautifulsoup4

pip install lxml (for windows, download from http://www.lfd.uci.edu/~gohlke/pythonlibs/)

pip install spacy  (might additionally need: conda install libgcc)

python -m spacy download en

Glove embeddings from:  https://nlp.stanford.edu/projects/glove/


Download and Preprocess Data
============================

## download and extract data

latexml.download_arxmliv.py

[dataset].extract_to_json.py

latexml.xml_to_text.py

## process text

json_to_tsv.py

tsv_to_annotated.py

## process tables

[dataset].tables_to_processed.py

tables_processed_to_annotated.py

## build vocab and bucket

dataset.common.join_all.py

preproc.build_vocab.py

preproc.bucketing.py

Train and Test Language Models
==============================

python lm_jtr.py

--data [clinical|arxmliv]

--train number_of_epochs

--batch batch_size

--config [a1|a2|a3|a4|b1|b2|c1]  # strategy for outputting numerals (inferred if model is loaded)

--no-test    # to suppress test-time evaluation

--no-inspect # to suppress diagnostics (plots, intermediata values, etc)

--load a1_2018_02_17_16_50_13_clinical

e.g.

python lm_jtr.py --data arxmliv --no-inspect --no-test --train 500 --batch 50 --config a1  # train model

python lm_jtr.py --data arxmliv --no-inspect --load a1_2018_02_18_11_55_11_arxmliv   # test model

python lm_jtr.py --data arxmliv --no-test --load  a1_2018_02_18_11_55_11_arxmliv   # get plots and other diagnostics


## config options
a1: softmax

a2: softmax+rnn

a3: h-softmax

a4: h-softmax+rnn

b1: d-RNN

b2: MoG

c1: combination



