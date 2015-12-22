#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
prep_exe=../util/prepText
text_tool=../util/tools.py
options="LowerCase UTF8 RemoveNumbers"
max_num=30000
#min_word_count=5
word_dm=100

echo Generating vocabulary for training data ... \n
vocab_fn=data/spanish_protest.trn-${max_num}.vocab
#$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
#    $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=./data/trained_w2v_${word_dm}.pkl
vec_random_fn=./data/random_w2v_${word_dm}.pkl
pretrained_fn=../data/100d_vectors.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

echo Start Training the model
doc_prefix=spanish_protest



