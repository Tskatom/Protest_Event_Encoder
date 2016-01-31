#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
prep_exe=../../util/prepText
text_tool=../../util/tools.py
model_exe=../../DLBE_cnn.py
options="LowerCase UTF8 RemoveNumbers"
max_num=100000
min_word_count=10
word_dm=50
k=$1
d=$2
echo Generating vocabulary for training data ... \n
vocab_fn=data/spanish_protest.trn-${max_num}.vocab
$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
    min_word_count=$min_word_count $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=./data/trained_w2v_${word_dm}.pkl
vec_random_fn=./data/random_w2v_${word_dm}.pkl
pretrained_fn=../../data/${word_dm}d_vectors.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

echo Start Training the model
exp_name=DLBE_MLT_w${word_dm}_k${k}_d${d}
log_fn=./log/${exp_name}.log
perf_fn=./results/
param_fn=./DLBE_param_d${d}_N23.json
python $model_exe --prefix ../data/single_label/spanish_protest --sufix_pop pop_cat --sufix_type type_cat --word2vec $vec_trained_fn --dict_pop_fn ../data/pop_cat.dic --dict_type_fn ../data/type_cat.dic --max_sens 30 --max_words 70 --padding 3 --exp_name $exp_name --max_iter 75 --batch_size 100 --log_fn $log_fn --perf_fn $perf_fn --param_fn $param_fn --top_k $k 


