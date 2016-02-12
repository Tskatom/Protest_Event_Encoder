#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
prep_exe=../../util/prepText
text_tool=../../util/tools.py
model_exe=../../CNN_no_validation.py
options="LowerCase UTF8 RemoveNumbers"
max_num=100000
min_word_count=5
word_dm=100

echo Generating vocabulary for training data ... \n
vocab_fn=data/spanish_protest.trn-${max_num}.vocab
$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
    min_word_count=$min_word_count $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=./data/trained_w2v_${word_dm}.pkl
vec_random_fn=./data/random_w2v_${word_dm}.pkl
pretrained_fn=/home/ubuntu/workspace/ssd/data/${word_dm}d_vectors_w2v.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

echo Start Training the model
for dim in 200 100;
do
exp_name=word_pop_d100_B50_F${dim}_N23_fold_0
log_fn=./log/${exp_name}.log
perf_fn=./results/
param_fn=./pop_param_d${dim}.json
python $model_exe --prefix ../../data/single_label/0/spanish_protest --sufix pop_cat --word2vec $vec_trained_fn --dict_fn ../../data/pop_cat.dic --max_len 700 --padding 2 --exp_name $exp_name --max_iter 70 --batch_size 50 --log_fn $log_fn --perf_fn $perf_fn --param_fn $param_fn
done

