#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
prep_exe=../../util/prepText
text_tool=../../util/tools.py
model_exe=../../MI_cnn_kmax_mix_v4.py
options="LowerCase UTF8 RemoveNumbers"
max_num=100000
min_word_count=5
word_dm=100
echo Generating vocabulary for training data ... \n
vocab_fn=data/protest.trn-${max_num}.vocab
$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
    min_word_count=$min_word_count $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=./data/trained_w2v_${word_dm}.pkl
vec_random_fn=./data/random_w2v_${word_dm}.pkl
pretrained_fn=/home/ubuntu/workspace/ssd/data/100d_vectors_w2v.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

for i in `seq 0 4`;
do
    echo Start Training the model
    exp_name=mi_k_max_v4_mix${i}_balanced
    log_fn=./log/${exp_name}.log
    perf_fn=./results/
    param_fn=./options.json
    python $model_exe --prefix ../../data/balanced_set/${i}/event --sufix event_cat --word2vec $vec_trained_fn --event_fn ../../data/event_cat.dic --option ${param_fn} --exp_name $exp_name
done

