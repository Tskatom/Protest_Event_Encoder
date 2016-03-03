#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
prep_exe=../../util/prepText
text_tool=../../util/tools.py
model_exe=../../DLBE_Event_Advance.py
options="LowerCase UTF8 RemoveNumbers"
max_num=100000
min_word_count=5
word_dm=100
k=$1
d=100
echo Generating vocabulary for training data ... \n
vocab_fn=data/spanish_protest.trn-${max_num}.vocab
$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
    min_word_count=$min_word_count $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=./data/trained_w2v_${word_dm}.pkl
vec_random_fn=./data/random_w2v_${word_dm}.pkl
#pretrained_fn=../../data/${word_dm}d_vectors.txt
pretrained_fn=/home/ubuntu/workspace/ssd/data/${word_dm}d_vectors_w2v.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

echo Start Training the model
for i in `seq 0 4`;
do
    exp_name=DLBE_Event_fold${i}_k${k}
    log_fn=./log/${exp_name}.log
    perf_fn=./results/
    param_fn=./DLBE_param.json
    python $model_exe --prefix ../../data/new_multi_label/${i}/event --sufix_event event_cat --word2vec $vec_trained_fn --dict_event_fn ../../data/event_cat.dic --max_sens 15 --max_words 70 --padding 2 --exp_name $exp_name --max_iter 30 --batch_size 50 --log_fn $log_fn --perf_fn $perf_fn --param_fn $param_fn --top_k $k --data_type json
done
