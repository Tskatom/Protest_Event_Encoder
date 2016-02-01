#!/bin/bash

# preprocess the data 

# generate vocab from training dataset
home=/home/ubuntu
prep_exe=${home}/workspace/Protest_Event_Encoder/util/prepText
text_tool=${home}/workspace/Protest_Event_Encoder/util/tools.py
model_exe=${home}/workspace/Protest_Event_Encoder/CNN_Sen_max.py
options="LowerCase UTF8 RemoveNumbers"
max_num=100000
min_word_count=10
word_dm=50

echo Generating vocabulary for training data ... \n
vocab_fn=data/spanish_protest.trn-${max_num}.vocab
$prep_exe gen_vocab input_fn=./data/tokens.lst vocab_fn=$vocab_fn max_vocab_size=$max_num \
    min_word_count=$min_word_count $options WriteCount

echo Construct two set of vocabulary embedding: ramdom and pretrained \n
vec_trained_fn=${home}/workspace/ssd/data/trained_w2v_${word_dm}.pkl
vec_random_fn=${home}/workspace/ssd/data/random_w2v_${word_dm}.pkl
pretrained_fn=${home}/workspace/ssd/data/${word_dm}d_vectors.txt
python $text_tool --task gen_emb --vocab_fn $vocab_fn --vec_random_fn $vec_random_fn --vec_trained_fn $vec_trained_fn --pretrained_fn $pretrained_fn --emb_dm $word_dm

echo Start Training the model
for i in `seq 0 4`;
do
    exp_name=doc_type_max_fold_${i}
    log_fn=./log/${exp_name}.log
    perf_fn=./results/
    param_fn=./doc_type_max_param.json
    python $model_exe --prefix ${home}/workspace/Protest_Event_Encoder/data/single_label/${i}/spanish_protest --sufix type_cat --word2vec $vec_trained_fn --dict_fn ${home}/workspace/Protest_Event_Encoder/data/type_cat.dic --max_sens 30 --max_words 70 --padding 2 --exp_name $exp_name --max_iter 100 --batch_size 100 --log_fn $log_fn --perf_fn $perf_fn --param_fn $param_fn

done
