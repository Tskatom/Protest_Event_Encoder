#!/bin/bash

# $Id: $
# construct ssd folder and download word2vec file
ssd=/home/ubuntu/workspace/ssd/
mkdir -p /home/ubuntu/workspace/ssd/data
scp tskatom@embers4.cs.vt.edu:/home/tskatom/workspace/word2vec/100d_vectors_w2v.txt ${ssd}/data

# download new label data
scp -r tskatom@embers4.cs.vt.edu:/home/tskatom/workspace/Protest_Event_Encoder/data/new_single_label ./data


# install ipython
sudo apt-get install ipython
sudo apt-get install nltk
python util/download_punkt.py

# prepare the data
python ./util/transorm_json_tokens.py
