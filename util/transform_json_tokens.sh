#!/bin/bash

# $Id: $

for i in `seq 0 4`;
do
    inf=/home/weiw/Workspace/Protest_Event_Encoder/data/new_multi_label/$i/event_train.txt.tok
    otf=/home/weiw/Workspace/Protest_Event_Encoder/data/new_multi_label/$i/event_train.tokens

    python transorm_json_tokens.py $inf $otf
done
