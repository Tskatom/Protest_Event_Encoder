#!/bin/bash

# $Id: $
for i in `seq 1 4`;
do
    python mlp.py --fold $i --sufix pop_cat --dict_fn ./data/pop_cat.dic --n_out 11
done

