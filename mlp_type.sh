#!/bin/bash

# $Id: $
for i in `seq 3 4`;
do
    python mlp.py --fold $i --sufix type_cat --dict_fn ./data/type_cat.dic --n_out 5
done

