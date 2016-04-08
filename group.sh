#!/bin/bash

# $Id: $
for i in `seq 0 4`;
do
    python group_to_individual_v3.py -fold $i
done

