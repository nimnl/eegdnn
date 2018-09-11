#!/bin/bash

NUM_EPOCAS=$1

first=1
start=1
train=0
evaluate=1
for((i=start; i < $NUM_EPOCAS+1; i++)); do
	echo "$i"
	python3 AE_Noise_Batch.py -name 20_08_18 $i $first $train $evaluate
	first=0
done
