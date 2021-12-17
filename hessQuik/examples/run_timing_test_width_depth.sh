#!/bin/sh

num_input=11    # powers of 2 from 2^0 to 2^(num_input - 1)
num_output=1    # powers of 2 from 2^0 to 2^(num_output - 1)
num_examples=1  # powers of 10 from 10^0 to 10^(num_examples - 1)
num_trials=10   # number of trials for each setting

for width in 8 16 32 64
do
  for depth in 2 4 8 16
  do
    echo $width
    echo $depth
    python run_timing_test.py --num-input $num_input --num-output $num_output --num-examples $num_examples --num-trials $num_trials --network-type hessQuik --save --width $width --depth $depth
  done
done
