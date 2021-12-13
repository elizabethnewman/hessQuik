#!/bin/sh

num_input=11
num_output=1
num_examples=1
num_trials=10

for type in hessQuik PytorchAD PytorchHessian
do
  echo $type
  echo "forward"
  python ex_timing_test.py --num-input $num_input --num-output $num_output --num-examples $num_examples --num-trials $num_trials --network-type $type --save
  echo "backward"
  python ex_timing_test.py --num-input $num_input --num-output $num_output --num-examples $num_examples --num-trials $num_trials --network-type $type --reverse-mode --save
done
