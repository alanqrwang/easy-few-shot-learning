#!/bin/bash
mkdir -p job_err
mkdir -p job_out

for A in 0
do
  sbatch --export=ALL,A=$A --requeue -p sablab -t 128:00:00 --mem=32G --gres=gpu:1  --job-name=$2$A -e ./job_err/%j-$2$A.err -o ./job_out/%j-$2$A.out $1
done
