#!/bin/bash

source activate aw847-wilds

METHOD=nw
CD=0
LR=1e-2
EPISODIC=true

command="python -u train.py \
    --method ${METHOD} \
    --wandb_kwargs project=easy-few-shot-learning name=${METHOD}_cd${CD} \
    --use_wandb \
    --class_dropout ${CD} \
    --fine_tuning_steps 0 \
    --lr ${LR}
    "

if [ $METHOD == "matchingnets" ] || [ $METHOD == "protonets" ] || [ $EPISODIC ]
then
    command+=" --episodic_training"
fi

# Debug mode if -d flag is set
while getopts ":d" opt; do
  case $opt in
    d)
      command+=" --debug_mode"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo $command
$command