#!/bin/bash

# Usage:
# sh script.sh nproc_per_node master_port run_name split exp method dataset

# Arguments
nproc_per_node=$1
master_port=$2
run_name=$3
project_name=$4
split=$5
exp=$6
method=$7
dataset=$8

# Default values if not provided
config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

# Create directories as needed
mkdir -p $save_path

# Execute the Python script with distributed training configuration
python -m torch.distributed.launch \
    --nproc_per_node=$nproc_per_node \
    --master_addr=localhost \
    --master_port=$master_port \
    ${method}.py \
    --projectname $project_name --runname $run_name \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $master_port 2>&1 | tee $save_path/out.log
