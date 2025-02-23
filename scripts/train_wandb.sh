#!/bin/bash

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'ade20k', 'coco']
# method: ['unimatch_v2', 
#          'fixmatch', 
#          'supervised', 
#          'unimatch_v2_wandb', 
#          'unimatch_v2_wandb_wo_FA',
#          'unimatch_v2_wandb_normloss',
#          'unimatch_v2_wandb_wo_FA_normloss',
#          'unimatch_v2_wandb_normloss_gradient',
#          'unimatch_v2_wandb_wo_FA_normloss_gradient',
#          'unimatch_v2_wandb_gradient',
#          'unimatch_v2_wandb_wo_FA_gradient']
# exp: just for specifying the 'save_path' 
# split: ['92', '1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset='pascal'
method='unimatch_v2_wandb_wo_FA_gradient'
exp='dinov2_small'
split='92'
run_name='pascal_wo_FA_92_gradient'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    ${method}.py \
    --projectname Unimatch_RAE --runname $run_name \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log \
    
