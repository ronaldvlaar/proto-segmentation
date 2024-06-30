#!/bin/bash
# Example script for Cityscapes baseline, ReLu+G-softmax/DINOv2

source ./venv/bin/activate
SOURCE_DATA_PATH="./data/cityscapes"
DATA_PATH="./data/cityscapes" 
LOG_DIR="./logs"

export SOURC_DATA_PATH
export DATA_PATH
export LOG_DIR

# train
python -m segmentation.train cityscapes_kld_imnet_baseline baseline-imnet-cityscapes &
python -m segmentation.train cityscapes_kld_imnet_dino dino-imnet-cityscapes &
python -m segmentation.train cityscapes_kld_imnet_gsoft gsoft-imnet-cityscapes &
wait

# prune
python -m segmentation.run_pruning cityscapes_kld_imnet_baseline baseline-imnet-cityscapes &
python -m segmentation.run_pruning cityscapes_kld_imnet_dino dino-imnet-cityscapes &
python -m segmentation.run_pruning cityscapes_kld_imnet_gsoft gsoft-imnet-cityscapes &
wait

# finetune after pruning
python -m segmentation.train cityscapes_kld_imnet_baseline baseline-imnet-cityscapes --pruned &
python -m segmentation.train cityscapes_kld_imnet_dino dino-imnet-cityscapes --pruned &
python -m segmentation.train cityscapes_kld_imnet_gsoft gsoft-imnet-cityscapes --pruned &
wait

# evaluate 
python -m segmentation.eval_valid baseline-imnet-cityscapes pruned &
python -m segmentation.eval_valid dino-imnet-cityscapes pruned &
python -m segmentation.eval_valid gsoft-imnet-cityscapes pruned &
wait

# evaluate with crf post-processing
python -m segmentation.eval_valid baseline-imnet-cityscapes pruned -c &
python -m segmentation.eval_valid dino-imnet-cityscapes pruned -c &
python -m segmentation.eval_valid gsoft-imnet-cityscapes pruned -c &
wait 

# create visual inspection plots
python -m analysis.activation baseline-imnet-cityscapes pruned &
python -m analysis.activation dino-imnet-cityscapes pruned &
python -m analysis.activation gsoft-imnet-cityscapes pruned &
wait

echo done
