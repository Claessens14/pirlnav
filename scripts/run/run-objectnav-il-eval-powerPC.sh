#!/bin/bash

config="configs/experiments/il_objectnav.yaml"

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_10k/"
# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
TENSORBOARD_DIR="tb/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
CHECKPOINT_DIR="data/new_checkpoints/objectnav_il/overfitting/ovrl_resnet50/seed_3_wd_zero/"
INFLECTION_COEF=3.234951275740812
EVAL_CKPT_PATH_DIR="data/checkpoints/pretrained"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"

mkdir -p $TENSORBOARD_DIR
mkdir -p $CHECKPOINT_DIR
set -x
    
echo "In ObjectNav IL DDP"

python -u -m run --exp-config $config --run-type train TENSORBOARD_DIR $TENSORBOARD_DIR CHECKPOINT_FOLDER $CHECKPOINT_DIR NUM_UPDATES 20000 NUM_ENVIRONMENTS 2 RL.DDPPO.force_distributed True TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF 

