#!/bin/bash
#SBATCH --job-name=pirlnav
#SBATCH --gpus-per-node=1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --output=slurm_logs/ddpil-eval-%j.out
#SBATCH --error=slurm_logs/ddpil-eval-%j.err


module load MistEnv/2020a cuda/10.2.89 gcc/8.4.0 anaconda3/2019.10 cudnn/7.6.5.32 pybind11/2.6.2\
source activate pirlnav
cd $SCRATCH/pirlnav

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)
export MASTER_ADDR

# cd /srv/flash1/rramrakhya6/spring_2022/pirlnav

# DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_hd"
config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il/ovrl_resnet50/seed_1/"
EVAL_CKPT_PATH_DIR=$1

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL eval"
srun python -u -m run \
--exp-config $config \
--run-type eval \
TENSORBOARD_DIR $TENSORBOARD_DIR \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
NUM_ENVIRONMENTS 16 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.SPLIT "val" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
