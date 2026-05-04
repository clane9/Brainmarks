#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-6

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="flops_benchmark"
EXP_DIR="experiments/${EXP_NAME}"

configs=(
    brainlm_vitmae_111m/32
    brain_jepa_vitb_ep300/32
    brain_harmonix_f/32
    brain_semantoks/32
    swift/32
    neurostorm/32
    flat_mae_base_patch16_2/32
)

dataset=hcpya_task21

configid=$SLURM_ARRAY_TASK_ID

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
bs=$(echo $config | cut -d / -f 2)

overrides="batch_size=${bs}"

for ii in {0..4}; do
    uv run --no-sync python -m fmri_fm_eval.main_flops \
        $model \
        $dataset \
        --overrides \
        $overrides
done
