#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-5

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="semantoks_repeats"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "brain_semantoks/patch/attn/1"
    "brain_semantoks/patch/attn/2"
    "brain_semantoks/patch/attn/3"
)

datasets=(
    hcpya_task21
    nsd_cococlip
)

num_configs=${#configs[@]}
datasetid=$(($SLURM_ARRAY_TASK_ID / $num_configs))
configid=$(($SLURM_ARRAY_TASK_ID % $num_configs))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)
seed=$(echo $config | cut -d / -f 4)

dataset=${datasets[datasetid]}

overrides="model_kwargs.seed=${seed} batch_size=32 accum_iter=4"

name="eval/${model}_${seed}/${dataset}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

uv run --no-sync python -m fmri_fm_eval.main_probe \
    $model \
    $repr \
    $clf \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    $overrides
