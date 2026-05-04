#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-3

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="nsd_brainlm_brainjepa"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    "brainlm_vitmae_111m/patch/attn/false"
    "brain_jepa_vitb_ep300/patch/attn/false"
    "brainlm_vitmae_111m/patch/attn/true"
    "brain_jepa_vitb_ep300/patch/attn/true"
)

datasets=(
    nsd_cococlip
)

num_configs=${#configs[@]}
datasetid=$(($SLURM_ARRAY_TASK_ID / $num_configs))
configid=$(($SLURM_ARRAY_TASK_ID % $num_configs))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)
coordnorm=$(echo $config | cut -d / -f 4)

dataset=${datasets[datasetid]}

overrides="batch_size=32 accum_iter=4 model_kwargs.coord_normalize=${coordnorm}"

name="eval/${model}_${coordnorm}/${dataset}__${repr}__${clf}"
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
