#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --nodelist=n-1,n-2,n-4
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
#SBATCH --array=0-41

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="260126"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    connectome_schaefer400/cls
    brainlm_vitmae_111m/patch
    brain_jepa_vitb_ep300/patch
    brain_harmonix_f/patch
    brain_semantoks/cls
    flat_mae_base_patch16_2/cls
    swift/patch
)

datasets=(
    abide_dx
    adhd200_dx
    ppmi_dx
    aabc_age
    aabc_sex
    hcpya_rest1lr_gender
)

# 7 models x 6 datasets

num_datasets=${#datasets[@]}
configid=$(($SLURM_ARRAY_TASK_ID / $num_datasets))
datasetid=$(($SLURM_ARRAY_TASK_ID % $num_datasets))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)

dataset=${datasets[datasetid]}

base_config="${EXP_DIR}/logistic_loop.yaml"
overrides=""

name="eval_logistic_loop/${dataset}__${model}__${repr}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

notes="logistic loop eval sweep ${EXP_NAME} (${dataset} ${model} ${repr})"

uv run --no-sync python -W ignore -m fmri_fm_eval.main_logistic_loop \
    $model \
    $repr \
    $dataset \
    --config \
    $base_config \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
