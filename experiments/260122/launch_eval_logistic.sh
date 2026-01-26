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
#SBATCH --array=0-95

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="${HOME}/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="260122"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    connectome_schaefer400/cls
    brainlm_vitmae_111m/patch
    brain_jepa_vitb_ep300/patch
    brain_harmonix_f/patch
    brain_semantoks/patch
    flat_mae_base_patch16_2/patch
    swift/patch
    neurostorm_mae_0p5/patch
)

datasets=(
    abide_dx
    abide_age
    adhd200_dx
    adhd200_sex
    adni_ad_vs_cn
    adni_sex
    ppmi_dx
    ppmi_sex
    aabc_sex
    aabc_age
    hcpya_rest1lr_gender
    hcpya_rest1lr_age
)

# 8 models x 12 datasets
# 96 runs

num_datasets=${#datasets[@]}
configid=$(($SLURM_ARRAY_TASK_ID / $num_datasets))
datasetid=$(($SLURM_ARRAY_TASK_ID % $num_datasets))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)

dataset=${datasets[datasetid]}

base_config="${EXP_DIR}/logistic.yaml"
overrides=""

name="eval_logistic/${dataset}__${model}__${repr}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

notes="logistic eval sweep ${EXP_NAME} (${dataset} ${model} ${repr})"

uv run --no-sync python -m fmri_fm_eval.main_logistic \
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
