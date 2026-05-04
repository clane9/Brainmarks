#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --nodelist=n-1,n-2,n-3,n-4
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=sophont
#SBATCH --array=0-63

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="aabc_extra"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    connectome_schaefer400/cls
    brainlm_vitmae_111m/patch
    brain_jepa_vitb_ep300/patch
    brain_harmonix_f/patch
    swift/patch
    neurostorm/patch
    brain_semantoks/patch
    flat_mae_base_patch16_2/patch
)

datasets=(
    aabc_neo_n
    aabc_neo_e
    aabc_neo_o
    aabc_neo_a
    aabc_neo_c
    aabc_fluid_iq
    aabc_cryst_iq
    aabc_memory
)

# 8 configs x 8 datasets
num_datasets=${#datasets[@]}
configid=$(($SLURM_ARRAY_TASK_ID / $num_datasets))
datasetid=$(($SLURM_ARRAY_TASK_ID % $num_datasets))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)

dataset=${datasets[datasetid]}

overrides="batch_size=2"

name="eval_logistic/${dataset}__${model}__${repr}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

notes="logistic eval sweep ${EXP_NAME} (${dataset} ${model} ${repr})"

uv run --no-sync python -W ignore -m fmri_fm_eval.main_logistic \
    $model \
    $repr \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
