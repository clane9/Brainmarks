#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --nice
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --array=0-47%16

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="trait_attn_probe"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    connectome_schaefer400/cls/linear
    identity_schaefer400/patch/mlp
    brainlm_vitmae_111m/patch/attn
    brain_jepa_vitb_ep300/patch/attn
    brain_harmonix_f/patch/attn
    brain_semantoks/patch/attn
    swift/patch/attn
    neurostorm/patch/attn
)

datasets=(
    abide_dx
    adhd200_dx
    adni_ad_vs_cn
    ppmi_dx
    aabc_age
    aabc_sex
)

num_datasets=${#datasets[@]}
datasetid=$(($SLURM_ARRAY_TASK_ID % $num_datasets))
configid=$(($SLURM_ARRAY_TASK_ID / $num_datasets))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

dataset=${datasets[datasetid]}

overrides="batch_size=4 accum_iter=4 base_lr=3e-5 metrics=[acc,f1,bacc] cv_metric=bacc"

name="eval_probe/${dataset}__${model}__${repr}__${clf}"
result="${OUT_DIR}/${name}/eval_table.csv"
if [[ -f $result ]]; then
    echo "result $result exists; skipping"
    exit
fi

notes="probe eval sweep ${EXP_NAME} (${dataset} ${model} ${repr} ${clf})"

uv run --no-sync python -m fmri_fm_eval.main_probe \
    $model \
    $repr \
    $clf \
    $dataset \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
