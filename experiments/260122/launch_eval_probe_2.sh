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
# #SBATCH --array=0-5
# #SBATCH --array=2-4
# #SBATCH --array=8-10
#SBATCH --array=11

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
    brainlm_vitmae_111m/patch/attn
    brain_jepa_vitb_ep300/patch/attn
    brain_semantoks/patch/attn
    brain_harmonix_f/patch/attn
    flat_mae_base_patch16_2/patch/attn
    swift/patch/attn
)

datasets=(
    hcpya_task21
    nsd_cococlip
)

# 6 models x 12 datasets
# 72 runs

num_configs=${#configs[@]}
datasetid=$(($SLURM_ARRAY_TASK_ID / $num_configs))
configid=$(($SLURM_ARRAY_TASK_ID % $num_configs))

config=${configs[configid]}
model=$(echo $config | cut -d / -f 1)
repr=$(echo $config | cut -d / -f 2)
clf=$(echo $config | cut -d / -f 3)

dataset=${datasets[datasetid]}

base_config="${EXP_DIR}/probe.yaml"
overrides="batch_size=32 accum_iter=4"

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
    --config \
    $base_config \
    --overrides \
    output_root="${OUT_DIR}" \
    name="${name}" \
    notes="${notes}" \
    $overrides
