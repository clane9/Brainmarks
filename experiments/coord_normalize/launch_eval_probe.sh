#!/usr/bin/env bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=infinite
#SBATCH --partition=main
#SBATCH --nodelist=n-4
#SBATCH --output=slurms/slurm-%A_%a.out
#SBATCH --account=training
# #SBATCH --array=0-7
# #SBATCH --array=3,6,7
#SBATCH --array=4,9

set -euo pipefail

export OMP_NUM_THREADS=8

ROOT="/data/connor/fmri-fm-eval"
cd $ROOT

# export all env variables
set -a
source .env.medarc.r2
set +a

EXP_NAME="coord_normalize"
EXP_DIR="experiments/${EXP_NAME}"
OUT_DIR="${EXP_DIR}/output"

configs=(
    brainlm_vitmae_111m/patch/attn
    brain_jepa_vitb_ep300/patch/attn
    swift/patch/attn
    neurostorm/patch/attn
    brain_harmonix_f/patch/attn
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

dataset=${datasets[datasetid]}

overrides="batch_size=32 accum_iter=4 model_kwargs.coord_normalize=true"

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
