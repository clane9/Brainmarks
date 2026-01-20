#!/bin/bash

# all target spaces required by different models
spaces=(
    schaefer400
    flat
    mni_cortex
    schaefer400_tians3_buckner7
)

# nb, volume data not currently stored locally
# but remote is fine since the script is not blocked waiting for download
roots=(
    data/NSD
    data/NSD
    s3://medarc/fmri-datasets/source/NSD
    s3://medarc/fmri-datasets/source/NSD
)

OUT_ROOT="s3://medarc/fmri-datasets/eval"

SPACEIDS="2"

log_path="logs/make_nsd_cococlip_arrow.log"

for ii in $SPACEIDS; do
    space=${spaces[ii]}
    root=${roots[ii]}
    uv run python scripts/make_nsd_cococlip_arrow.py \
        --space "${space}" \
        --root "${root}" \
        --out-root "${OUT_ROOT}" \
        2>&1 | tee -a "${log_path}"
done
