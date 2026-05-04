# NSD

Homepage: https://naturalscenesdataset.org/

## 1. Download source data

Download official preprocessed source data from NSD S3 bucket.

```bash
# Stimuli and metadata
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  data/NSD/
aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  data/NSD/

# Design files (trial timing)
aws s3 sync --no-sign-request s3://natural-scenes-dataset "data/NSD/" \
  --exclude "*" \
  --include "nsddata_timeseries/ppdata/subj*/func1mm/design/design_*tsv"

# Timeseries data, surfaces, transforms (only needed for resampling)
aws s3 sync --no-sign-request s3://natural-scenes-dataset "data/NSD" \
    --exclude "*" \
    --include "nsddata_timeseries/ppdata/subj*/func1mm/timeseries/timeseries_*.nii.gz" \
    --include "nsddata/ppdata/subj*/transforms/*" \
    --include "nsddata/freesurfer/*"

# Betas
aws s3 sync --no-sign-request s3://natural-scenes-dataset "data/NSD" \
    --exclude "*" \
    --include "nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf/?h.betas_session??.mgh"
```

## 2. Resample to template space

The NSD timeseries data are distributed in high-resolution native surface space. We resample to `32k_fs_LR` template space (i.e. `fslr64k`).

```bash
export OMP_NUM_THREADS=1

parallel --jobs 64 uv run --no-sync \
    python scripts/resample_nsd.py {} \
    ::: data/NSD/nsddata_timeseries/ppdata/subj*/func1mm/timeseries/timeseries_*.nii.gz
```

To run this step you will also need to install [`wb_command`](https://www.humanconnectome.org/software/get-connectome-workbench) and clone [`HCPpipelines`](https://github.com/Washington-University/HCPpipelines).

The outputs are available at `s3://medarc/fmri-datasets/source/NSD`.

> **TODO**: add resampling to MNI volume space

```bash
export OMP_NUM_THREADS=1

parallel --jobs 64 -a metadata/nsd_path_list.txt \
    uv run --no-sync \
    python scripts/resample_nsd_mni.py \
    --out-dir data/NSD \
    --nsd-dir data/NSD \
    {} \
    2>&1 | tee -a logs/resample_nsd_mni.log
```

## 3. Generate eval datasets

### `nsd_cococlip`

A decoding evaluation dataset for classifying the viewed image category from BOLD responses. The task is 24-way image category classification from 16-second fMRI clips (16 TRs at 1.0s TR). Categories are a subset of COCO object categories assigned to images by CLIP zero-shot assignment. Categories were selected for having sufficient number of unambiguous samples (> 600 samples with CLIP confidence > 0.9).

| Split      | Description                          | Subjects | Sessions        |
|------------|--------------------------------------|----------|-----------------|
| train      | Training set                         | 6        | 184 |
| validation | Held-out subject                     | 1 (subj04) | 30            |
| test       | Held-out subject                     | 1 (subj05) | 30  |
| testid     | In-distribution test (held-out sessions) | 6     | 30 |


#### Filter criteria

1. CLIP classification confidence >0.9 on the viewed image
2. Categories with >600 distinct high-confidence images (24 categories retained)
3. Excluding the `shared1000` image set (shown to all subjects; all remaining images viewed by one subject only)
4. Excluding consecutive trials (to reduce HRF overlap)

#### Pipeline

1. `make_nsd_metadata.py` - Extract run/trial metadata from design files
2. `extract_nsd_clip_embeds.py` - Compute CLIP embeddings and COCO category logits
3. `nsd_curation.ipynb` - Filter trials by confidence/category/overlap criteria
4. `make_nsd_cococlip_arrow.py` - Generate final Arrow dataset
