A unified framework for evaluating pre-trained fMRI foundation models across diverse neuroimaging datasets and prediction tasks.

## Structure

```
fmri-fm-eval/
├── src/fmri_fm_eval/       # Main package
│   ├── models/             # Foundation model wrappers
│   ├── datasets/           # Dataset implementations
│   ├── config/             # YAML configuration files
│   ├── main_probe.py       # Primary evaluation script
│   └── classifiers.py      # Classification heads (Linear, MLP, AttnPool)
├── datasets/               # Dataset preparation scripts
└── experiments/            # Experiment outputs
```

## Experiments

Experiments are organized under `experiments/`, each with its own set of scripts

```
experiments/
├── 260122/       # Initial evaluation sweep with attentive probe
├── 260126/       # Second evaluation sweep with logistic probe
```
