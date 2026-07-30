# Contributing

Thanks for your interest in NeuroFlow. This repository accompanies a manuscript,
so `main` is kept deliberately small: it holds only what is needed to reproduce
the reported results.

## Branches

| Branch | Contents |
| --- | --- |
| `main` | The publication artifact: the `neuroflow` package, the five reproducibility scripts, docs, and README assets. |
| `neuroflow_dev` | The full research tree: manuscript sources, exploratory notebooks, the legacy HDF5/TensorFlow workflow, the cascade and phase-attention ablations, and every operational script. |

If your change concerns an ablation or an exploratory analysis, it belongs on
`neuroflow_dev`. Only changes that affect the published pipeline belong on `main`.

## Development setup

```bash
git clone https://github.com/zheng-sk/NeuroFlow.git
cd NeuroFlow
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,segmentation,dev]"
```

Install PyTorch matching your platform and CUDA version first if the default
index does not resolve a suitable build.

## Before opening a pull request

```bash
python -m pytest tests/ -v          # or: python -m unittest discover -s tests
python -m compileall -q neuroflow
```

Please also check that:

- No module manipulates `sys.path`. Everything imports from the installed
  `neuroflow` package.
- No absolute developer path (`/Users/...`, `/home/...`) appears in code, docs,
  or scripts.
- **No subject identifier appears anywhere**, including inside file contents,
  example commands, and test fixtures. Cohort IDs embed acquisition dates. Use
  `subject_001`-style placeholders. Scripts should discover fold names from the
  split directories at runtime rather than hardcoding them.

A quick check for the last two:

```bash
git ls-files | xargs grep -n "/Users/\|/home/[a-z]*/\|[0-9]\{3\}_20[0-9]\{6\}"
```

## Style

Match the surrounding code. The package uses plain `argparse` entry points with
a `main()` function per module, which is what the `pyproject.toml` console
scripts bind to. New pipeline stages should follow the same shape and be added
to the stage table in the README.

## Data

The paired 3T/7T cohort is not redistributed. Tests that need real volumes must
skip cleanly when the data is absent — see `ExporterTimeRangeTests` for the
pattern.
