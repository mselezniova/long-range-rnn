# How Long Does Infinite Width Last? Signal Propagation in Long-Range Linear Recurrences

This repository contains the code to reproduce the experiments in [How Long Does Infinite Width Last? Signal Propagation in Long-Range Linear Recurrences](https://arxiv.org/abs/2605.05113) by [Mariia Seleznova](https://mselezniova.github.io) (NeurIPS 2026).

It includes:

- [`experiments.py`](experiments.py): a script for running the Monte Carlo experiments. Run `python experiments.py --help` for the available options.
- [`figures.ipynb`](figures.ipynb): a notebook for reproducing the paper's experimental figures from the saved results in [`data/`](data/). Run all cells to generate the plots.

Install the Python dependencies with `pip install -r requirements.txt`. The figures notebook also requires LaTeX and `dvipng`.
