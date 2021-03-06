Sequences of cognitive decline in typical Alzheimer’s disease and posterior cortical atrophy estimated using a novel event-based model of disease progression
=================

This repo contains the code used for the manuscript: [Firth et al., Alzheimer's and Dementia 2020](https://doi.org/10.1002/alz.12083)

This is the Kernel Density Estimation (KDE) Event-Based Model (EBM).

Install the KDE EBM
============
Once inside this directly you can just install with `pip` using

`pip install .`

Scripts relevant to the paper
============
In the folder `kde_ebm_paper` you'll find scripts to generate synthetic data, the figures, and the related analyses in [Firth et al., Alzheimer's and Dementia 2020](https://doi.org/10.1002/alz.12083).

Dependencies
============
- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)

The code depends heavily on NumPy, uses SciPy to calculate some stats and do some optimisation, uses Matplotlib just to do the plotting, and used scikit-learn for Kernel Density Estimation

