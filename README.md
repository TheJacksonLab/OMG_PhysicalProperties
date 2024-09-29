# Functional monomer design for synthetically accessible polymers

This repository contains python scripts used to obtain comprehensive monomer-level properties for [synthetically accessible polymers.](https://pubs.acs.org/doi/10.1021/acspolymersau.3c00003)  

<p align="center">
<img src="https://github.com/TheJacksonLab/OMG_PhysicalProperties/blob/main/figure_publication/TOC.png" width="500" height="250">
</p>

## Set up Python environment with Anaconda 
```
conda env create -f environment.yml
``` 

## Script components
To run a script, a file path in the script should be modified to be consistent with an attempted directory.

### 1. active_learning
This directory contains scripts to perform active learning campaign based on [evidential learning.](https://pubs.acs.org/doi/10.1021/acscentsci.1c00546) 

### 2. experimental_chi
This directory contains scripts to estimate Flory-Huggins interaction parameters from [COSMO-SAC](https://pubs.acs.org/doi/10.1021/ie001047w) calculations.

### 3. external_packages
This directory contains external packages.

### 4. figure_publication 
This directory contains scripts to draw figures in the paper.

### 5. qm9_active_learning_add_100 / qm9_active_learning_add_1000
This directory contains example scripts to perform active learning on QM9 for the figures in SI

### 6. run_calculation
This directory contains scripts to run quantum chemistry calculations. 

### 7. utils
This directory contains functions needed to run scripts. 

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson

## Funding Acknowledgements
This work was supported by the IBM-Illinois Discovery Accelerator Institute. N.E.J. thanks the 3M Nontenured Faculty Award for support of this research.  

<p align="right">
<img src="https://github.com/TheJacksonLab/OMG_PhysicalProperties/blob/main/figure_publication/OMG.png" width="200" height="60"> 
</p>