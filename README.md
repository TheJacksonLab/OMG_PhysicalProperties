# Functional monomer design for synthetically accessible polymers

This repository contains python scripts used to obtain comprehensive monomer-level properties for [synthetically accessible Open Macromolecular Genome (OMG) polymers.](https://pubs.acs.org/doi/10.1021/acspolymersau.3c00003) 
Trained ML models, conformer geometry, and ML-based monomer-level properties for 12M OMG polymers with prediction 
uncertainties are available at [Zotero](TODO).

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
This directory contains scripts to perform active learning campaign based on [evidential learning.](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html) 
To use trained models, download "pareto_greedy_check_point.tar.gz" from [Zotero](TODO) and extract "pareto_greedy_check_point.tar.gz" with the following command.
```
tar -xvzf pareto_greedy_check_point.tar.gz
```
The current file path in the scripts assume that the "pareto_greedy_check_point" directory is located at "./active_learning" 

### 2. data
This directory contains quantum chemistry calculation results during the active learning campaign. 
For example, "./active_learning/pareto_greedy/OMG_train_batch_0_chemprop_with_reaction_id.csv" contains quantum chemistry calculation results from the initial train data. 
Similarly, "./active_learning/pareto_greedy/OMG_train_batch_1_chemprop_with_reaction_id.csv" contains quantum chemistry calculation results from Round 1. 
This directory also contains 200 RDKit features ("./rdkit_features") used in training ML models to overcome [the local nature of message passing](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237) in graph neural networks.

### 3. experimental_chi
This directory contains scripts to estimate Flory-Huggins interaction parameters from [COSMO-SAC](https://pubs.acs.org/doi/10.1021/ie001047w) calculations.

### 4. external_packages
This directory contains external packages including [D-MPNN networks](https://pubs.acs.org/doi/10.1021/acscentsci.1c00546) and [Pareto front search algorithm](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_52) from [GitHub.](https://github.com/KernelA/nds-py)

### 5. figure_publication 
This directory contains scripts to draw main/SI figures in the paper.

### 6. qm9_active_learning_add_100 / qm9_active_learning_add_1000
This directory contains example scripts to perform active learning on QM9 for the figures in SI

### 7. run_calculation
This directory contains scripts to run quantum chemistry calculations.
  - batch_conformer_FF.py (conformer search and geometry optimization with UFF)
  - batch_conformer_xtb.py (geometry optimization with XTB2)
  - batch_rdkit_dft_property_calculation.py (DFT and TD-DFT calculations)
  - batch_save_dft_chi_results.py (save DFT results and calculate Flory-Huggins interaction parameters)
  - batch_save_total_results.py  (save results)

### 8. utils
This directory contains functions needed to run scripts. 

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson

## Funding Acknowledgements
This work was supported by the IBM-Illinois Discovery Accelerator Institute. N.E.J. thanks the 3M Nontenured Faculty Award for support of this research.  

<p align="right">
<img src="https://github.com/TheJacksonLab/OMG_PhysicalProperties/blob/main/figure_publication/OMG.png" width="200" height="60"> 
</p>