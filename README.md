# Analysis code for: "Reorganization of structural connectivity in the brain supports preservation of cognitive ability in healthy aging"
This folder contains the code used for analyses in:

Neudorf, J., Shen, K., & McIntosh, A. R. (2023). Reorganization of structural connectivity in the brain supports preservation of cognitive ability in healthy aging. bioRxiv. https://doi.org/10.1101/2023.10.25.564045

Order to run:
1. `PLS_SC.py` - this will run the PLS analyses using the `PLS_wrapper` library found [here](https://github.com/neudorf/PLS_wrapper)
2. Next you can run `PLS_bsr_processing.py` and `PLS_reliability/SC_age_CattellTotal_reliability_analyses.m`
3. After this the other three files can be run: `PLS_SC_bsr_analyses.py`, `subcortical_visualization.py`, and `ggseg.r`