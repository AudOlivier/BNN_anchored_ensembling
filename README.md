# BNN with anchored ensembles

Please cite our preprint whenever using this code:
Ghorbanian, Casaprima and Olivier, "Empowering Bayesian Neural Networks with Functional Priors through Anchored Ensembling for Mechanics Surrogate Modeling Applications" https://doi.org/10.48550/arXiv.2409.05234

Main files:
- Investigate_prior_ensembling.ipynb is where the prior study is performed. It calls pre-trained NNs saved in 'priors_073024_N[200].pkl'. This pre-training happens off-line, in file do_prior_study_parallel.py (training of ensemble is parallelized)
- Example_1D_synthetic.ipynb has the small 1d problem, pre-training happens in BNN_functional_prior_parallel.py
- Materials_data_Abaqus_all_outputs.ipynb has the materials surrogate problem, pre-training happens in BNN_functional_prior_parallel.py. It also uses files and data contained in the folder Data_Abaqus_v3

Utility files:
- Aux.py contains all algorithms and helper functions
- Compare_deterministic_NNs.ipynb compares our different implementations of NNs

Environment: environment.yml - requires packages such as numpy, scipy, matplotlib, tensorflow, scikit-learn
