# Advanced Machine Learning Project 1

```
.
├── README.md
├── .gitignore
├── .pre-commit-config.yaml <- pre-commit hooks for standardizing code formatting
├── .project-root <- root folder indicator
│
├── notebooks
│   ├── 01_implementation.ipynb <- IRLS usage example
│   ├── 02_simulation_experiments.ipynb <- testing correctness of the IRLS implementation
│   └── 03_real_datasets_experiments.ipynb <- comparing with different methods and datasets
│ 
├── requirements.txt <- python dependencies
│ 
└── src
    ├── data <- data loading and preprocessing
    │   ├── artificial.py <- artificial dataset
    │   ├── heart_disease.py <- heart disease dataset
    │   └── titanic.py <- titanic dataset
    │   
    └── irls.py <- implementation of the IRLS algorithm

```