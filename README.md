# Advanced Machine Learning Project 1
Repository for Advanced Machine Learning course first project. 

My main AML course repository can be found under this [link](https://github.com/Wladeko/advanced-machine-learning).

---
## Description
The aim of the project is to implement Iterative Reweighted Least Squares optimization algorithm for logistic regression and perform experiments proving that algorithm works properly.

Full project task is located in `resources` subdirectory.

---
## Project structure
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
---
## Results
We presented obtained results in short [report](https://github.com/Wladeko/aml-project-1/blob/main/report.txt).

---
## Co-author
Be sure to check co-author of this project, [Lukas](https://github.com/ashleve).