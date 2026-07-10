# Introduction
This study proposes an multi-task learning framework based on GNN (chemical activity prediction on GyrB, CAP-GyrB), which integrates chemical and BRICS molecular fragment structural representations. The model is designed to predict the inhibitory activity of chemicals on GyrB of Escherichia coli.
# Usage
## Data processing
Running molpro.py for chemical structure cleaning and data_split.py for dataset splitting.
## Model training
Users can customize the relevant hyperparameters for each experiment with yaml file in configs folder.
## Hyperparameter tuning
The experiment can be run via the following command for cross-validation and Bayesian hyperparameter tuning
## Interpretation of result
To get the interpretation of predictive result, the heatmap of BRICS contribution can be obtained by running the aw2.0.py in interpretation folder.
