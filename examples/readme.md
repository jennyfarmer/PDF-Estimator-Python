# Example distributions for testing and comparison

Examples provided for extensive comparison between known true distributions and evaluated
estimates. Currently supported estimation methods are PDFE and kernel density estimation (KDE) 
from scikit-learn. Supported distributions currently implemented are found in examples/distributions.
Additional distributions can be implemented using base.py as a template.

## Experiment parameters

| Parameter name     | Description                                                        |
|--------------------|--------------------------------------------------------------------|
| `distribution_ids` | Numeric list of distributions to compare; IDs correspond to distributions defined in compare_estimators.py.              |
| `sample_sizes`     | Numeric list of size of random samples to generate for comparison  |
| `seeds`            | Numeric list of random seeds used to generate reproducible samples |
| `plot_pdfs`        | true/false flag for plotting estimated and true distribution       |
| `evaluate_kde`     | true/false flag indicating if KDE method evaluated for comparison  |
| `verbose`          | true/false flag for console output                                 |
| `write_file`       | true/false flag for writing results to CSV file                    |



## Local comparison runs

Edit the experiment parameters in run_examples.py and run the python script directly. 
Selected parameters are passed to compare_estimators.py

## SLURM comparison runs

Edit the experiment parameters in run_examples_slurm.sh, then execute:

    ./run_examples_slurm.sh

The submission script calculates the required SLURM array size and submits
run_examples.slurm. Each array task invokes run_examples_slurm.py, which
passes the selected parameters to compare_estimators.py.

Normal stdout is discarded unless verbose output is enabled; stderr is retained 
per array task, with empty error files removed.

Results are written as CSV files named according to distribution, sample size,
and random seed. 

## CSV file results

| Result         | Description                                                   |
|----------------|---------------------------------------------------------------|
| `distribution` | Name of distribution                                          |
| `sample_size`  | Size of random samples generated for estimation               |
| `seed`         | Random seed used to generate reproducible sample              |
| `estimator`    | Method used for density estimation                            |
| `eval_time`    | Total elapsed time for evaluation of estimate                 |
| `mse `         | Mean squared error of estimate compared to true distribution  |
| `mae `         | Mean absolute error of estimate compared to true distribution |
| `max_error`    | Maximum error of estimate compared to true distribution       |