# Partial Mean Behavior Poisson (PMBP) Implementation

## Description 

Implementation of the PMBP(D,E) model introduced in `Partial Multivariate Interval-censored Hawkes Processes` [Calderon, et al. '21](https://arxiv.org/abs/2111.02062).

Current implementation focuses on the D=3 case. Other cases are left for future work.

## Usage

Create a conda environment from the provided YAML file `environment.yml` using the command

```
conda env create
```

A sample script is provided in `sample.py`. Running this triggers the end-to-end pipeline on a sample dataset. The pipeline consists of PMBP (1) initialization, (2) fitting, and (3) evaluation.

The PMBP class in `code/pmbp.py` is a wrapper for defining, fitting, and evaluating the PMBP(D,E) given input data and configuration.

The basic commands to define and fit the model are:

```
pmbp = PMBP()
pmbp.initialize(data_label, history, E, end_train, end_validation, end_test)
pmbp.fit()
```

Without any input to the `fit()` method, hyperparameter tuning on the validation set is skipped and the default hyoerparameters are used. If one wishes to perform hyperparameter tuning, a grid of hyperparameters values must be provided. The `fit()` command should be replaced with

```
pmbp.fit(perform_hyperparameter_tuning=True, grid=grid)
```

where `grid` is a list of hyperparameter tuples (`dimension_weights`, `nu_regularization_weight`, `gamma_init`). Once the model is fitted, a `_fittedmodel.p` file is exported in `output/`, containing the PMBP object with the optimized parameters in `pmbp.parameters`. Log files are also written in `log/` for diagnostics.

Once the model is fitted, the model can be evaluated in terms of RMSE error on the dimension 1 time series predictions using

```
pmbp.evaluate()
```

This function implicitly samples the PMBP(D,E) process on the test set, with the number of samples controlled by the `pmbp.n` parameter (default set to 5). Once the model is evaluated, an `_evaluatedmodel.p` file is exported in `output/`, containing the PMBP object with the sampled test set histories in `pmbp.sampled_histories`. Log files are also written in `log/` for diagnostics.

Performance metrics and predictions can be printed out using the following two commands:

```
pmbp.print_performance_metrics()
pmbp.print_parameters()
```

Lastly, the fit of the model on the train-val set and predictions on the test set can be plotted using

```
pmbp.plot_predictions()
```

This saves a PDF plot to the `output/` folder.

![png](util/sample_E12.png)

## Example

An example Python script is provided to run the entire fitting pipeline for a PMBP(3,2) or PMBP(3,3) model on sample data.

To run the example, go to the command line and run the Python script.

```
python sample.py
```
on the same directory. Ideally, run the script on a tmux or screen tab so that the entire procedure runs in the background.

## License

Both dataset and code are distributed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/). If you require a different license, please contact us at <piogabrielle.b.calderon@student.uts.edu.au>
or <Marian-Andrei@rizoiu.eu>.

