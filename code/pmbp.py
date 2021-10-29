import pickle
import sys, os, copy
import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, sleep
from itertools import product
import matplotlib.pyplot as plt

from .pmbp_base import interval_censor_dimension
from .pmbp_opt import run_optimization_one_sequence_given_starting_point
from .pmbp_thinning import (
    complete_mbp_percentile_comparison,
    sample_forward,
    sample_forward_once,
    aggregate_sample_forwards,
    get_volume_per_unit_bet_a_b,
    get_volume_per_unit_bet_a_b_recompute_h,
)
from .pmbp_base import exponential_kernel, exponential_kernel_integral, return_h_and_H
from .pmbp_utils import (
    get_spectral_radius_from_flat_params,
    get_starting_points,
    return_ic_data_given_ytid,
    get_Ec_volumes,
    get_rmse_error,
)


class PMBP(object):
    def __init__(self):
        # params and dataset to None
        # hyperparameters set to default

        # default settings
        self.n = 5  # number of test samples for prediction
        self.theta_ub = 2
        self.max_workers = None  # use maximum number of workers
        self.num_starting_points = 3

        # default hyperparameters
        self.dimension_weights = [1, 1, 1]
        self.nu_reg = 10
        self.gamma_hyp = "start"  # options: start, max

        # pmbp parameter list
        self.parameters = None
        # nll of optimal parameter set
        self.nll = None

        # items below to be filled up by initialize
        self.data_label = None
        self.D = None
        self.E = None
        self.Ec = None

        self.history = None
        self.history_train = None
        self.history_val = None
        self.history_trainval = None
        self.history_test = None
        self.history_trainval_volumes = None
        self.history_test_volumes = None
        self.history_traintest = None

        self.gamma_start = None
        self.gamma_max = None
        self.val_volume = None
        self.test_volume = None

        self.train_volume_prediction = None
        self.test_volume_prediction = None

        # settings to split data
        self.end_train = None
        self.end_validation = None
        self.end_test = None

        # prediction output
        self.test_rmse = None
        self.test_mean_volume = None
        self.test_samples = None

    def initialize(self, data_label, history, E, end_train, end_validation, end_test):
        # supply the PMBP model with the training data, E, and the train-val-test split

        self.data_label = data_label
        self.D = len(history)
        self.E = E  # [x for x in range(len(history)) if type(x[0]) = type([1])]
        self.Ec = [x for x in list(range(self.D)) if x not in E]
        self.history = history
        self.end_train = end_train
        self.end_validation = end_validation
        self.end_test = end_test

        self.history_train = copy.deepcopy(self.history)
        self.history_val = copy.deepcopy(self.history)

        self.history_trainval = copy.deepcopy(self.history)
        self.history_trainval_volumes = copy.deepcopy(self.history)
        self.history_test = copy.deepcopy(self.history)
        self.history_test_volumes = copy.deepcopy(self.history)
        self.history_traintest = copy.deepcopy(self.history)

        gamma_start = [
            [x[0] for x in self.history_train[i]][0] for i in range(len(self.E))
        ]
        gamma_max = [
            np.max(
                [x[0] for x in self.history_train[i]][
                    : min([len(self.history_train[i]), 10])
                ]
            )
            for i in range(len(self.E))
        ]

        for d in self.E:
            train = [x for x in self.history[d] if x[1][1] <= end_train]
            trainval = [x for x in self.history[d] if x[1][1] <= end_validation]
            val = [
                x
                for x in self.history[d]
                if (x[1][0] >= end_train) and (x[1][1] <= end_validation)
            ]
            test = [
                x
                for x in self.history[d]
                if (x[1][0] >= end_validation) and (x[1][1] <= end_test)
            ]
            traintest = [x for x in self.history[d] if (x[1][1] <= end_test)]
            self.history_trainval[d] = trainval
            self.history_trainval_volumes[d] = trainval
            self.history_train[d] = train
            self.history_val[d] = val
            self.history_test[d] = test
            self.history_test_volumes[d] = test
            self.history_traintest[d] = traintest

        for d in self.Ec:
            train = list(self.history[d][self.history[d] < end_train])
            trainval = list(self.history[d][self.history[d] < end_validation])
            val = list(
                self.history[d][
                    (self.history[d] >= end_train) & (self.history[d] < end_validation)
                ]
            )
            test = list(
                self.history[d][
                    (self.history[d] >= end_validation) & (self.history[d] < end_test)
                ]
            )
            traintest = list(self.history[d][(self.history[d] < end_test)])
            self.history_trainval[d] = trainval
            self.history_trainval_volumes[d] = get_Ec_volumes(self.history_trainval, d)
            self.history_train[d] = train
            self.history_val[d] = val
            self.history_test[d] = test
            self.history_test_volumes[d] = get_Ec_volumes(self.history_test, d)
            self.history_traintest[d] = traintest

            Ec_train = get_Ec_volumes(self.history_train, d)
            gamma_start.append(Ec_train[0][0])
            gamma_max.append(
                np.max([x[0] for x in Ec_train[: min([len(Ec_train), 10])]])
            )

        self.gamma_start = np.array(gamma_start)
        self.gamma_max = np.array(gamma_max)
        self.val_volume = np.sum([x[0] for x in self.history_val[0]])
        self.test_volume = np.sum([x[0] for x in self.history_test[0]])

    def print_parameters(self):
        print("### THETA ###")
        print(np.array(self.parameters[: self.D * self.D]).reshape(self.D, self.D))
        print("### ALPHA ###")
        print(
            np.array(self.parameters[self.D * self.D : 2 * self.D * self.D]).reshape(
                self.D, self.D
            )
        )
        print("### NU ###")
        print(
            np.array(
                self.parameters[2 * self.D * self.D : 2 * self.D * self.D + self.D]
            )
        )

    def tune_hyperparameters(self, hyperparameter_grid):
        # perform hyperparameter tuning over the specified hyperparameter_grid

        x0_list = get_starting_points(
            self.D, self.E, self.theta_ub, self.num_starting_points
        )

        # hyperparameters used to fit. if perform_hyperparameter_tuning is True, selected via self.tune_hyperparameters
        hyparam_x0_list = list(product(hyperparameter_grid, x0_list))

        ##### FITTING ON TRAIN SET. PARALLELIZE OVER STARTING POINTS ACROSS CORES ######

        r_gammas = []
        for hpl, x0_k in hyparam_x0_list:
            if hpl[2] == "start":
                r_gammas.append(self.gamma_start)
            elif hpl[2] == "max":
                r_gammas.append(self.gamma_max)

        (
            r_history_train,
            r_data_label,
            r_theta_ub,
            r_end_train,
            r_E,
            r_logfit_label,
        ) = list(
            zip(
                *repeat(
                    [
                        self.history_train,
                        self.data_label,
                        self.theta_ub,
                        self.end_train,
                        self.E,
                        "hypertuning",
                    ],
                    len(hyparam_x0_list),
                )
            )
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            res_outputs = executor.map(
                run_optimization_one_sequence_given_starting_point,
                hyparam_x0_list,
                r_gammas,
                r_history_train,
                r_data_label,
                list(range(len(hyparam_x0_list))),
                r_theta_ub,
                r_end_train,
                r_E,
                r_logfit_label,
            )
        res_outputs = list(res_outputs)

        nlls = [x[1]["obj_val"] for x in res_outputs][: self.num_starting_points]
        min_nll = min(nlls)
        min_nll_index = nlls.index(min(nlls))

        hyperparameter_best_params = []
        hyperparameter_best_nlls = []
        for i in range(len(hyperparameter_grid)):
            nlls = [x[1]["obj_val"] for x in res_outputs][
                self.num_starting_points * i : self.num_starting_points * (i + 1)
            ]
            min_nll = min(nlls)
            min_nll_index = nlls.index(min(nlls))
            min_nll_param = res_outputs[self.num_starting_points * i + min_nll_index][0]
            hyperparameter_best_params.append(min_nll_param)
            hyperparameter_best_nlls.append(min_nll)

        ##### EVALUATION ON VAL SET. ######

        if len(self.Ec) != 0:
            r_gammas = []
            for sim_sample in range(self.n):
                for hyp in hyperparameter_grid:
                    if hyp[2] == "start":
                        gammas = self.gamma_start
                        r_gammas.append(gammas)
                    elif hyp[2] == "max":
                        gammas = self.gamma_max
                        r_gammas.append(gammas)

            (
                r_end_train,
                r_end_validation,
                r_history_trainval,
                r_data_label,
                r_val_volume,
                r_E,
            ) = list(
                zip(
                    *repeat(
                        [
                            self.end_train,
                            self.end_validation,
                            self.history_trainval,
                            self.data_label,
                            self.val_volume,
                            self.E,
                        ],
                        len(hyperparameter_grid) * self.n,
                    )
                )
            )

        else:
            r_gammas = []
            for hyp in hyperparameter_grid:
                if hyp[2] == "start":
                    r_gammas.append(self.gamma_start)
                elif hyp[2] == "max":
                    r_gammas.append(self.gamma_max)

            (
                r_n,
                r_end_train,
                r_end_validation,
                r_history_trainval,
                r_data_label,
                r_val_volume,
                r_E,
            ) = list(
                zip(
                    *repeat(
                        [
                            self.n,
                            self.end_train,
                            self.end_validation,
                            self.history_trainval,
                            self.data_label,
                            self.val_volume,
                            self.E,
                        ],
                        len(hyperparameter_grid),
                    )
                )
            )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            if len(self.Ec) != 0:
                once_collection = executor.map(
                    sample_forward_once,
                    [
                        item
                        for sublist in [
                            [i] * len(hyperparameter_grid) for i in range(self.n)
                        ]
                        for item in sublist
                    ],  # run index
                    r_end_train,
                    r_end_validation,
                    r_history_trainval,
                    hyperparameter_best_params * self.n,
                    r_gammas,
                    r_data_label,
                    list(range(len(hyperparameter_grid)))
                    * self.n,  # hyperparameter index
                    r_val_volume,
                    r_E,
                )

            else:
                sim_outputs = executor.map(
                    complete_mbp_percentile_comparison,
                    r_end_train,
                    r_end_validation,
                    r_history_trainval,
                    hyperparameter_best_params,
                    r_gammas,
                    r_data_label,
                    list(range(len(hyperparameter_grid))),
                    r_val_volume,
                    r_E,
                )

        if len(self.Ec) != 0:
            once_collection = list(once_collection)
            sim_outputs = []
            for hyp_index in range(len(hyperparameter_grid)):
                if hyperparameter_grid[hyp_index][2] == "start":
                    gammas = self.gamma_start
                elif hyperparameter_grid[hyp_index][2] == "max":
                    gammas = self.gamma_max
                collection_for_hyp = once_collection[
                    hyp_index :: len(hyperparameter_grid)
                ]
                collector = [x[0] for x in collection_for_hyp]
                sim_outputs.append(
                    aggregate_sample_forwards(
                        collector,
                        self.end_train,
                        self.end_validation,
                        self.history_trainval,
                        hyperparameter_best_params[hyp_index],
                        gammas,
                        self.data_label,
                        hyp_index,
                        self.val_volume,
                        self.E,
                    )
                )
        errors = [x[1] for x in sim_outputs]
        min_index = errors.index(min(errors))

        self.dimension_weights = hyperparameter_grid[min_index][0]
        self.nu_reg = hyperparameter_grid[min_index][1]
        self.gamma_hyp = hyperparameter_grid[min_index][2]

    def fit(self, perform_hyperparameter_tuning=False, grid=None):
        # fit the model to the data. if perform_hyperparameter_tuning is True, perform hyperparameter tuning using the train-val set before fitting.

        if perform_hyperparameter_tuning:
            self.tune_hyperparameters(grid)

        x0_list = get_starting_points(
            self.D, self.E, self.theta_ub, self.num_starting_points
        )

        # hyperparameters used to fit. if perform_hyperparameter_tuning is True, selected via self.tune_hyperparameters
        hyperparameter_list = [[self.dimension_weights, self.nu_reg, self.gamma_hyp]]
        hyparam_x0_list = list(product(hyperparameter_list, x0_list))

        ##### FITTING ON TRAIN-VAL SET. PARALLELIZE OVER STARTING POINTS ACROSS CORES ######

        if self.gamma_hyp == "start":
            gammas = self.gamma_start
        else:
            gammas = self.gamma_max

        (
            r_gammas,
            r_history_trainval,
            r_data_label,
            r_theta_ub,
            r_end_validation,
            r_E,
            r_logfit_label,
        ) = list(
            zip(
                *repeat(
                    [
                        gammas,
                        self.history_trainval,
                        self.data_label,
                        self.theta_ub,
                        self.end_validation,
                        self.E,
                        "fit_trainval",
                    ],
                    len(hyparam_x0_list),
                )
            )
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            res_outputs = executor.map(
                run_optimization_one_sequence_given_starting_point,
                hyparam_x0_list,
                r_gammas,
                r_history_trainval,
                r_data_label,
                list(range(len(hyparam_x0_list))),
                r_theta_ub,
                r_end_validation,
                r_E,
                r_logfit_label,
            )
        res_outputs = list(res_outputs)

        nlls = [x[1]["obj_val"] for x in res_outputs][: self.num_starting_points]
        min_nll = min(nlls)
        min_nll_index = nlls.index(min(nlls))
        self.nll = min_nll
        self.parameters = np.array(res_outputs[min_nll_index][0])

        pickle.dump(self, open(f"output/{self.data_label}", "wb"))

    #         self.nll = 1
    #         self.parameters = np.array([1,1.1,1.2,0.9,0.8,0.7,0.6,0.5,1.3] + [0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1] + [0.1,0.2,0.3])

    def evaluate(self):
        # evaluate the PMBP parameters vs. the test data by sampling the process on the test set and calculating the RMSE error on dimension 1.

        hyperparameter_best_params = [self.parameters]

        if self.gamma_hyp == "start":
            gammas = self.gamma_start
        else:
            gammas = self.gamma_max

        (
            r_gammas,
            r_end_validation,
            r_end_test,
            r_history,
            r_data_label,
            r_test_volume,
            r_E,
            r_folder,
        ) = list(
            zip(
                *repeat(
                    [
                        gammas,
                        self.end_validation,
                        self.end_test,
                        self.history_traintest,
                        self.data_label,
                        self.test_volume,
                        self.E,
                        "output",
                    ],
                    self.n,
                )
            )
        )

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            if len(self.Ec) != 0:
                once_collection = executor.map(
                    sample_forward_once,
                    list(range(self.n)),
                    r_end_validation,
                    r_end_test,
                    r_history,
                    hyperparameter_best_params * self.n,
                    r_gammas,
                    r_data_label,
                    [0] * self.n,
                    r_test_volume,
                    r_E,
                )
            else:
                sim_outputs = executor.map(
                    complete_mbp_percentile_comparison,
                    r_end_validation,
                    r_end_test,
                    r_history,
                    hyperparameter_best_params,
                    r_gammas,
                    r_data_label,
                    [0] * self.n,
                    r_test_volume,
                    r_E,
                    r_folder,
                )

        if len(self.Ec) != 0:
            once_collection = list(once_collection)
            collector = [x[0] for x in once_collection]
            sampled_histories = [x[1] for x in once_collection]
            sim_outputs = [
                aggregate_sample_forwards(
                    collector,
                    self.end_validation,
                    self.end_test,
                    self.history_traintest,
                    hyperparameter_best_params[0],
                    gammas,
                    self.data_label,
                    0,
                    self.test_volume,
                    self.E,
                    folder="output",
                )
            ]
            self.test_samples = collector
            self.sampled_histories = np.array(sampled_histories)

        prediction_output = list(sim_outputs)[0]
        self.test_mean_volume = prediction_output[2]

        self.get_test_count_predictions()
        if len(self.Ec) != 0:
            test_mean_prediction = np.mean(self.test_volume_prediction, axis=0)[:, 0]
        else:
            test_mean_prediction = self.test_volume_prediction[:, 0]
        # get rmse of dimension 0 predictions on test set
        self.test_rmse = get_rmse_error(
            np.array([x[0] for x in self.history_test_volumes[0]]), test_mean_prediction
        )

        pickle.dump(self, open(f"output/{self.data_label}_evaluatedmodel.p", "wb"))

    def print_performance_metrics(self):
        prediction = int(np.round(self.test_mean_volume))
        rmse = np.round(self.test_rmse, 2)
        print(f"Test Volume Time Series RMSE: {rmse}")
        print(
            f"Total Dimension 1 Test Volume: {self.test_volume}. Predicted Total Volume: {prediction}"
        )

    def get_train_count_predictions(self):
        # get daily volumes on the train-val set

        kp = np.array(self.parameters[: -self.D]).reshape(2, self.D, self.D)
        h_dt = 0.01
        h_grid = np.arange(0, self.end_validation + 0.1 * h_dt, h_dt)
        h, H, flag = return_h_and_H(
            exponential_kernel,
            exponential_kernel_integral,
            kp,
            h_grid,
            self.E,
            h_dt,
            self.end_validation,
            1e-6,
        )
        if self.gamma_hyp == "start":
            gammas = self.gamma_start
        else:
            gammas = self.gamma_max
        vols = get_volume_per_unit_bet_a_b(
            0,
            self.end_validation,
            self.history,
            self.parameters,
            gammas,
            h,
            H,
            h_grid,
            self.E,
        )
        self.train_volume_prediction = vols

    def get_test_count_predictions(self):
        # get daily volumes on the test set

        if self.gamma_hyp == "start":
            gammas = self.gamma_start
        else:
            gammas = self.gamma_max

        if len(self.Ec) != 0:
            r_gammas, r_end_validation, r_end_test, r_params, r_E = list(
                zip(
                    *repeat(
                        [
                            gammas,
                            self.end_validation,
                            self.end_test,
                            self.parameters,
                            self.E,
                        ],
                        self.n,
                    )
                )
            )

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                sim_outputs = executor.map(
                    get_volume_per_unit_bet_a_b_recompute_h,
                    r_end_validation,
                    r_end_test,
                    self.sampled_histories,
                    r_params,
                    r_gammas,
                    r_E,
                )
            self.test_volume_prediction = np.array(list(sim_outputs))
        else:
            h_dt = 0.01
            kp = np.array(self.parameters[: -self.D]).reshape(2, self.D, self.D)
            h_grid = np.arange(0, self.end_test + 0.1 * h_dt, h_dt)
            h, H, flag = return_h_and_H(
                exponential_kernel,
                exponential_kernel_integral,
                kp,
                h_grid,
                self.E,
                h_dt,
                self.end_test,
                1e-6,
            )

            self.test_volume_prediction = get_volume_per_unit_bet_a_b(
                self.end_validation,
                self.end_test,
                self.history_traintest,
                self.parameters,
                gammas,
                h,
                H,
                h_grid,
                self.E,
            )

    def plot_predictions(self):
        if self.train_volume_prediction is None:
            self.get_train_count_predictions()
        train_predictions = self.train_volume_prediction
        if self.test_volume_prediction is None:
            self.get_test_count_predictions()

        if len(self.Ec) != 0:
            test_mean_prediction = np.mean(self.test_volume_prediction, axis=0)
            test_std_prediction = np.std(self.test_volume_prediction, axis=0)
        else:
            test_mean_prediction = self.test_volume_prediction

        train_volumes = [
            [x[0] for x in self.history_trainval_volumes[d]] for d in range(self.D)
        ]
        test_volumes = [
            [x[0] for x in self.history_test_volumes[d]] for d in range(self.D)
        ]
        if self.gamma_hyp == "start":
            gammas = self.gamma_start
        else:
            gammas = self.gamma_max

        plt.figure(figsize=(25, 6))

        for i in range(self.D):
            plt.subplot(1, self.D, i + 1)
            plt.title(f"Dimension {i+1}", size=20)
            plt.plot(
                range(1, self.end_validation + 1),
                train_volumes[i],
                alpha=0.5,
                color="black",
                label="data",
            )
            plt.plot(
                [self.end_validation, self.end_validation + 1],
                [train_volumes[i][-1], test_volumes[i][0]],
                color="black",
                alpha=0.5,
            )
            plt.plot(
                range(self.end_validation + 1, self.end_test + 1),
                test_volumes[i],
                alpha=0.5,
                color="black",
            )

            plt.plot(
                range(0, self.end_validation + 1),
                [gammas[i]] + list(train_predictions[:, i]),
                color="blue",
            )
            plt.plot(
                [self.end_validation, self.end_validation + 1],
                [train_predictions[-1, i], test_mean_prediction[0, i]],
                color="blue",
            )
            if len(self.Ec) != 0:
                plt.fill_between(
                    range(self.end_validation + 1, self.end_test + 1),
                    (test_mean_prediction[:, i] - test_std_prediction[:, i]),
                    (test_mean_prediction[:, i] + test_std_prediction[:, i]),
                    alpha=0.5,
                )
            plt.plot(
                range(self.end_validation + 1, self.end_test + 1),
                test_mean_prediction[:, i],
                color="blue",
                label="PMBP",
            )

            plt.axvspan(
                self.end_validation + 1, self.end_test + 1, facecolor="gray", alpha=0.2
            )
            plt.xlim(0, self.end_test + 1)
            plt.xticks(size=15)
            plt.yticks(size=15)
            plt.ylabel("Event count", size=20)
            plt.xlabel("Time", size=20)
            if i == 0:
                plt.legend(fontsize=15)
        plt.savefig(f"output/{self.data_label}.pdf", bbox_inches="tight")
