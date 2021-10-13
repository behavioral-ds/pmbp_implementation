import pickle
import sys, os, copy
import pandas as pd
import numpy as np
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, sleep
from itertools import product

from pmbp_base import interval_censor_dimension
from pmbp_opt import run_optimization_one_sequence_given_starting_point
from pmbp_thinning import (
    average_sample_forwards_with_percentile_comparison, 
    complete_mbp_percentile_comparison,
    sample_forward_once,
    aggregate_sample_forwards
)
from pmbp_utils import (
    get_spectral_radius_from_flat_params,
    get_starting_points,
    return_ic_data_given_ytid,
)

t_start = perf_counter()

D = 3
n = 6 # number of forward samples
theta_ub = 2
end_train = 75
end_validation = 90
end_test = 120
max_workers = 24
num_starting_points = 6

refvals_percentile_comparison = pd.read_csv("dat/active75to90tot.csv")["v"].values

hipper_data = pickle.load(open("dat/hipper_aggregated.p", "rb")).set_index("YoutubeID")
video_dict = hipper_data.reset_index()["YoutubeID"].to_dict()

def main():

    # Get the starting integer and the ending integer.
    if len(sys.argv) != 2:
        print("Usage: %s start end" % sys.argv[0])
        print("Type in pbs index.")
        sys.exit()
    try:
        video_index = int(sys.argv[1])
    except:
        print("One of your arguments was not an integer.")
        sys.exit()

    video_name = video_dict[video_index]
    history = return_ic_data_given_ytid(video_name, hipper_data, 0, end_train)
    val_view_volume = np.sum([x[0] for x in return_ic_data_given_ytid(video_name, hipper_data, end_train, end_validation)[0]])

    gamma_start = np.array([[x[0] for x in history[i]][0] for i in range(D)])
    gamma_max = np.array([np.max([x[0] for x in history[i]][:10]) for i in range(D)])

#     tweets = np.load(f'/data/pbcalder/active20sample/{video_index}.npy')
    tweets = np.load(f'dat/{video_index}.npy')
    history[2] = list(tweets[(tweets<=end_train) & (tweets > 0)])
    
    num_tweets_until_90 = list(tweets[(tweets<=end_validation) & (tweets > 0)])    
    # SWITCH. if hawkes points # is not too high, use pmbp. else use mbp
    if len(num_tweets_until_90) < 1000:
        model_used = "pmbp"
        E = [0, 1]
# # GADI
        hyperparameter_list = [
            [[1,1,1], 1000, "random", "start"],
            [[1,1,1000], 1000, "random", "start"],
            [[1,1,1], 1000, "random", "max"],
            [[1,1,1000], 1000, "random", "max"]
        ]

    else:
        model_used = "mbp"
        history[2] = interval_censor_dimension(history[2], T = end_train, partition_length = 1)
        E = [0, 1, 2]
        hyperparameter_list = [
            [[1000,1,1], 10, "random", "start"],
            [[1000,1,1], 1000, "random", "start"],
            [[1000,1,1], 10, "random", "max"],
            [[1000,1,1], 1000, "random", "max"]
        ]

    
    print("MODEL", model_used)
    
    x0_kernel_parameters = get_starting_points(D,theta_ub,num_starting_points)[:num_starting_points]

    random_nus = np.array([np.random.random() for _ in range(3)])
#     mean_nus = np.array([np.mean([x[0] for x in history[i]][-10:]) for i in range(D)])
        
    # set starting point for nus
    for k in range(len(hyperparameter_list)):
        if hyperparameter_list[k][2] == "random":
            hyperparameter_list[k][2] = random_nus
        elif hyperparameter_list[k][2] == "mean":
            hyperparameter_list[k][2] = mean_nus
    
#     starting_points = [np.hstack([x, x0_nus]) for x in x0_kernel_parameters]
    hyparam_x0_list = list(product(hyperparameter_list, x0_kernel_parameters))

    
    video_index = str(video_index).zfill(5)
   
    r_gammas = []
    for hpl, x0_k in hyparam_x0_list:
        if hpl[3] == "start":
            gammas = gamma_start
            r_gammas.append(gamma_start)
        elif hpl[3] == "max":
            gammas = gamma_max
            r_gammas.append(gamma_max)


    r_history, r_video_index, r_theta_ub, r_end_train, r_E = list(zip(
            *repeat([
                history,
                video_index,
                theta_ub,
                end_train,
                E
            ], len(hyparam_x0_list))
        ))
    
    if not os.path.exists(f"hypertuning/v{video_index}.p"):

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            res_outputs = executor.map(run_optimization_one_sequence_given_starting_point,
                                       hyparam_x0_list,
                                       r_gammas,
                                       r_history,
                                       r_video_index,
                                       list(range(len(hyparam_x0_list))),
                                       r_theta_ub,
                                       r_end_train,
                                       r_E
                                      )
        res_outputs = list(res_outputs)
        pickle.dump([model_used, res_outputs], open(f"hypertuning/v{video_index}.p", "wb"))

        fit_duration = perf_counter() - t_start
    else:
        model_used, res_outputs = pickle.load(open(f"hypertuning/v{video_index}.p", "rb"))
        fit_duration = 0
    
#    sample forward 75-90    
    hyperparameter_best_params = []
    hyperparameter_best_nlls = []
    for i in range(len(hyperparameter_list)):
        nlls = [x[1]["obj_val"] for x in res_outputs][num_starting_points*i : num_starting_points*(i+1)]
        min_nll = min(nlls)
        min_nll_index = nlls.index(min(nlls))
#         min_nll_param = res_outputs[len(hyperparameter_list)*min_nll_index + i][0]
        min_nll_param = res_outputs[num_starting_points*i + min_nll_index][0]
        hyperparameter_best_params.append(min_nll_param)
        hyperparameter_best_nlls.append(min_nll)
            
    
    # if pmbp, parallelize across SAMPLES (hyp x samples)
    if model_used == "pmbp":
        r_gammas = []
        for sim_sample in range(n):
            for hyp in hyperparameter_list:
                if hyp[3] == "start":
                    gammas = gamma_start
                    r_gammas.append(gammas)
                elif hyp[3] == "max":
                    gammas = gamma_max
                    r_gammas.append(gammas)

        r_end_train, r_end_validation, r_history, r_video_index, r_val_view_volume, r_refvals_percentile_comparison, r_E, r_folder = list(zip(
                *repeat([
                    end_train,
                    end_validation,
                    history,
                    video_index,
                    val_view_volume,
                    refvals_percentile_comparison,
                    E,
                    "hypertuning"
                ], len(hyperparameter_list) * n)
            ))
    else:
    # if mbp, parallelize across HYPERPARAMETERS (hyp)
        r_gammas = []
        for hyp in hyperparameter_list:
            if hyp[3] == "start":
                gammas = gamma_start
                r_gammas.append(gammas)
            elif hyp[3] == "max":
                gammas = gamma_max
                r_gammas.append(gammas)

        r_n, r_end_train, r_end_validation, r_history, r_video_index, r_val_view_volume, r_refvals_percentile_comparison, r_E = list(zip(
                *repeat([
                    n,
                    end_train,
                    end_validation,
                    history,
                    video_index,
                    val_view_volume,
                    refvals_percentile_comparison,
                    E
                ], len(hyperparameter_list))
            ))


    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if model_used == "pmbp":
            once_collection = executor.map(sample_forward_once,
                                       [item for sublist in [[i]*len(hyperparameter_list) for i in range(n)] for item in sublist], # run index
                                       r_end_train,
                                       r_end_validation,
                                       r_history, 
                                       hyperparameter_best_params*n,            
                                       r_gammas, 
                                       hyperparameter_best_nlls*n,
                                       r_video_index,
                                       list(range(len(hyperparameter_list)))*n, # hyperparameter index
                                       r_val_view_volume,
                                       r_refvals_percentile_comparison,
                                       r_E
                                      )

        else:
            sim_outputs = executor.map(complete_mbp_percentile_comparison,
                                       r_end_train, 
                                       r_end_validation,
                                       r_history, 
                                       hyperparameter_best_params,            
                                       r_gammas, 
                                       hyperparameter_best_nlls,
                                       r_video_index,
                                       list(range(len(hyperparameter_list))),
                                       r_val_view_volume,
                                       r_refvals_percentile_comparison,
                                       r_E
                                      )

    if model_used == "pmbp":
        once_collection = list(once_collection)
        sim_outputs = []
        for hyp_index in range(len(hyperparameter_list)):
            collection_for_hyp = once_collection[hyp_index::len(hyperparameter_list)]
            collector = [x[0] for x in collection_for_hyp]
            durations = [x[1] for x in collection_for_hyp]
            sim_outputs.append(aggregate_sample_forwards(collector, 
                                      durations, 
                                      end_train, 
                                      end_validation, 
                                      history, 
                                      hyperparameter_best_params[hyp_index], 
                                      gammas,
                                      hyperparameter_best_nlls[hyp_index], 
                                      video_index, 
                                      hyp_index, 
                                      val_view_volume, 
                                      refvals_percentile_comparison, 
                                      E))
    sim_outputs = list(sim_outputs)
    perc_errors = [x[2] for x in sim_outputs]
    
    min_perc_error_index = perc_errors.index(min(perc_errors))    
    best_hyperparameter_details = sim_outputs[min_perc_error_index]
    
    simulation_duration = perf_counter()-t_start
    
    pickle.dump([best_hyperparameter_details, fit_duration, simulation_duration, model_used], open(f"hypertuning/v{video_index}_h.p", "wb"))

if __name__ == "__main__":
    main()