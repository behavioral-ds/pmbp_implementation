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
max_workers = 6
num_starting_points = 6

refvals_percentile_comparison = pd.read_csv("dat/active90to120tot.csv")["v"].values

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
    
    hyperparameter_index = pickle.load(open("dat/final_hyperdict.p", "rb"))[video_index]
    
    history = return_ic_data_given_ytid(video_name, hipper_data, 0, end_validation)
    val_view_volume = np.sum([x[0] for x in return_ic_data_given_ytid(video_name, hipper_data, end_validation, end_test)[0]])

    gamma_start = np.array([[x[0] for x in history[i]][0] for i in range(D)])
    gamma_max = np.array([np.max([x[0] for x in history[i]][:10]) for i in range(D)])
    
    tweets = np.load(f'dat/{video_index}.npy')
    history[2] = list(tweets[(tweets<=end_validation) & (tweets > 0)])
    
    num_tweets_until_90 = list(tweets[(tweets<=end_validation) & (tweets > 0)])    
    # SWITCH. if hawkes points # is not too high, use pmbp. else use mbp
    if len(num_tweets_until_90) < 1000:
        model_used = "pmbp"
        E = [0, 1]
        hyperparameter_map = [
            [[1,1,1], 1000, "random", "start"],
            [[1,1,1000], 1000, "random", "start"],
            [[1,1,1], 1000, "random", "max"],
            [[1,1,1000], 1000, "random", "max"]
        ]

        
    else:
        model_used = "mbp"
        history[2] = interval_censor_dimension(history[2], T = end_validation, partition_length = 1)
        E = [0, 1, 2]
        hyperparameter_map = [
            [[1000,1,1], 10, "random", "start"],
            [[1000,1,1], 1000, "random", "start"],
            [[1000,1,1], 10, "random", "max"],
            [[1000,1,1], 1000, "random", "max"]
        ]


    print("MODEL", model_used)
    
    x0_kernel_parameters = get_starting_points(D,theta_ub,num_starting_points)[:num_starting_points]
    hyperparameter_list = [hyperparameter_map[hyperparameter_index]]

    random_nus = np.array([np.random.random() for _ in range(3)])
#     mean_nus = np.array([np.mean([x[0] for x in history[i]][-10:]) for i in range(D)])
#     if hyperparameter_list[0][2] == "random":
    hyperparameter_list[0][2] = random_nus
#     elif hyperparameter_list[0][2] == "mean":
#         hyperparameter_list[0][2] = mean_nus

#     starting_points = [np.hstack([x, x0_nus]) for x in x0_kernel_parameters]
    hyparam_x0_list = list(product(hyperparameter_list, x0_kernel_parameters))

    video_index = str(video_index).zfill(5)

    r_gammas = []
    for hpl, x0_k in hyparam_x0_list:
        if hpl[3] == "start":
            r_gammas.append(gamma_start)
        elif hpl[3] == "max":
            r_gammas.append(gamma_max)
    
    
    logfit_label = "finalfit"
    r_history, r_video_index, r_theta_ub, r_end_validation, r_E, r_logfit_label = list(zip(
            *repeat([
                history,
                video_index,
                theta_ub,
                end_validation,
                E,
                logfit_label
            ], len(hyparam_x0_list))
        ))
    
    if not os.path.exists(f"refit/v{video_index}.p"):
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            res_outputs = executor.map(run_optimization_one_sequence_given_starting_point,
                                       hyparam_x0_list,
                                       r_gammas,
                                       r_history,
                                       r_video_index,
                                       list(range(len(hyparam_x0_list))),
                                       r_theta_ub,
                                       r_end_validation,
                                       r_E,
                                       r_logfit_label
                                      )
        res_outputs = list(res_outputs)
        pickle.dump([model_used, res_outputs], open(f"refit/v{video_index}.p", "wb"))

        fit_duration = perf_counter() - t_start
    
    else:
        model_used, res_outputs = pickle.load(open(f"refit/v{video_index}.p", "rb"))
        fit_duration = 0
    
#    sample forward 90-120        
    nlls = [x[1]["obj_val"] for x in res_outputs][:num_starting_points]
    min_nll = min(nlls)
    min_nll_index = nlls.index(min(nlls))
    min_nll_param = res_outputs[min_nll_index][0]
    hyperparameter_best_params = [min_nll_param]
    hyperparameter_best_nlls = [min_nll]
            
    
    gamma_opt = hyperparameter_map[hyperparameter_index][3]
    print(gamma_opt)
    if gamma_opt == "start":
        gammas = gamma_start
        r_gammas = [gamma_start]*n
    elif gamma_opt == "max":
        gammas = gamma_max
        r_gammas = [gamma_max]*n

        
    r_end_validation, r_end_test, r_history, r_video_index, r_val_view_volume, r_refvals_percentile_comparison, r_E, r_folder = list(zip(
            *repeat([
                end_validation,
                end_test,
                history,
                video_index,
                val_view_volume,
                refvals_percentile_comparison,
                E,
                "refit"
            ], n)
        ))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if model_used == "pmbp":
            # parallelize across thinning samples, not across hyperparameter combinations
            once_collection = executor.map(sample_forward_once,
                                       list(range(n)), 
                                       r_end_validation,
                                       r_end_test,
                                       r_history, 
                                       hyperparameter_best_params*n,            
                                       r_gammas, 
                                       hyperparameter_best_nlls*n,
                                       r_video_index,
                                       [0]*n,
                                       r_val_view_volume,
                                       r_refvals_percentile_comparison,
                                       r_E
                                      )
        else:
            sim_outputs = executor.map(complete_mbp_percentile_comparison,
                                       r_end_validation,
                                       r_end_test,
                                       r_history, 
                                       hyperparameter_best_params,            
                                       r_gammas, 
                                       hyperparameter_best_nlls,
                                       r_video_index,
                                       [0]*n,
                                       r_val_view_volume,
                                       r_refvals_percentile_comparison,
                                       r_E,
                                       r_folder
                                      )
    
    if model_used == "pmbp":
        once_collection = list(once_collection)
        collector = [x[0] for x in once_collection]
        durations = [x[1] for x in once_collection]
        sim_outputs = [aggregate_sample_forwards(collector, 
                                  durations, 
                                  end_validation, 
                                  end_test, 
                                  history, 
                                  hyperparameter_best_params[0], 
                                  gammas,
                                  hyperparameter_best_nlls[0], 
                                  video_index, 
                                  0, 
                                  val_view_volume, 
                                  refvals_percentile_comparison, 
                                  E, 
                                  folder="refit")]
        
    sim_outputs = list(sim_outputs)
    perc_errors = [x[2] for x in sim_outputs]
    min_perc_error_index = perc_errors.index(min(perc_errors))    
    best_hyperparameter_details = sim_outputs[min_perc_error_index]
    
    simulation_duration = perf_counter()-t_start
    
    pickle.dump([best_hyperparameter_details, fit_duration, simulation_duration, model_used], open(f"refit/v{video_index}_h.p", "wb"))

if __name__ == "__main__":
    main()
