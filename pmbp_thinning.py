import pickle, copy, logging
import numpy as np
from scipy.signal import fftconvolve
from time import perf_counter

from pmbp_base import (
    exponential_kernel,
    exponential_kernel_integral,
    return_kernel_matrix_at_t,
    return_kernel_matrix,
    pointwise_nu,
    pointwise_nu_integral,
    convolve_matrices,
    get_approx_to_f,
    preprocess_history,
    get_effective_history,
    return_h_and_H
)

from pmbp_utils import get_percentile_error

D=3

def return_Xi_at_s(
    s,
    history, # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h, 
    H,
    h_grid,
    E, # the mbp dims
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01):
    
    params = [x[:9].reshape(3,3), x[9:18].reshape(3,3)]
    nus = x[18:21]

    return_nu_integral_at_t = lambda t: pointwise_nu_integral(t, nus)
    return_nu_integral_over_time = lambda arr: np.moveaxis(np.stack([return_nu_integral_at_t(i) for i in arr]), 0, -1)

    Ec = [x for x in range(D) if x not in E]
    
    ic_dims = []
    for j in range(len(history)):
        if (len(history[j]) != 0) and type(history[j][0]) == type([]):
            ic_dims.append(j)

    # only less than s
    t_arr = get_effective_history(history, ic_dims, s, p_dt)
    t_arr = [x for x in t_arr if x[0] < s]
    
    A_matrix = np.zeros(shape=(D, len(t_arr)))
    
    if len(Ec) != 0:
        # iterate over all previous points < s, and store in a-matrix
        for index, t_data in enumerate(t_arr):
            t, t_roles = t_data

            A = np.zeros(shape=D)
            history_prior_to_t = [x for x in t_arr if x[0] < t]
            for t_p, roles_p in history_prior_to_t:
                for role_p, dim_p in roles_p:
                    if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                        A += return_kernel_matrix_at_t(t - t_p, kernel_integral, params)[:, dim_p]

                        break # only one event can happen at any instant

            A_matrix[:, index] = A
        
    # value of hawkes at s
    A = np.zeros(shape=D)
    
    if len(Ec) != 0:
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                    A += return_kernel_matrix_at_t(s - t_p, kernel_integral, params)[:, dim_p]
                    break
    
    Xi = gammas + return_nu_integral_at_t(s) + A
    Xi += np.dot(get_approx_to_f(s, h_dt, H, None, None), gammas)

    time_points_prior_to_t = np.array([x[0] for x in t_arr])
    time_points_up_to_t = np.array(list(time_points_prior_to_t) + [s])
    delta_H = np.diff(get_approx_to_f(s - time_points_up_to_t, h_dt, H, None, E),axis=-1).T
            
    for d in range(D):
        Xi[d] += np.sum(((-1*(return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:]))[E].T) * delta_H[:,:,d])
            
    return Xi

def return_xi2_at_s(
    s,
    history, # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h, 
    H,
    h_grid,
    E, # the mbp dims
    kernel=exponential_kernel,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01):
    
    params = [x[:9].reshape(3,3), x[9:18].reshape(3,3)]
    nus = x[18:21]

    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(np.stack([return_nu_at_t(i) for i in arr]), 0, -1)
    
    Ec = [x for x in range(D) if x not in E]
    
    # only less than s
    t_arr = get_effective_history(history, [0, 1], s, p_dt)
    t_arr = [x for x in t_arr if x[0] < s]
    
    a_matrix = np.zeros(shape=(D, len(t_arr)))
    
    if len(Ec) != 0:
        # iterate over all previous points < s, and store in a-matrix
        for index, t_data in enumerate(t_arr):
            t, t_roles = t_data

            a = np.zeros(shape=D)
            history_prior_to_t = [x for x in t_arr if x[0] < t]
            for t_p, roles_p in history_prior_to_t:
                for role_p, dim_p in roles_p:
                    if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                        a += return_kernel_matrix_at_t(t - t_p, kernel, params)[:, dim_p]

                        break # only one event can happen at any instant

            a_matrix[:, index] = a
        
    # value of hawkes at s
    a = 0
    
    if len(Ec) != 0:
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                    a += return_kernel_matrix_at_t(s - t_p, kernel, params)[2, dim_p]
                    break
    
    xi = nus[2] + a
    xi += np.dot(get_approx_to_f(s, h_dt, h, 2, None), gammas)

    time_points_prior_to_t = np.array([x[0] for x in t_arr])
    time_points_up_to_t = np.array(list(time_points_prior_to_t) + [s])
    delta_H = np.diff(get_approx_to_f(s - time_points_up_to_t, h_dt, H, None, E),axis=-1).T
    
    xi += np.sum(((-1*(return_nu_over_time(time_points_prior_to_t) + a_matrix[:,:]))[E].T) * delta_H[:,:,2])

    return xi

def return_ub2_at_s(
    s,
    history, # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h, 
    H,
    h_grid,
    E, # the mbp dims
    kernel=exponential_kernel,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01,
    T=120):
    
    params = [x[:9].reshape(3,3), x[9:18].reshape(3,3)]
    nus = x[18:21]
    D = len(history)

    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(np.stack([return_nu_at_t(i) for i in arr]), 0, -1)
    
    Ec = [x for x in range(D) if x not in E]
    
    # only less than s
    t_arr = get_effective_history(history, [0, 1], T, p_dt)
    t_arr = [x for x in t_arr if x[0] <= s]
    t_arr_no_s = [x for x in t_arr if x[0] < s]
        
    a_matrix = np.zeros(shape=(D, len(t_arr_no_s)))
    
    if len(Ec) != 0:
        # iterate over all previous points < s, and store in a-matrix
        for index, t_data in enumerate(t_arr_no_s):
            t, t_roles = t_data

            a = np.zeros(shape=D)
            history_prior_to_t = [x for x in t_arr_no_s if x[0] < t]
            for t_p, roles_p in history_prior_to_t:
                for role_p, dim_p in roles_p:
                    if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                        a += return_kernel_matrix_at_t(t - t_p, kernel, params)[:, dim_p]

                        break # only one event can happen at any instant

            a_matrix[:, index] = a
        
    a = np.zeros(D)
    
    if len(Ec) != 0:
        # value of hawkes ub at s
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                    a += return_kernel_matrix_at_t(s - t_p, kernel, params)[2, :]
                    break
    
    time_points_up_to_t = np.array([x[0] for x in t_arr_no_s] + [s])
    H_hat = return_H_hat(h, H, h_grid)
    delta_H = np.diff(get_approx_to_f(s - time_points_up_to_t, h_dt, H_hat, None, E),axis=-1).T    
        
    h_max = np.max(h, axis=-1)  
    
    diff = (s - h_grid) >= 0
    idx = np.sum(diff)-1
    h_max_past_s = np.max(h[:,:,idx:], axis=-1)  

    ub = nus[2] + a[2] + np.dot(h_max_past_s[2,:], gammas) + np.dot(H[2,:,-1], nus) + \
        np.sum(((-1*(a_matrix[:,:]))[E].T) * delta_H[:,:,2])
    
    
    h_hat = h.copy()
    for j in range(D):
        h_hat[2,j,:np.where(h_hat[2,j] == h_max[2,j])[0][0]] = h_max[2,j]

    t_index = np.floor(T - s / h_dt).astype(int)  
    H_hat = get_approx_to_f(T - s, h_dt, H_hat, 2, None)
    
    ub += np.dot(H_hat, a)
    
#     print("UB", ub, "COMPONENTS", nus[2], a[2], np.dot(h_max_past_s[2,:], gammas), np.dot(H[2,:,-1], nus), np.sum(((-1*(a_matrix[:,:]))[E].T) * delta_H[:,:,2]), np.dot(H_hat, a))
#     print("HHAT", H_hat, a, params)
    return ub

def return_H_hat(h, H, h_grid):
    h_max = np.max(h, axis=-1)
    t_max = np.array([[int(np.where(h[i,j] == h_max[i,j])[0][0]) for j in range(D)] for i in range(D)])
    
    H_hat = H.copy()
    for i in range(D):
        for j in range(D):
            if t_max[i,j] != 0:
                H_hat[i,j,:t_max[i,j]] = h_max[i,j] * h_grid[:t_max[i,j]]
            H_hat[i,j,t_max[i,j]:] = H[i,j, t_max[i,j]:] + h_grid[t_max[i,j]] * h_max[i,j] - H[i,j, t_max[i,j]]
    return H_hat
            
def return_history2(s_0, T, p_dt, kernel, plist, gammas, h, H, h_grid, history, E, hyper_index, video_index, run_index, to_log = True, max_time = 600, logfile_label = "pmbpthinning"):

    logging.basicConfig(filename=f"log/{logfile_label}_{video_index}_{hyper_index}_{run_index}.log", level=logging.INFO, format='%(asctime)s | %(message)s', force=True)
    
    start = perf_counter()
    
    n_list = np.zeros(shape=2)
    
    a_collector = np.empty(shape=[3,0])
    delta_collector = []
    grid_collector = []
    
    s = s_0
    
    too_long = False
    
    while s < T:
        
        # if sampling takes longer than 10 minutes, use Complete MBP.
        if perf_counter() - start > max_time:
            logging.info(f"breaking out. sampling too long. use complete mbp.")
            print("breaking out. sampling too long. use complete mbp")
            too_long = True
            break
        
        
        # set upper bound until next event
        lambda_star = return_ub2_at_s(
            s,
            history,
            plist,
            gammas,
            h, 
            H,
            h_grid,
            E,
            T = T)

        u = np.random.uniform()
        w = -np.log(u) / lambda_star
                
        s = s + w
        
        if s > T:
            break
        
        rd = np.random.uniform()
        current_lambda = return_xi2_at_s(
            s,
            history,
            plist,
            gammas,
            h, 
            H,
            h_grid,
            E,
            p_dt = p_dt
            )

        if (rd * lambda_star) <= current_lambda:
            history[2].append(s)
            logging.info(f"{s}. accept.")
            print(s,"accept", current_lambda.round(2), lambda_star.round(2))
        else:
            logging.info(f"{s}. reject.")
            print("\t", s,"reject", current_lambda.round(2), lambda_star.round(2))
            continue

    if len(history[2]) != 0:
        if history[2][-1] > T:
            history[2] = history[2][:-1]
        
    if too_long:
        print("TOOK TOO LONG. GO AS POISSON.")
        history[2] = [x for x in history[2] if x < s_0]
        pts = np.array(history[2])
        # if too long, use history of last 15 days as sample instead.
        rate = len(pts[(pts >= s_0-15) & (pts < s_0)]) / 15
        
        if rate == 0:
            return history
        s = s_0
        while True:
            u = np.random.uniform()
            w = -np.log(u) / rate
            s = s + w
            if s > T:
                break
            else:
                history[2].append(s)
                print(f"poisson. {s}")
        return history
    else:
        return history

# ex: a = 1, b = 90; to get volume at 1, 2, ..., 90
def get_volume_per_unit_bet_a_b(a, b, history, plist, gammas, h, H, h_grid, E):
    return np.diff([return_Xi_at_s(
        x,
        history, # dataset: interval-censored in the ic-dims
        plist,
        gammas,
        h, 
        H,
        h_grid,
        E) for x in range(a,b+1)], axis=0)

def sample_forward(s_0, T, hist, plist, gammas, h, H, h_grid, E, p_dt, kernel, hyper_index, video_index, run_index, max_time=600, logfile_label="pmbpthinning"):
    forward_history = return_history2(s_0, T, p_dt, kernel, plist, gammas, h, H, h_grid, hist, E, hyper_index, video_index, run_index, max_time=max_time, logfile_label = logfile_label)
    
    if forward_history == "fail":
        return "fail", "fail"
    else:    
        duration = perf_counter()
        print("getting vol")
        volumes = get_volume_per_unit_bet_a_b(s_0, T, hist, plist, gammas, h, H, h_grid, E)
        print("...done getting vol")
        duration = perf_counter() - duration

        return volumes, duration


def average_sample_forwards_with_percentile_comparison(n, s_0, T, input_history, plist, gammas, nll, video_index, hyper_index, actual_volume, percentile_refvals, E, p_dt=1, h_dt=0.01, kernel = exponential_kernel, kernel_integral = exponential_kernel_integral, h_tol=1e-6, folder="hypertuning"):
    
    D = len(input_history)
    kp = np.array(plist[:-D]).reshape(2,D,D)

    h_grid = np.arange(0, T+0.1*h_dt, h_dt)
    h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, T, h_tol)
    
    collector = []
    durations = []
    for i in range(n):
        np.random.seed(i)
        hist = copy.deepcopy(input_history)
        volumes, duration = sample_forward(s_0, T, hist, plist, gammas, h, H, h_grid, E, p_dt, kernel, hyper_index, video_index, i)
        
        if volumes == "fail":
            break
        
        collector.append(volumes)
        durations.append(duration)
    
    if volumes != "fail":
        mean_volume = np.mean(collector, axis=0).T
        val_view_volume = np.sum(mean_volume[0])
        percentile_error = get_percentile_error(val_view_volume, actual_volume, percentile_refvals)

        pickle.dump([nll, hyper_index, percentile_error, mean_volume, collector, durations], open(f"{folder}/v{video_index}_h{hyper_index}.p", "wb"))
        return [nll, hyper_index, percentile_error, mean_volume, collector, durations]
    
    else:
        return [1e9, hyper_index, 1e9, 1e9, [], []]
    ###
    
        E = [0,1,2]
        h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, T, h_tol)
        end_Xi = return_Xi_at_s(
                T,
                input_history,
                plist,
                gammas,
                h, 
                H,
                h_grid,
                E,
                kernel,
                kernel_integral,
                pointwise_nu=pointwise_nu,
                p_dt=p_dt,
                h_dt=h_dt)

        start_Xi = return_Xi_at_s(
            s_0,
            input_history,
            plist,
            gammas,
            h, 
            H,
            h_grid,
            E,
            kernel,
            kernel_integral,
            pointwise_nu=pointwise_nu,
            p_dt=p_dt,
            h_dt=h_dt)
        
        mean_volume = end_Xi - start_Xi
        val_view_volume = mean_volume[0]
        percentile_error = get_percentile_error(val_view_volume, actual_volume, percentile_refvals)
        
        pickle.dump([nll, hyper_index, percentile_error, mean_volume], open(f"{folder}/v{video_index}_h{hyper_index}.p", "wb"))
        return [nll, hyper_index, percentile_error, mean_volume]
    
#### for final fitting ####    
    
def sample_forward_once(i, s_0, T, input_history, plist, gammas, nll, video_index, hyper_index, actual_volume, percentile_refvals, E, p_dt=1, h_dt=0.01, kernel = exponential_kernel, kernel_integral = exponential_kernel_integral, h_tol=1e-6, max_time=600, logfile_label="finalthinning"):
    
    D = len(input_history)
    kp = np.array(plist[:-D]).reshape(2,D,D)

    h_grid = np.arange(0, T+0.1*h_dt, h_dt)
    h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, T, h_tol)
    
    collector = []
    durations = []
    
    np.random.seed(i)
    hist = copy.deepcopy(input_history)
    volumes, duration = sample_forward(s_0, T, hist, plist, gammas, h, H, h_grid, E, p_dt, kernel, hyper_index, video_index, i, max_time=max_time, logfile_label=logfile_label)

    return [volumes, duration]

def aggregate_sample_forwards(collector, durations, s_0, T, input_history, plist, gammas, nll, video_index, hyper_index, actual_volume, percentile_refvals, E, p_dt=1, h_dt=0.01, kernel = exponential_kernel, kernel_integral = exponential_kernel_integral, h_tol=1e-6, folder="hypertuning"):
    if "fail" not in collector:
        mean_volume = np.mean(collector, axis=0).T
        val_view_volume = np.sum(mean_volume[0])
        percentile_error = get_percentile_error(val_view_volume, actual_volume, percentile_refvals)

        pickle.dump([nll, hyper_index, percentile_error, mean_volume, collector, durations], open(f"{folder}/v{video_index}_h{hyper_index}.p", "wb"))
        return [nll, hyper_index, percentile_error, mean_volume, collector, durations]
    
    else:
        # fail. takes too long to sample
        return [nll, hyper_index, 1e9, 1e9]
                                                       
########    

def complete_mbp_percentile_comparison(s_0, T, input_history, plist, gammas, nll, video_index, hyper_index, actual_volume, percentile_refvals, E, folder="hypertuning", p_dt=1, h_dt=0.01, kernel = exponential_kernel, kernel_integral = exponential_kernel_integral, h_tol=1e-6):
    
    D = len(input_history)
    kp = np.array(plist[:-D]).reshape(2,D,D)

    h_grid = np.arange(0, T+0.1*h_dt, h_dt)
    h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, T, h_tol)

    end_Xi = return_Xi_at_s(
            T,
            input_history,
            plist,
            gammas,
            h, 
            H,
            h_grid,
            E,
            kernel,
            kernel_integral,
            pointwise_nu=pointwise_nu,
            p_dt=p_dt,
            h_dt=h_dt)
    
    start_Xi = return_Xi_at_s(
        s_0,
        input_history,
        plist,
        gammas,
        h, 
        H,
        h_grid,
        E,
        kernel,
        kernel_integral,
        pointwise_nu=pointwise_nu,
        p_dt=p_dt,
        h_dt=h_dt)
        
    mean_volume = end_Xi - start_Xi

    val_view_volume = mean_volume[0]
    
    percentile_error = get_percentile_error(val_view_volume, actual_volume, percentile_refvals)
    
    pickle.dump([nll, hyper_index, percentile_error, mean_volume], open(f"{folder}/v{video_index}_h{hyper_index}.p", "wb"))
    
    return [nll, hyper_index, percentile_error, mean_volume]


