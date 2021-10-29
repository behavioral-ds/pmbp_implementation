import pickle, copy, logging
import numpy as np
from scipy.signal import fftconvolve
from time import perf_counter

from .pmbp_base import (
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
    return_h_and_H,
)

from .pmbp_utils import get_percentile_error, get_rmse_error


def return_Xi_at_s(
    s,
    history,  # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h,
    H,
    h_grid,
    E,  # the mbp dims
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01,
):

    D = len(history)
    params = [x[: D * D].reshape(D, D), x[D * D : 2 * D * D].reshape(D, D)]
    nus = x[2 * D * D : 2 * D * D + D]

    return_nu_integral_at_t = lambda t: pointwise_nu_integral(t, nus)
    return_nu_integral_over_time = lambda arr: np.moveaxis(
        np.stack([return_nu_integral_at_t(i) for i in arr]), 0, -1
    )

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
                    if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                        A += return_kernel_matrix_at_t(
                            t - t_p, kernel_integral, params
                        )[:, dim_p]

                        break  # only one event can happen at any instant

            A_matrix[:, index] = A

    # value of hawkes at s
    A = np.zeros(shape=D)

    if len(Ec) != 0:
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                    A += return_kernel_matrix_at_t(s - t_p, kernel_integral, params)[
                        :, dim_p
                    ]
                    break

    Xi = gammas + return_nu_integral_at_t(s) + A
    Xi += np.dot(get_approx_to_f(s, h_dt, H, None, None), gammas)

    time_points_prior_to_t = np.array([x[0] for x in t_arr])
    time_points_up_to_t = np.array(list(time_points_prior_to_t) + [s])
    delta_H = np.diff(
        get_approx_to_f(s - time_points_up_to_t, h_dt, H, None, E), axis=-1
    ).T

    for d in range(D):
        Xi[d] += np.sum(
            (
                (
                    -1
                    * (
                        return_nu_integral_over_time(time_points_prior_to_t)
                        + A_matrix[:, :]
                    )
                )[E].T
            )
            * delta_H[:, :, d]
        )

    return Xi


def return_xi2_at_s(
    s,
    history,  # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h,
    H,
    h_grid,
    E,  # the mbp dims
    kernel=exponential_kernel,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01,
):

    D = len(history)
    params = [x[: D * D].reshape(D, D), x[D * D : 2 * D * D].reshape(D, D)]
    nus = x[2 * D * D : 2 * D * D + D]

    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(
        np.stack([return_nu_at_t(i) for i in arr]), 0, -1
    )

    Ec = [x for x in range(D) if x not in E]

    # only less than s
    t_arr = get_effective_history(history, E, s, p_dt)
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
                    if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                        a += return_kernel_matrix_at_t(t - t_p, kernel, params)[
                            :, dim_p
                        ]

                        break  # only one event can happen at any instant

            a_matrix[:, index] = a

    # value of hawkes at s
    a = 0

    if len(Ec) != 0:
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                    a += return_kernel_matrix_at_t(s - t_p, kernel, params)[2, dim_p]
                    break

    xi = nus[2] + a
    xi += np.dot(get_approx_to_f(s, h_dt, h, 2, None), gammas)

    time_points_prior_to_t = np.array([x[0] for x in t_arr])
    time_points_up_to_t = np.array(list(time_points_prior_to_t) + [s])
    delta_H = np.diff(
        get_approx_to_f(s - time_points_up_to_t, h_dt, H, None, E), axis=-1
    ).T

    xi += np.sum(
        ((-1 * (return_nu_over_time(time_points_prior_to_t) + a_matrix[:, :]))[E].T)
        * delta_H[:, :, 2]
    )

    return xi


def return_ub2_at_s(
    s,
    history,  # dataset: interval-censored in the ic-dims
    x,
    gammas,
    h,
    H,
    h_grid,
    E,  # the mbp dims
    kernel=exponential_kernel,
    pointwise_nu=pointwise_nu,
    p_dt=0.1,
    h_dt=0.01,
    T=120,
):

    D = len(history)
    params = [x[: D * D].reshape(D, D), x[D * D : 2 * D * D].reshape(D, D)]
    nus = x[2 * D * D : 2 * D * D + D]

    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(
        np.stack([return_nu_at_t(i) for i in arr]), 0, -1
    )

    Ec = [x for x in range(D) if x not in E]

    # only less than s
    t_arr = get_effective_history(history, E, T, p_dt)
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
                    if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                        a += return_kernel_matrix_at_t(t - t_p, kernel, params)[
                            :, dim_p
                        ]

                        break  # only one event can happen at any instant

            a_matrix[:, index] = a

    a = np.zeros(D)

    if len(Ec) != 0:
        # value of hawkes ub at s
        for t_p, roles_p in t_arr:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec:  # is a Hawkes event?
                    a += return_kernel_matrix_at_t(s - t_p, kernel, params)[2, :]
                    break

    time_points_up_to_t = np.array([x[0] for x in t_arr_no_s] + [s])
    H_hat = return_H_hat(h, H, h_grid)
    delta_H = np.diff(
        get_approx_to_f(s - time_points_up_to_t, h_dt, H_hat, None, E), axis=-1
    ).T

    h_max = np.max(h, axis=-1)

    diff = (s - h_grid) >= 0
    idx = np.sum(diff) - 1
    h_max_past_s = np.max(h[:, :, idx:], axis=-1)

    ub = (
        nus[2]
        + a[2]
        + np.dot(h_max_past_s[2, :], gammas)
        + np.dot(H[2, :, -1], nus)
        + np.sum(((-1 * (a_matrix[:, :]))[E].T) * delta_H[:, :, 2])
    )

    h_hat = h.copy()
    for j in range(D):
        h_hat[2, j, : np.where(h_hat[2, j] == h_max[2, j])[0][0]] = h_max[2, j]

    t_index = np.floor(T - s / h_dt).astype(int)
    H_hat = get_approx_to_f(T - s, h_dt, H_hat, 2, None)

    ub += np.dot(H_hat, a)
    return ub


def return_H_hat(h, H, h_grid):
    D = h.shape[0]
    h_max = np.max(h, axis=-1)
    t_max = np.array(
        [
            [int(np.where(h[i, j] == h_max[i, j])[0][0]) for j in range(D)]
            for i in range(D)
        ]
    )

    H_hat = H.copy()
    for i in range(D):
        for j in range(D):
            if t_max[i, j] != 0:
                H_hat[i, j, : t_max[i, j]] = h_max[i, j] * h_grid[: t_max[i, j]]
            H_hat[i, j, t_max[i, j] :] = (
                H[i, j, t_max[i, j] :]
                + h_grid[t_max[i, j]] * h_max[i, j]
                - H[i, j, t_max[i, j]]
            )
    return H_hat


def return_history2(
    s_0,
    T,
    p_dt,
    kernel,
    plist,
    gammas,
    h,
    H,
    h_grid,
    history,
    E,
    hyper_index,
    video_index,
    run_index,
    to_log=True,
    max_time=600,
    logfile_label="pmbpthinning",
):

    logging.basicConfig(
        filename=f"log/{video_index}_{logfile_label}_{hyper_index}_{run_index}.log",
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        force=True,
    )

    start = perf_counter()

    n_list = np.zeros(shape=2)

    a_collector = np.empty(shape=[3, 0])
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
        lambda_star = return_ub2_at_s(s, history, plist, gammas, h, H, h_grid, E, T=T)

        u = np.random.uniform()
        w = -np.log(u) / lambda_star

        s = s + w

        if s > T:
            break

        rd = np.random.uniform()
        current_lambda = return_xi2_at_s(
            s, history, plist, gammas, h, H, h_grid, E, p_dt=p_dt
        )

        if (rd * lambda_star) <= current_lambda:
            history[2].append(s)
            logging.info(f"run#{run_index}, hyperindex#{hyper_index}, {s}. accept.")
        #             print(f"run#{run_index}", s,"accept", current_lambda.round(2), lambda_star.round(2))
        else:
            logging.info(f"run#{run_index}, hyperindex#{hyper_index}, {s}. reject.")
            #             print(f"run#{run_index}", "\t", s,"reject", current_lambda.round(2), lambda_star.round(2))
            continue

    if len(history[2]) != 0:
        if history[2][-1] > T:
            history[2] = history[2][:-1]

    if too_long:
        print("Takes too long. Perform Poisson approximation.")
        history[2] = [x for x in history[2] if x < s_0]
        pts = np.array(history[2])
        # if too long, use history of last 15 days as sample instead.
        rate = len(pts[(pts >= s_0 - 15) & (pts < s_0)]) / 15

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
        #                 print(f"poisson. {s}")
        return history
    else:
        return history


# ex: a = 1, b = 90; to get volume at 1, 2, ..., 90
def get_volume_per_unit_bet_a_b(a, b, history, plist, gammas, h, H, h_grid, E):

    D = len(history)
    if a == 0:
        initial = [
            -1 * gammas
            + return_Xi_at_s(
                1,
                history,  # dataset: interval-censored in the ic-dims
                plist,
                gammas,
                h,
                H,
                h_grid,
                E,
            )
        ]
    else:
        initial = np.empty(shape=(0, D))

    return np.vstack(
        [
            initial,
            np.diff(
                [
                    return_Xi_at_s(
                        x,
                        history,  # dataset: interval-censored in the ic-dims
                        plist,
                        gammas,
                        h,
                        H,
                        h_grid,
                        E,
                    )
                    for x in range(max([a, 1]), b + 1)
                ],
                axis=0,
            ),
        ]
    )


def get_volume_per_unit_bet_a_b_recompute_h(
    a,
    b,
    history,
    plist,
    gammas,
    E,
    p_dt=1,
    h_dt=0.01,
    h_tol=1e-6,
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
):
    D = len(history)
    kp = np.array(plist[:-D]).reshape(2, D, D)

    h_grid = np.arange(0, b + 0.1 * h_dt, h_dt)
    h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, b, h_tol)

    if a == 0:
        initial = [
            -1 * gammas
            + return_Xi_at_s(
                1,
                history,  # dataset: interval-censored in the ic-dims
                plist,
                gammas,
                h,
                H,
                h_grid,
                E,
            )
        ]
    else:
        initial = np.empty(shape=(0, D))

    return np.vstack(
        [
            initial,
            np.diff(
                [
                    return_Xi_at_s(
                        x,
                        history,  # dataset: interval-censored in the ic-dims
                        plist,
                        gammas,
                        h,
                        H,
                        h_grid,
                        E,
                    )
                    for x in range(max([a, 1]), b + 1)
                ],
                axis=0,
            ),
        ]
    )


def sample_forward(
    s_0,
    T,
    hist,
    plist,
    gammas,
    h,
    H,
    h_grid,
    E,
    p_dt,
    kernel,
    hyper_index,
    video_index,
    run_index,
    max_time=600,
    logfile_label="pmbpthinning",
):
    forward_history = return_history2(
        s_0,
        T,
        p_dt,
        kernel,
        plist,
        gammas,
        h,
        H,
        h_grid,
        hist,
        E,
        hyper_index,
        video_index,
        run_index,
        max_time=max_time,
        logfile_label=logfile_label,
    )

    volumes = get_volume_per_unit_bet_a_b(s_0, T, hist, plist, gammas, h, H, h_grid, E)
    print(f"...done with sample #{run_index}")

    return volumes, hist


#### for final fitting ####


def sample_forward_once(
    i,
    s_0,
    T,
    input_history,
    plist,
    gammas,
    video_index,
    hyper_index,
    actual_volume,
    E,
    p_dt=1,
    h_dt=0.01,
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
    h_tol=1e-6,
    max_time=600,
    logfile_label="thin",
):

    D = len(input_history)
    kp = np.array(plist[:-D]).reshape(2, D, D)

    h_grid = np.arange(0, T + 0.1 * h_dt, h_dt)
    h, H, flag = return_h_and_H(kernel, kernel_integral, kp, h_grid, E, h_dt, T, h_tol)

    np.random.seed(i)
    hist = copy.deepcopy(input_history)
    volumes, hist = sample_forward(
        s_0,
        T,
        hist,
        plist,
        gammas,
        h,
        H,
        h_grid,
        E,
        p_dt,
        kernel,
        hyper_index,
        video_index,
        i,
        max_time=max_time,
        logfile_label=logfile_label,
    )
    return volumes, hist


def aggregate_sample_forwards(
    collector,
    s_0,
    T,
    input_history,
    plist,
    gammas,
    video_index,
    hyper_index,
    actual_volume,
    E,
    p_dt=1,
    h_dt=0.01,
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
    h_tol=1e-6,
    folder="hypertuning",
):
    mean_volume = np.mean(collector, axis=0).T
    val_view_volume = np.sum(mean_volume[0])

    rmse_error = get_rmse_error(val_view_volume, actual_volume)

    pickle.dump(
        [hyper_index, rmse_error, val_view_volume, collector],
        open(f"output/{video_index}_hypertuning.p", "wb"),
    )
    return [hyper_index, rmse_error, val_view_volume, collector]


def complete_mbp_percentile_comparison(
    s_0,
    T,
    input_history,
    plist,
    gammas,
    video_index,
    hyper_index,
    actual_volume,
    E,
    folder="hypertuning",
    p_dt=1,
    h_dt=0.01,
    kernel=exponential_kernel,
    kernel_integral=exponential_kernel_integral,
    h_tol=1e-6,
):

    D = len(input_history)
    kp = np.array(plist[:-D]).reshape(2, D, D)

    h_grid = np.arange(0, T + 0.1 * h_dt, h_dt)
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
        h_dt=h_dt,
    )

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
        h_dt=h_dt,
    )

    mean_volume = end_Xi - start_Xi
    val_view_volume = mean_volume[0]

    rmse_error = get_rmse_error(val_view_volume, actual_volume)
    pickle.dump(
        [hyper_index, rmse_error, val_view_volume],
        open(f"output/{video_index}_hypertuning.p", "wb"),
    )

    return [hyper_index, rmse_error, val_view_volume]
