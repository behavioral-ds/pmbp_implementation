import numpy as np
from time import perf_counter
from scipy.signal import fftconvolve

def exponential_kernel(t, theta, kappa):
    return kappa * theta * np.exp(-theta * t)

def exponential_kernel_integral(t, theta, kappa):
    return kappa * (1 - np.exp(-theta * t))

def return_kernel_matrix_at_t(t, kernel, parameters):
    D = len(parameters[0][0])
    phi = np.zeros(shape=(D,D))
    thetas, kappas = parameters
    for i in range(D):
        for j in range(D):
            theta = thetas[i,j]
            kappa = kappas[i,j]
            phi[i,j] = kernel(t, theta, kappa)
    return phi

def return_kernel_matrix(x_range, pointwise_phi_function):
    return np.moveaxis(np.array([pointwise_phi_function(x) for x in x_range]), 0, -1)

def pointwise_nu(t, nu_vec):
    return nu_vec*np.ones(shape=len(nu_vec))

def pointwise_nu_integral(t, nu_vec):
    return nu_vec*t*np.ones(shape=len(nu_vec))

def convolve_matrices(A, B, increment):
    D = A.shape[0]
    timerange = A.shape[-1]

    convolution = np.zeros(shape=(D, D, timerange))
    for i in range(D):
        for j in range(D):
            convolution[i,j,:] = np.sum([fftconvolve(A[i, k, :], B[k, j, :])[:timerange] for k in range(D)], axis=0)
    convolution *= increment
    return convolution

def get_approx_to_f(t_arr, h_dt, f, i = None, j = None):
    t_index = np.floor(t_arr / h_dt).astype(int)
    if i is None and j is not None and type(j) == type([1]): # list
        return f[:,j][:,:,t_index]
    elif i is None and j is not None and type(j) == type(1): # float
        return f[:,j][:,t_index]
    elif i is not None and j is None and type(i) == type([1]): # list
        return f[i,:][:,:,t_index]
    elif i is not None and j is None and type(i) == type(1): # float
        return f[i,:][:,t_index]
    
    elif i is not None and j is not None and type(j) == type([1]):
        return f[i,j][:, t_index]
    elif i is not None and j is not None and type(j) == type(1) and type(i) == type(1):
        return f[i,j, t_index]
    elif i is None and j is None:
        return f[:,:][:,:,t_index]

def interval_censor_dimension(dim_history, T, partition_length = 1):
    cs, edges = np.histogram(dim_history, bins=np.arange(0, T+0.01, partition_length))
    censored_data = []
    for i in range(len(cs)):
        censored_data.append([cs[i], [edges[i], edges[i+1]]])
    return censored_data

def convert_point_to_interval_histories(history, ic_dims, T, partition_length = 1):
    processed_histories = []
    for i in range(len(history)):
        if i in ic_dims:
            processed_histories.append(interval_censor_dimension(history[i], T, partition_length))
        else:
            processed_histories.append(history[i])
    return processed_histories

def preprocess_history(history, ic_dims):
    labeled_points = []
    labeled_censors = []
    for data_dimension, data_list in enumerate(history):
        if data_dimension in ic_dims:
            for pair in data_list:
                c, endpoints = pair
                labeled_censors.append((endpoints[0], ("o", data_dimension)))
                labeled_censors.append((endpoints[1], ("o", data_dimension)))

        else:
            for t in data_list:
                labeled_points.append((t, ("T", data_dimension)))
                
    labeled_points.sort(key = lambda x: x[0])

    return labeled_points, labeled_censors

def get_effective_history(history, ic_dims, T, p_dt = 1):
    
    D = len(history)
    
    # convert history into a list of timestamps to iterate over (+ discretization for hawkes calc)
    labeled_points, labeled_censors = preprocess_history(history, ic_dims)
    
    grid = np.arange(0, T+0.1*p_dt, p_dt)
    grid_labeled = list([(x, ("P", -1)) for x in grid])
                                                         
    labeled_points_dict = dict(labeled_points)
    labeled_censors_dict = dict(labeled_censors)
    grid_dict = dict(grid_labeled)
    
    # split labeled_censors_dict by dim
    c_dicts = []
    for d in range(D):
        c_dat = [x for x in labeled_censors if x[1][1] == d]
        c_dicts.append(dict(c_dat))
                                                         
    labeled_poi = []
    poi_list = list(set(list(labeled_points_dict.keys()) + list(labeled_censors_dict.keys()) + list(grid_dict.keys())))
    poi_list.sort()
    
    for poi in poi_list:
        poi_roles = []
        for p_dict in [labeled_points_dict, grid_dict] + c_dicts:
            try:
                poi_roles.append(p_dict[poi])
            except KeyError:
                continue
        labeled_poi.append((poi, poi_roles))
                                                         
    return labeled_poi

def return_h_and_H(
        kernel,
        kernel_integral,
        kernel_parameters,
        x_range,
        E,
        h_dt,
        T,
        gamma = 1e-4
    ):
    flag = False
    
    D = len(kernel_parameters[0])
    E_c = [i for i in range(D) if i not in E]
    pointwise_phi = lambda t: return_kernel_matrix_at_t(t, kernel, kernel_parameters)
    pointwise_Phi = lambda t: return_kernel_matrix_at_t(t, kernel_integral, kernel_parameters)
    
    phi = return_kernel_matrix(x_range, pointwise_phi)
    Phi = return_kernel_matrix(x_range, pointwise_Phi)

    phi_E = phi.copy()
    phi_E[:,E_c,:] = 0
    
    Phi_E = Phi.copy()
    Phi_E[:,E_c,:] = 0

    h = phi_E.copy()
    Delta_h = phi_E.copy()
    
    running = 1e9
    while True:
            
        Delta_h = convolve_matrices(Delta_h, phi_E, h_dt)
        h = h + Delta_h
        
        if (np.max(np.abs(Delta_h)) - running > 100):
            return np.nan, np.nan, "divergent"

        if (np.max(np.abs(Delta_h)) <= gamma):
            break
        if np.isnan(np.max(np.abs(Delta_h))):
            flag = True # getting h leads to nan, probably too large theta
            print("h", "nan")
            break
        
        running = np.max(np.abs(Delta_h))

    H = Phi_E + convolve_matrices(Phi_E, h, h_dt)
    H[:,E_c,:] = 0
    return h, H, flag

def return_derivative_of_phi(t, kernel_parameters, i,j, E_c, var = "kappa"):
    theta, kappa = kernel_parameters
    deriv = np.zeros(shape=(len(theta),len(theta),len(t)))
    if var == "kappa":
        deriv[i,j,:] = theta[i,j] * np.exp(-theta[i,j] * t)
    elif var == "theta":
        deriv[i,j,:] = kappa[i,j] * np.exp(-theta[i,j] * t) * (1 - theta[i,j] * t)
    deriv[:,E_c,:] = 0
    return deriv

def return_derivative_of_Phi(t, kernel_parameters, i,j, E_c, var = "kappa"):
    theta, kappa = kernel_parameters
    deriv = np.zeros(shape=(len(theta),len(theta),len(t)))
    # exponential
    if var == "kappa":
        deriv[i,j,:] = 1 - np.exp(-theta[i,j] * t)
    elif var == "theta":
        deriv[i,j,:] = kappa[i,j] * t * np.exp(-theta[i,j] * t)
    deriv[:,E_c,:] = 0
    return deriv

def return_gradient_of_h_recursive(
        kernel,
        kernel_parameters,
        return_derivative_of_phi,
        x_range,
        E,
        h_dt,
        T,
        gamma,
        N,
        return_at_N = False
    ):
    flag = False
    
    D = len(kernel_parameters[0])
    E_c = [i for i in range(D) if i not in E]
    pointwise_phi = lambda t: return_kernel_matrix_at_t(t, kernel, kernel_parameters)
    
    phi = return_kernel_matrix(x_range, pointwise_phi)
    phi_E = phi.copy()
    phi_E[:,E_c,:] = 0
    
    deriv_phi_array = np.zeros(shape=(2,D,D, D, D, len(x_range)),dtype=object)
    h_grad = np.zeros(shape=(2,D,D, D, D, len(x_range)))
    A = np.zeros(shape=(2,D, D, D,D,len(x_range)))
    B = np.zeros(shape=(2,D, D, D,D,len(x_range)))
    
    for index, var in enumerate(['theta', 'kappa']):
        for i in range(D):
            for j in range(D):
                deriv_phi_array[index, i, j, :] = return_derivative_of_phi(x_range, kernel_parameters, i,j, E_c, var)
                A[index,i,j,:] = deriv_phi_array[index, i, j, :]
                B[index,i,j,:] = deriv_phi_array[index, i, j, :]
                h_grad[index, i, j, :] = A[index,i,j,:]
    
    n = 2
    running = 1e9
    time_start = perf_counter()
    while True:                    
        additionals = []
        for index, var in enumerate(['theta', 'kappa']):
            for i in range(D):
                for j in range(D):      
                    B[index,i,j,:] = convolve_matrices(phi_E, B[index,i,j,:], h_dt)
                    A[index,i,j,:] = B[index,i,j,:] + convolve_matrices(A[index,i,j,:], phi_E, h_dt)
                    h_grad[index, i, j, :] += A[index,i,j,:]
                            
                    additionals.append(np.max(np.abs(A[index,i,j,:])))
                
        if np.max(additionals) <= gamma:
            break
        if np.isnan(np.max(additionals)):
            flag = True # getting h leads to nan, probably too large theta
            break
        if not return_at_N: # don't return just at N
            if (np.max(additionals) - running > 100) or (n == N) or ((perf_counter() - time_start) > 30):
                return np.nan, "divergent"
        else:
            if (n == N) or (np.max(additionals) - running > 100):
                return h_grad, "return_at_N"
                
        running = np.max(additionals)
        n += 1
        
    return h_grad, flag

def return_gradient_of_H(
    kernel_integral,
    kernel_parameters,
    return_derivative_of_Phi,
    h, 
    h_grad,
    x_range,
    E,
    h_dt):

    D = len(kernel_parameters[0])
    E_c = [i for i in range(D) if i not in E]
    pointwise_Phi = lambda t: return_kernel_matrix_at_t(t, kernel_integral, kernel_parameters)
    
    Phi_E = return_kernel_matrix(x_range, pointwise_Phi)
    Phi_E[:,E_c,:] = 0

    H_grad = np.zeros(shape=(2,D,D, D, D, len(x_range)))
    for index, var in enumerate(['theta', 'kappa']):
        for i in range(D):
            for j in range(D):
                deriv_Phi_E = return_derivative_of_Phi(x_range, kernel_parameters, i,j, E_c, var)
                H_grad[index, i, j, :] = deriv_Phi_E + convolve_matrices(h_grad[index, i, j, :], Phi_E, h_dt) + convolve_matrices(h, deriv_Phi_E, h_dt)

    return H_grad

def return_derivative_of_phi_at_t(t, params, param_type, i,j):
    theta, kappa = params
    D = len(theta)
    deriv = np.zeros(shape=(D,D))
    if param_type == 1: # kappa
        deriv[i,j] = theta[i,j] * np.exp(-theta[i,j] * t)
    elif param_type == 0: # theta
        deriv[i,j] = kappa[i,j] * np.exp(-theta[i,j] * t) * (1 - theta[i,j] * t)
    return deriv

def return_derivative_of_Phi_at_t(t, params, param_type, i,j):
    theta, kappa = params
    D = len(theta)
    deriv = np.zeros(shape=(D,D))
    # exponential
    if param_type == 1: # kappa
        deriv[i,j] = 1 - np.exp(-theta[i,j] * t)
    elif param_type == 0: # theta
        deriv[i,j] = kappa[i,j] * t * np.exp(-theta[i,j] * t)
    return deriv

def return_nll_without_gradients(
    kernel,
    kernel_integral,
    x,
    gammas,
    pointwise_nu,
    pointwise_nu_integral,
    history, # dataset: interval-censored in the ic-dims
    E, # the mbp dims
    h, 
    H,
    h_grid,
    p_dt,
    h_dt,
    T,
    ll_weights_map,
    nu_regularization):
    
    params = [x[:9].reshape(3,3), x[9:18].reshape(3,3)]
    nus = x[-3:]

    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_integral_at_t = lambda t: pointwise_nu_integral(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(np.stack([return_nu_at_t(i) for i in arr]), 0, -1)
    return_nu_integral_over_time = lambda arr: np.moveaxis(np.stack([return_nu_integral_at_t(i) for i in arr]), 0, -1)
    
    D = len(params[0])
    ic_dims = []
    for j in range(len(history)):
        if (len(history[j]) != 0) and type(history[j][0]) == type([]):
            ic_dims.append(j)

    pp_dims = [x for x in range(D) if x not in ic_dims]
    Ec = [x for x in range(D) if x not in E]
    
    ic_compensators = [[] for i in range(D)] # only non-empty if IC dim
    
    ppll = [0] * D # only nonzero if PP dimension

    t_arr = get_effective_history(history, ic_dims, T, p_dt)
    a_matrix = np.zeros(shape=(D, len(t_arr)))
    A_matrix = np.zeros(shape=(D, len(t_arr)))
    
    for index, t_data in enumerate(t_arr):
        t, t_roles = t_data
        
        a, A = np.zeros(shape=D), np.zeros(shape=D)
        history_prior_to_t = t_arr[:index]
        for t_p, roles_p in history_prior_to_t:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                    a += return_kernel_matrix_at_t(t - t_p, kernel, params)[:, dim_p]
                    A += return_kernel_matrix_at_t(t - t_p, kernel_integral, params)[:, dim_p]
                    
                    break # only one event can happen at any instant
                    
        a_matrix[:, index] = a
        A_matrix[:, index] = A
        
        for role_t, dim_t in t_roles:                                
            time_points_prior_to_t = np.array([x[0] for x in t_arr][:index])
            time_points_up_to_t = np.array(list(time_points_prior_to_t) + [t])
            
            if role_t == "T": # is a PP dim
                xi = return_nu_at_t(t)[dim_t] + a[dim_t]
                                
                xi += np.sum(get_approx_to_f(t, h_dt, h, dim_t, None) * gammas)

                if index != 0 and len(E) != 0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim_t, E),axis=1).T
                    xi += np.sum(((-1*(return_nu_over_time(time_points_prior_to_t) + a_matrix[:,:index]))[E].T) * delta_H)
                
                if xi > 0:
                    ppll[dim_t] += -np.log(xi)
                    

            if role_t == "o": # is an IC dim
                Xi = gammas[dim_t] + return_nu_integral_at_t(t)[dim_t] + A[dim_t]

                if index != 0 and len(E) != 0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim_t, E),axis=1).T
                    Xi += np.sum(((-1*(gammas[:,None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * delta_H)
                ic_compensators[dim_t].append(Xi)            



        if index == len(t_arr) - 1:

            time_points_prior_to_t = np.array([x[0] for x in t_arr][:index])
            time_points_up_to_t = np.array(list(time_points_prior_to_t) + [t])
          
            for dim in pp_dims:
                Xi = gammas[dim] + return_nu_integral_at_t(t)[dim] + A[dim]

                if len(E)!=0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim, E),axis=1).T
                    Xi += np.sum(((-1*(gammas[:,None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * delta_H)

                ppll[dim] += Xi

    iclls = []
    for dim in ic_dims:
        compensators = ic_compensators[dim]
        c_vals = np.array([x[0] for x in history[dim]])
        diffs = np.diff(compensators)
        icll = np.sum(diffs) - np.sum(c_vals * np.log(diffs))
        iclls.append(ll_weights_map[dim] * icll)
    
    ppll = [ll_weights_map[i]*ppll[i] for i in range(D)]
    
    nll = np.sum(ppll) + np.sum(iclls)
    nll += nu_regularization * np.sum(nus)

    return nll

def return_nll_with_gradients(
    kernel,
    kernel_integral,
    params,
    gammas,
    nus,
    pointwise_nu,
    pointwise_nu_integral,
    history, # dataset: interval-censored in the ic-dims
    E, # the mbp dims
    h, 
    H,
    h_grad,
    H_grad,
    h_grid,
    p_dt,
    h_dt,
    T,
    ll_weights_map,
    nu_regularization):
    
    return_nu_at_t = lambda t: pointwise_nu(t, nus)
    return_nu_integral_at_t = lambda t: pointwise_nu_integral(t, nus)
    return_nu_over_time = lambda arr: np.moveaxis(np.stack([return_nu_at_t(i) for i in arr]), 0, -1)
    return_nu_integral_over_time = lambda arr: np.moveaxis(np.stack([return_nu_integral_at_t(i) for i in arr]), 0, -1)
    
    D = len(params[0])
    ic_dims = []
    for j in range(len(history)):
        if (len(history[j]) != 0) and type(history[j][0]) == type([]):
            ic_dims.append(j)

    pp_dims = [x for x in range(D) if x not in ic_dims]
    Ec = [x for x in range(D) if x not in E]
    
    ic_compensators = [[] for i in range(D)] # only non-empty if IC dim
    ic_compensators_grad = [[[[[] for _ in range(D)] for _ in range(D)] for _ in range(D)] for _ in range(2)]
    ic_compensators_exo_grad = [[[] for _ in range(D)] for _ in range(D)]
    
    ppll = [0] * D # only nonzero if PP dimension
    ppll_grads = np.zeros(shape = (2,D,D))
    ppll_exo_grads = np.zeros(shape = (D))

    t_arr = get_effective_history(history, ic_dims, T, p_dt)
    a_matrix = np.zeros(shape=(D, len(t_arr)))
    A_matrix = np.zeros(shape=(D, len(t_arr)))
    
    a_grad_matrix = np.zeros(shape=(2, D, D, D, len(t_arr)))
    A_grad_matrix = np.zeros(shape=(2, D, D, D, len(t_arr)))
    
    
    for index, t_data in enumerate(t_arr):
        time_start = perf_counter()

        
        t, t_roles = t_data
        
#         print(index, len(t_arr))
        
        a, A = np.zeros(shape=D), np.zeros(shape=D)
        a_grad = np.zeros(shape=(2,D,D,D))
        A_grad = np.zeros(shape=(2,D,D,D))
        history_prior_to_t = t_arr[:index]
        for t_p, roles_p in history_prior_to_t:
            for role_p, dim_p in roles_p:
                if role_p == "T" and dim_p in Ec: # is a Hawkes event?
                    a += return_kernel_matrix_at_t(t - t_p, kernel, params)[:, dim_p]
                    A += return_kernel_matrix_at_t(t - t_p, kernel_integral, params)[:, dim_p]
                    
                    for param_type in range(2):
                        for param_i in range(D):
                            for param_j in range(D):
                                a_grad[param_type, param_i, param_j] += return_derivative_of_phi_at_t(t - t_p, params, param_type, param_i, param_j)[:, dim_p]
                                A_grad[param_type, param_i, param_j] += return_derivative_of_Phi_at_t(t - t_p, params, param_type, param_i, param_j)[:, dim_p]
                    break # only one event can happen at any instant
                    
        a_matrix[:, index] = a
        A_matrix[:, index] = A
        for param_type in range(2):
            for param_i in range(D):
                for param_j in range(D):
                    a_grad_matrix[param_type, param_i, param_j, :, index] = a_grad[param_type, param_i, param_j]
                    A_grad_matrix[param_type, param_i, param_j, :, index] = A_grad[param_type, param_i, param_j]
                    
        
        for role_t, dim_t in t_roles:                                
            time_points_prior_to_t = np.array([x[0] for x in t_arr][:index])
            time_points_up_to_t = np.array(list(time_points_prior_to_t) + [t])
            
            if role_t == "T": # is a PP dim
                xi = return_nu_at_t(t)[dim_t] + a[dim_t]
                                
                xi += np.sum(get_approx_to_f(t, h_dt, h, dim_t, None) * gammas)

                if index != 0 and len(E) != 0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim_t, E),axis=1).T
                    xi += np.sum(((-1*(return_nu_over_time(time_points_prior_to_t) + a_matrix[:,:index]))[E].T) * delta_H)
                
                if xi > 0:
                    ppll[dim_t] += -np.log(xi)
                    
                # calculate gradients
                for param_type in range(2):
                    for param_i in range(D):
                        for param_j in range(D):
                            xi_grad = a_grad[param_type, param_i, param_j, dim_t]

                            xi_grad += np.sum(get_approx_to_f(t, h_dt, h_grad[param_type, param_i, param_j], dim_t, None) * gammas)

                            if index != 0 and len(E) != 0:

                                h_times_kernel_grad = np.sum(((-1*(a_grad_matrix[param_type, param_i, param_j, :,:index]))[E].T) * delta_H)
                                h_grad_times_exo_and_kernel = np.sum(((-1*(return_nu_over_time(time_points_prior_to_t) + a_matrix[:,:index]))[E].T) * np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H_grad[param_type, param_i, param_j], dim_t, E),axis=1).T)

                                xi_grad += h_times_kernel_grad + h_grad_times_exo_and_kernel
                            ppll_grads[param_type, param_i, param_j] += ((-1 / xi) * xi_grad) * ll_weights_map[dim_t]
                        
                        # exo grads
                        if param_type == 0:
                            xi_grad_exo = 0
                            xi_grad_exo += float(dim_t==param_i)
                            if index != 0 and len(E) != 0:
                                xi_grad_exo += get_approx_to_f(t, h_dt, H, dim_t, param_i)
                            ppll_exo_grads[param_i] += ((-1 / xi) * xi_grad_exo) * ll_weights_map[dim_t]
            
            if role_t == "o": # is an IC dim
                Xi = gammas[dim_t] + return_nu_integral_at_t(t)[dim_t] + A[dim_t]

                if index != 0 and len(E) != 0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim_t, E),axis=1).T
                    Xi += np.sum(((-1*(gammas[:,None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * delta_H)
                ic_compensators[dim_t].append(Xi)            

                # calculate gradients
                for param_type in range(2):
                    for param_i in range(D):
                        for param_j in range(D):
                            Xi_grad = A_grad[param_type, param_i, param_j, dim_t]

                            if index != 0 and len(E) != 0:
                                h_times_Kernel_grad = np.sum(((-1*(A_grad_matrix[param_type, param_i, param_j, :,:index]))[E].T) * delta_H)

                                h_grad_times_Exo_and_Kernel = np.sum(((-1*(gammas[:, None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H_grad[param_type, param_i, param_j], dim_t, E),axis=1).T)

                                Xi_grad += h_times_Kernel_grad + h_grad_times_Exo_and_Kernel
                            ic_compensators_grad[param_type][param_i][param_j][dim_t].append(Xi_grad)
                        
                        if param_type == 0:
                            Xi_grad_exo = 0
                            Xi_grad_exo += t * float(dim_t==param_i)
                            if index != 0 and len(E) != 0:
                                Xi_grad_exo += np.sum(
                                    -1 * (time_points_prior_to_t) * np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim_t, param_i))
                                )
                            ic_compensators_exo_grad[param_i][dim_t].append(Xi_grad_exo)


        if index == len(t_arr) - 1:

            time_points_prior_to_t = np.array([x[0] for x in t_arr][:index])
            time_points_up_to_t = np.array(list(time_points_prior_to_t) + [t])
          
            for dim in pp_dims:
                Xi = gammas[dim] + return_nu_integral_at_t(t)[dim] + A[dim]

                if len(E)!=0:
                    delta_H = np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim, E),axis=1).T
                    Xi += np.sum(((-1*(gammas[:,None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * delta_H)

                ppll[dim] += Xi
                
                for param_type in range(2): 
                    for param_i in range(D):
                        for param_j in range(D):
                            Xi_grad = A_grad[param_type, param_i, param_j, dim]

                            if len(E)!=0:
                                h_times_Kernel_grad = np.sum(((-1*(A_grad_matrix[param_type, param_i, param_j, :,:index]))[E].T) * np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim, E),axis=1).T)

                                h_grad_times_Exo_and_Kernel = np.sum(((-1*(gammas[:, None] + return_nu_integral_over_time(time_points_prior_to_t) + A_matrix[:,:index]))[E].T) * np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H_grad[param_type, param_i, param_j], dim, E),axis=1).T)

                                Xi_grad += h_times_Kernel_grad + h_grad_times_Exo_and_Kernel
                            ppll_grads[param_type, param_i, param_j] +=  ll_weights_map[dim] * Xi_grad
                        
                        if param_type == 0:
                            Xi_grad_exo = 0        
                            Xi_grad_exo += t * float(dim==param_i)
                            if index != 0 and len(E) != 0:
                                Xi_grad_exo += np.sum(
                                    -1 * time_points_prior_to_t * \
                                    np.diff(get_approx_to_f(t - time_points_up_to_t, h_dt, H, dim, param_i))
                                )
                            ppll_exo_grads[param_i] +=  ll_weights_map[dim] * Xi_grad_exo
        
#         print(t, index, perf_counter() - time_start)
    
                            
    iclls = []
    icll_grads = np.zeros(shape = (2,D,D))
    icll_exo_grads = np.zeros(shape = D)
    for dim in ic_dims:
        compensators = ic_compensators[dim]
        c_vals = np.array([x[0] for x in history[dim]])
        diffs = np.diff(compensators)
        icll = np.sum(diffs) - np.sum(c_vals * np.log(diffs))
        iclls.append(ll_weights_map[dim] * icll)
    
        
        for param_type in range(2):
            for param_i in range(D):
                for param_j in range(D):
                    compensator_grads = ic_compensators_grad[param_type][param_i][param_j][dim]
                    diffs_grad = np.diff(compensator_grads)
                    icll_grads[param_type, param_i, param_j] += ll_weights_map[dim] * np.sum(diffs_grad * (1 - c_vals / diffs))

                # exo grads
                if param_type == 0:
                    compensator_exo_grads = ic_compensators_exo_grad[param_i][dim]
                    diffs_exo_grad = np.diff(compensator_exo_grads)
                    icll_exo_grads[param_i] += ll_weights_map[dim] * np.sum(diffs_exo_grad * (1 - c_vals / diffs))
                    
    ppll = [ll_weights_map[i]*ppll[i] for i in range(D)]
    nll = np.sum(ppll) + np.sum(iclls)
    nll += nu_regularization * np.sum(nus)

    grads = ppll_grads + icll_grads
    grads_exo = ppll_exo_grads + icll_exo_grads
    
    for i in range(D):
        grads_exo[i] += nu_regularization
    
    return nll, np.hstack([grads.reshape(-1), grads_exo.reshape(-1)])