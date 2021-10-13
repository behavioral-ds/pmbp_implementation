import ipopt, logging
import numpy as np
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
    interval_censor_dimension,
    convert_point_to_interval_histories,
    preprocess_history,
    get_effective_history,
    return_h_and_H,
    return_derivative_of_phi,
    return_derivative_of_Phi,
    return_gradient_of_h_recursive,
    return_gradient_of_H,
    return_derivative_of_phi_at_t,
    return_derivative_of_Phi_at_t,
    return_nll_without_gradients,
    return_nll_with_gradients
)

from pmbp_utils import get_spectral_radius_from_flat_params, get_constraintEcEc, get_constraintCross

class PMBP(object):
    def __init__(self, history, plist, gammas, pbs_index, hyparam_x0_index, dim_weights, nu_reg, E, logfit_label):
        self.D = len(history)
        self.kernel = exponential_kernel
        self.kernel_integral = exponential_kernel_integral
        self.T = len(history[0])
        self.E = E
        self.Ec = [x for x in range(self.D) if x not in self.E]
        self.h_dt = 0.01
        self.p_dt = 1
        self.h_tol = 1e-6
        self.h_grad_tol = 1e-4
        self.history = history
        self.start_time = perf_counter()
        self.hyparam_x0_index = hyparam_x0_index
        
        self.gammas = gammas

        self.ll_weights_map = dim_weights
        self.nu_regularization = nu_reg
        
        self.logfit_label = logfit_label
        
        logging.basicConfig(filename=f"log/{logfit_label}_{pbs_index}_{hyparam_x0_index}.log", level=logging.INFO, format='%(asctime)s | %(message)s', force=True)
        
        # calculate gradient at initial point. self.grad will cache values during objective evaluation
        D = len(history)
        kp = np.array(plist[:-self.D]).reshape(2,self.D,self.D)
        h_grid = np.arange(0, self.T+0.1*self.h_dt, self.h_dt)
        h, H, flag = return_h_and_H(self.kernel, self.kernel_integral, kp, h_grid, self.E, self.h_dt, self.T, self.h_tol)

        logging.info(f"PLIST: {str(plist)}, hyparam_x0_index: {hyparam_x0_index}")
        
        h_grad, flag2 = return_gradient_of_h_recursive(
            self.kernel,
            kp,
            return_derivative_of_phi,
            h_grid,
            self.E,
            self.h_dt,
            self.T,
            self.h_grad_tol,
            1000
        )
        gammas = self.gammas
        nus = plist[-self.D:]
 
        H_grad = return_gradient_of_H(
            self.kernel_integral,
            kp,
            return_derivative_of_Phi,
            h, 
            h_grad,
            h_grid,
            self.E,
            self.h_dt)
        
        ll, grad = return_nll_with_gradients(
            self.kernel,
            self.kernel_integral,
            kp,
            gammas,
            nus,
            pointwise_nu,
            pointwise_nu_integral,
            history, # dataset: interval-censored in the ic-dims
            self.E, # the mbp dims
            h, 
            H,
            h_grad,
            H_grad,
            h_grid,
            self.p_dt,
            self.h_dt,
            self.T,
            self.ll_weights_map,
            self.nu_regularization)
        self.grad = grad
                

    def objective(self, plist):
        start = perf_counter()
        D = len(self.history)
#         print(get_spectral_radius_from_flat_params(plist, self.D), "PLIST", plist)
        sr = get_spectral_radius_from_flat_params(plist, self.D, self.E)
        if sr > 0.99:
            print(f"\tFAIL - constraint, (SR = {sr}), hyparam_x0_index: {self.hyparam_x0_index}")
            logging.info(f"\tFAIL - constraint, (SR = {sr}), hyparam_x0_index: {self.hyparam_x0_index}")
            return np.nan

        kp = np.array(plist[:-self.D]).reshape(2,self.D,self.D)

        h_grid = np.arange(0, self.T+0.1*self.h_dt, self.h_dt)
        h, H, flag = return_h_and_H(self.kernel, self.kernel_integral, kp, h_grid, self.E, self.h_dt, self.T, self.h_tol)
        
        print(f"\tcalculated h & H. calculating h_grad..., hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"\tcalculated h & H. calculating h_grad..., hyparam_x0_index: {self.hyparam_x0_index}")
        
        if flag != "divergent":
            h_grad, flag2 = return_gradient_of_h_recursive(
                self.kernel,
                kp,
                return_derivative_of_phi,
                h_grid,
                self.E,
                self.h_dt,
                self.T,
                self.h_grad_tol,
                50,
                True
            )
            
            print(f"\tcalculated h_grad. calculating H_grad..., hyparam_x0_index: {self.hyparam_x0_index}")
            logging.info(f"\tcalculated h_grad. calculating H_grad..., hyparam_x0_index: {self.hyparam_x0_index}")
            
            if flag2 != "divergent":
                H_grad = return_gradient_of_H(
                    self.kernel_integral,
                    kp,
                    return_derivative_of_Phi,
                    h, 
                    h_grad,
                    h_grid,
                    self.E,
                    self.h_dt)
                print(f"\tcalculated H_grad, hyparam_x0_index: {self.hyparam_x0_index}")
                logging.info(f"\tcalculated H_grad, hyparam_x0_index: {self.hyparam_x0_index}")
            else:
                print(f"\tdivergent h_grad or hit 30s limit. resort to finite differencing..., hyparam_x0_index: {self.hyparam_x0_index}")
                logging.info(f"\tdivergent h_grad or hit 30s limit. resort to finite differencing..., hyparam_x0_index: {self.hyparam_x0_index}")
                    
                def numerical_difference(x):
                    x_nd = np.array(x[:-2*self.D]).reshape(2,self.D,self.D)
                    h_nd, H_nd, flag = return_h_and_H(self.kernel, self.kernel_integral, x_nd, h_grid, self.E, self.h_dt, self.T, self.h_tol)
                    
                    ll = return_nll_without_gradients(
                        self.kernel,
                        self.kernel_integral,
                        x,
                        gammas,
                        pointwise_nu,
                        pointwise_nu_integral,
                        self.history, # dataset: interval-censored in the ic-dims
                        self.E, # the mbp dims
                        h_nd, 
                        H_nd,
                        h_grid,
                        self.p_dt,
                        self.h_dt,
                        self.T,
                        self.ll_weights_map,
                        self.nu_regularization)
                    
                    return ll
                ll = numerical_difference(plist)
                a=perf_counter()
                self.grad = approx_fprime(plist, numerical_difference, 1e-5)
                b=perf_counter()
                print(f"\tfinite differencing took {b-a}s. calculated ll: {ll}. (SR = {get_spectral_radius_from_flat_params(plist, self.D)}), hyparam_x0_index: {self.hyparam_x0_index}")
                print(f"\tobj and grad eval took {perf_counter()-start}s.")
                logging.info(f"\tfinite differencing took {b-a}s. calculated ll: {ll}. (SR = {get_spectral_radius_from_flat_params(plist, self.D)}), hyparam_x0_index: {self.hyparam_x0_index}")
                logging.info(f"\tobj and grad eval took {perf_counter()-start}s., hyparam_x0_index: {self.hyparam_x0_index}")
                logging.info(f"\tCurrent pt: {str(plist)}., hyparam_x0_index: {self.hyparam_x0_index}")
                return ll
        else: # divergent
            print(f"\tdivergent h. return nan..., hyparam_x0_index: {self.hyparam_x0_index}")
            logging.info(f"\tdivergent h. return nan..., hyparam_x0_index: {self.hyparam_x0_index}. plist: {plist}")
            return np.nan
        
        gammas = self.gammas
        nus = plist[-self.D:]

        print(f"\tcalculating ll..., hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"\tcalculating ll..., hyparam_x0_index: {self.hyparam_x0_index}")
        ll, grad = return_nll_with_gradients(
            self.kernel,
            self.kernel_integral,
            kp,
            gammas,
            nus,
            pointwise_nu,
            pointwise_nu_integral,
            self.history, # dataset: interval-censored in the ic-dims
            self.E, # the mbp dims
            h, 
            H,
            h_grad,
            H_grad,
            h_grid,
            self.p_dt,
            self.h_dt,
            self.T,
            self.ll_weights_map,
            self.nu_regularization)
        sr = get_spectral_radius_from_flat_params(plist, self.D, self.E)
        print(f"\tcalculated ll: {ll}. (SR = {sr}), hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"\tcalculated ll: {ll}. (SR = {sr}), hyparam_x0_index: {self.hyparam_x0_index}")

        self.grad = grad

        print(f"\tobj and grad eval took {perf_counter()-start}s., hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"\tobj and grad eval took {perf_counter()-start}s., hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"\tCurrent pt: {str(plist)}., hyparam_x0_index: {self.hyparam_x0_index}")

        return ll

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
#         print("GRADIENT",self.grad)
#         print("grad")
        return self.grad
    
    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        if len(self.E) == 3:
            return np.array([get_spectral_radius_from_flat_params(x, self.D, self.E), 0.5, 0.5])
        else:
            return np.array([get_spectral_radius_from_flat_params(x, self.D, self.E), get_constraintEcEc(x, self.D), get_constraintCross(x, self.D)])
        
    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return None
        
    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print(f"Objective value at iteration #{iter_count} is - {obj_value}. Running time: {int(perf_counter() - self.start_time)}s, hyparam_x0_index: {self.hyparam_x0_index}")
        logging.info(f"Objective value at iteration #{iter_count} is - {obj_value}. Running time: {int(perf_counter() - self.start_time)}s, hyparam_x0_index: {self.hyparam_x0_index}")
        

def run_optimization_one_sequence_given_starting_point(hyparam_x0, gammas, history, video_index, hyparam_x0_index, theta_ub, T, E, logfit_label="log"):   
    
    dim_weights, nu_reg, nu_x0, _ = hyparam_x0[0]
    x0 = np.hstack([hyparam_x0[1], nu_x0])
    D = len(dim_weights)
    
    print("starting", x0)
    
    time_start = perf_counter()    
    
    lb = [1e-5] * (2*D*D) + [0] * (D)
#     ub = [theta_ub] * (D*D) + [0.999] * (D*D) + list(x0[-3:] * 10)#+ [1e10] * (2*D)

    if len(E) == 3:
        ub = [theta_ub] * (D*D) + [1e10] * (D*D) + list(x0[-3:] * 10)#+ [1e10] * (2*D)
    else:
        ub = [theta_ub] * (D*D) + [0.95] + [1e10] * (D*D-1) + list(x0[-3:] * 10)#+ [1e10] * (2*D)
 
    cl = [0] * 3
    cu = [0.99] * 3

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=PMBP(history, x0, gammas, video_index, hyparam_x0_index, dim_weights, nu_reg, E, logfit_label),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 0.1)
    nlp.addOption('jacobian_approximation', 'finite-difference-values')
    nlp.addOption('hessian_approximation', 'limited-memory')    
    nlp.addOption('gamma_theta', 0.01) 
    nlp.addOption('acceptable_tol', 1e2)
    nlp.addOption('acceptable_obj_change_tol', 1e-3)
    nlp.addOption('acceptable_compl_inf_tol', 1e2)
    nlp.addOption('acceptable_iter', 5)
    nlp.addOption('print_level', 7)
    nlp.addOption('max_iter', 100)
    nlp.addOption('accept_after_max_steps', 2)

    x_opt, info = nlp.solve(x0)
    time_end = perf_counter()

#     pickle.dump([x0, x_opt, info, time_end-time_start], open(f"results_{pbs_index}_{x0_index}.p", "wb"))

    print("done", x0, time_end-time_start)
    return [x_opt, info, time_end-time_start]
