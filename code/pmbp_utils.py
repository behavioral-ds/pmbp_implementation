import time
import numpy as np
from scipy.linalg import eigvals
from scipy.stats import percentileofscore

def get_constraintEE(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(D,D)    
    return np.max(np.abs(eigvals(k[:2,:2])))

def get_constraintEcEc(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(D,D)    
    return k[-1,-1]

def get_constraintCross(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(D,D)    
    EE = k[:2,:2]
    EcEc = k[-1:,-1:]
    EEc = k[:2,-1:]
    EcE = k[-1:,:2]
    return np.matmul(np.matmul(EcE, np.linalg.inv(np.identity(2) - EE)), EEc)[0][0]

def get_spectral_radius_from_flat_params(plist, D, E):
    if len(E) == 3:
        return np.max(np.abs(eigvals(plist[D*D:(D*D)*2].reshape(D,D))))
    else:
        return np.max(np.abs(eigvals(plist[[9,10,12,13]].reshape(2,2))))

def get_starting_points(D, E, theta_ub, num=6):
    lspoints = []
    
    j = 0
    while True:
        np.random.seed(j)
        theta = np.array([theta_ub*np.random.random() for i in range(D*D)])
        kappa = np.array([np.random.random() for i in range(D*D)])
        nu = np.array([np.random.random() for i in range(D)])
        
        if np.max(np.abs(eigvals(kappa.reshape(D,D)))) < 0.95 and np.max(np.abs(eigvals(kappa.reshape(D,D)[:len(E),:len(E)].reshape(len(E),len(E))))) < 0.95:
            kappa = kappa/2
            lspoints.append(np.hstack([theta, kappa, nu]))        
        if len(lspoints) == num:
            break
        j+=1
        
    return lspoints

def return_ic_data_given_ytid(x, video_dat_file, start_day, end_day):
    views, tweets, shares = video_dat_file.loc[x].values
    views_p, shares_p, tweets_p = [], [], []
    for i in range(start_day+1,end_day+1):
        views_p.append([views[i-1], [i-1, i]])
        if tweets[i-1] < 0:
            tweets_p.append([0, [i-1, i]])
        else:
            tweets_p.append([tweets[i-1], [i-1, i]])
        shares_p.append([shares[i-1], [i-1, i]])
    return [views_p, shares_p, tweets_p]

def get_percentile_error(val1, val2, ref_vals):
    return np.abs(percentileofscore(ref_vals, val1) - percentileofscore(ref_vals, val2))

def get_rmse_error(val1, val2):
    return np.sqrt(np.mean((val2 - val1)**2))

def get_smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def get_Ec_volumes(history, d):
    partition = [x[1] for x in history[0]] # assumes that first dimension is E dimension
    
    collector = []
    for start, end in partition:
        count = np.sum((np.array(history[d]) >= start) & (np.array(history[d]) < end))
        collector.append([count, [start, end]])
    return collector