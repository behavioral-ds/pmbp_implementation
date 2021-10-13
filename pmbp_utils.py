import time
import numpy as np
from scipy.linalg import eigvals
from scipy.stats import percentileofscore

def get_constraintEE(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(3,3)    
    return np.max(np.abs(eigvals(k[:2,:2])))

def get_constraintEcEc(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(3,3)    
    return k[-1,-1]

def get_constraintCross(plist, D):
    k = np.array(plist[D*D:(D*D)*2]).reshape(3,3)    
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

def get_starting_points(D, theta_ub, num=6):
#    NOT specradius 0.15, 0.3, ..., 0.9
#     specradius 0.1, 0.2, ..., 0.6
    lspoints = []
    
    j = 0
    while True:
#     for i in range(num):
        np.random.seed(j)
        theta = np.array([theta_ub*np.random.random() for i in range(D*D)])
#         kappa = (np.diag([0.1*(i+1)]*D) + (0.01)*np.random.random(size=(D,D)))
        kappa = np.array([np.random.random() for i in range(D*D)])
        
        if np.max(np.abs(eigvals(kappa.reshape(3,3)))) < 0.9 and np.max(np.abs(eigvals(kappa[[0,1,3,4]].reshape(2,2)))) < 0.9:
            kappa = kappa/2
            lspoints.append(np.hstack([theta, kappa]))
           
        j+=1
        
        if len(lspoints) == num:
            break
        
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