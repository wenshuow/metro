import math
import numpy as np


def log(x):
    if x==0:
        return(float("-inf"))
    return(math.log(x))

def Gibbs_sampler(k1, k2, K, beta, N = 100):
    # Gibbs sampler with N iterations
    current = np.zeros([k1,k2])
    for i in range(N):
        for j in range(k1):
            for k in range(k2):
                probs = np.zeros(K)
                for state in range(K):
                    if j>0:
                        probs[state] = probs[state] - beta*(state-current[j-1,k])**2
                    if k>0:
                        probs[state] = probs[state] - beta*(state-current[j,k-1])**2
                    if j<k1-1:
                        probs[state] = probs[state] - beta*(state-current[j+1,k])**2
                    if k<k2-1:
                        probs[state] = probs[state] - beta*(state-current[j,k+1])**2
                probs = [x-max(probs) for x in probs]
                probs = [math.exp(x) for x in probs]
                probs = [x/sum(probs) for x in probs]
                current[j,k] = np.random.multinomial(1,probs).tolist().index(1) # conditional distribution is multinomial
    return(current)
    
def int_to_vec(int, k2, num_try, stepsize):
    # convert an integer to a configuration
    out = [0]*k2
    base = 2*num_try+1
    for i in range(k2):
        out[i] = int%base
        int = (int-out[i])//base
    for i in range(k2):
        out[i] = (out[i] - num_try)*stepsize
    return(out)
    
def vec_to_int(vec, num_try, stepsize):
    # convert a configuration to an integer
    out = 0
    base = 2*num_try+1
    for i in vec[::-1]:
        i = i//stepsize + num_try
        out = base*out + i
    return(out)
            
def p_marginal_change_log(x_ori, i, j, new, beta, k1, k2, K, base_prob_log = None):
    # change of log density when new is added to x_ori[i,j]
    if x_ori[i,j]+new>=K or x_ori[i,j]+new<0:
        return(log(0))
    res = 0
    if i>0:
        res = res - beta*(x_ori[i,j]+new-x_ori[i-1,j])**2 + beta*(x_ori[i,j]-x_ori[i-1,j])**2
    if j>0:
        res = res - beta*(x_ori[i,j]+new-x_ori[i,j-1])**2 + beta*(x_ori[i,j]-x_ori[i,j-1])**2
    if i<k1-1:
        res = res - beta*(x_ori[i,j]+new-x_ori[i+1,j])**2 + beta*(x_ori[i,j]-x_ori[i+1,j])**2
    if j<k2-1:
        res = res - beta*(x_ori[i,j]+new-x_ori[i,j+1])**2 + beta*(x_ori[i,j]-x_ori[i,j+1])**2
    if base_prob_log is not None:
        res = base_prob_log[i, j, x_ori[i,j]+new] - base_prob_log[i, j, x_ori[i,j]]
    return(res)

def SCEP_MH_Gibbs(k1, k2, K, beta, x_obs, gamma, half_num_try, step_size, base_prob_log = None):
    # half_num_try is the m in the paper
    # step_size is t
    # assumes k1 >= k2
    [k1, k2] = x_obs.shape
    num_try = 2*half_num_try
    grid = [x*step_size for x in list(range(-half_num_try,half_num_try+1))]
    grid = np.delete(grid,half_num_try) # [-mt, -(m-1)t, ..., -t, t, ..., (m-1)t, mt]
    num_config = (2*num_try+1)**k2
    indices = np.zeros([k1,k2]).astype(int) # which one is selected/proposed for x_{i,j}
    indicators = np.zeros([k1,k2]).astype(int) # acceptance or rejection
    tildexs = np.zeros([k1,k2]).astype(int) # relative knockoff (knockoff-original)
    parallel_cond_density_log = np.zeros([k1,k2,num_config]) 
    # P(X*_{1:(i,j)-1},Xk_{1:(i,j)-1}|X_{i,j},X_{i,j+1}...X_{i+1,j-1},others being observed x) # (i,j)-1 means the node before (i,j) when traversing
    for i in range(k1):
        for j in range(k2):
            probs = [0]*len(grid)
            for w in range(len(grid)):
                # calculate the density of the entire history had we started with x_obs[i,j] changed to x_obs[i,j]+grid[w]
                parallel_decendent = [0]*k2
                parallel_decendent[j] = grid[w]
                probs[w] = parallel_cond_density_log[i,j,vec_to_int(parallel_decendent,num_try,step_size)]
                probs[w] = probs[w] + p_marginal_change_log(x_obs,i,j,grid[w],beta, k1,k2, K, base_prob_log)
            oriprobs = probs.copy()
            probs = [x-max(probs) for x in probs]
            probs = [math.exp(x) for x in probs]
            probs = [x/sum(probs) for x in probs]
            indices[i,j] = np.random.multinomial(1,probs).tolist().index(1)
            proposal = grid[indices[i,j]] # proposal x*_{i,j}
            refprobs = [0]*len(grid)
            for w in range(len(grid)):
                # calculate the density of the entire history had we started with x_obs[i,j] changed to x_obs[i,j]+proposal+grid[w]
                ref_decendent = [0]*k2
                ref_decendent[j] = grid[w] + proposal
                refprobs[w] = parallel_cond_density_log[i,j,vec_to_int(ref_decendent,num_try,step_size)]
                refprobs[w] = refprobs[w] + p_marginal_change_log(x_obs,i,j,grid[w]+proposal,beta,k1,k2,K, base_prob_log)
            remove = max(refprobs + oriprobs)
            refprobs = [x-remove for x in refprobs]
            oriprobs = [x-remove for x in oriprobs]
            acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
            acc_ratio_log = log(acc_rg)
            if log(np.random.uniform())<=acc_ratio_log+log(gamma):
                tildexs[i,j] = proposal
                indicators[i,j] = 1
            else:
                tildexs[i,j] = 0
                indicators[i,j] = 0
            if i==k1-1 and j==k2-1:
                # if at the end, return the knockoffs; note that tildexs is the relative knockoff
                return(tildexs+x_obs)
            for dec_int in range(num_config):
                # do the same process as above, with all possible (X_{i,j+1},X_{i,j+2}...X_{i+1,j})
                decendent = int_to_vec(dec_int,k2,num_try,step_size) # decendent[j] is the activated variable at the jth column (ith or (i+1)st row)
                inf_flag = False
                if j==k2-1:
                    for k in range(k2):
                        if decendent[k] + x_obs[i+1,k]<0 or decendent[k] + x_obs[i+1,k]>=K:
                            inf_flag = True
                            break
                else:
                    if i<k1-1:
                        for k in range(j+1):
                            if decendent[k] + x_obs[i+1,k]<0 or decendent[k] + x_obs[i+1,k]>=K:
                                inf_flag = True
                                break
                    if inf_flag==False:
                        for k in range(j+1,k2):
                            if decendent[k] + x_obs[i,k]<0 or decendent[k] + x_obs[i,k]>=K:
                                inf_flag = True
                                break
                if inf_flag:
                    # if any X is outside of the support, record a -inf log density and move on
                    if j<k2-1:
                        parallel_cond_density_log[i,j+1,dec_int] = log(0)
                    else:
                        parallel_cond_density_log[i+1,0,dec_int] = log(0)
                    continue
                probs = [0]*len(grid)
                for w in range(len(grid)):
                    if x_obs[i,j]+grid[w]<0 or x_obs[i,j]+grid[w]>=K:
                        probs[w] = log(0)
                        continue
                    prev_decendent = decendent.copy()
                    prev_decendent[j] = grid[w]
                    # account for marginal density change; needs modification for slicing
                    if base_prob_log is not None:
                        probs[w] = probs[w] + base_prob_log[i,j,grid[w]+x_obs[i,j]] - base_prob_log[i,j,x_obs[i,j]]
                    if i>0:
                        probs[w] = probs[w]-beta*(grid[w]+x_obs[i,j]-x_obs[i-1,j])**2 + beta*(x_obs[i,j]-x_obs[i-1,j])**2
                    if j>0:
                        probs[w] = probs[w]-beta*(grid[w]+x_obs[i,j]-x_obs[i,j-1])**2 + beta*(x_obs[i,j]-x_obs[i,j-1])**2
                    if i<k1-1:
                        probs[w] = probs[w]-beta*(grid[w]+x_obs[i,j]-decendent[j]-x_obs[i+1,j])**2 + beta*(x_obs[i,j]-decendent[j]-x_obs[i+1,j])**2
                    if j<k2-1:
                        probs[w] = probs[w]-beta*(grid[w]+x_obs[i,j]-decendent[j+1]-x_obs[i,j+1])**2 + beta*(x_obs[i,j]-decendent[j+1]-x_obs[i,j+1])**2
                    probs[w] = probs[w] + parallel_cond_density_log[i,j,vec_to_int(prev_decendent,num_try,step_size)] # account for conditional density change
                oriprobs = probs.copy()
                probs = [x-max(probs) for x in probs]
                probs = [math.exp(x) for x in probs]
                probs = [x/sum(probs) for x in probs]
                selectionprob = probs[indices[i,j]]
                proposal = grid[indices[i,j]]
                refprobs = [0]*len(grid)
                for w in range(len(grid)):
                    if x_obs[i,j]+grid[w]+proposal<0 or x_obs[i,j]+grid[w]+proposal>=K:
                        refprobs[w] = log(0)
                        continue
                    ref_decendent = decendent.copy()
                    ref_decendent[j] = grid[w] + proposal
                    # account for marginal density change; needs modification for slicing
                    if base_prob_log is not None:
                        refprobs[w] =  refprobs[w] + base_prob_log[i, j, proposal + x_obs[i,j]] - base_prob_log[i,j, x_obs[i,j]]
                    if i>0:
                        refprobs[w] = refprobs[w]-beta*(grid[w]+proposal+x_obs[i,j]-x_obs[i-1,j])**2 + beta*(x_obs[i,j]-x_obs[i-1,j])**2
                    if j>0:
                        refprobs[w] = refprobs[w]-beta*(grid[w]+proposal+x_obs[i,j]-x_obs[i,j-1])**2 + beta*(x_obs[i,j]-x_obs[i,j-1])**2
                    if i<k1-1:
                        refprobs[w] = refprobs[w]-beta*(grid[w]+proposal+x_obs[i,j]-decendent[j]-x_obs[i+1,j])**2 + beta*(x_obs[i,j]-decendent[j]-x_obs[i+1,j])**2
                    if j<k2-1:
                        refprobs[w] = refprobs[w]-beta*(grid[w]+proposal+x_obs[i,j]-decendent[j+1]-x_obs[i,j+1])**2 + beta*(x_obs[i,j]-decendent[j+1]-x_obs[i,j+1])**2
                    refprobs[w] = refprobs[w] + parallel_cond_density_log[i,j,vec_to_int(ref_decendent,num_try,step_size)]
                remove = max(refprobs + oriprobs)
                refprobs = [x-remove for x in refprobs]
                oriprobs = [x-remove for x in oriprobs]
                acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
                acc_ratio_log = log(acc_rg)
                # now update P(X*_{1:(i,j)},Xk_{1:(i,j)}|X_{i,j+1},X_{i,j+2}...X_{i+1,j},others being observed x)
                if j<k2-1:
                    prev_decendent = decendent.copy()
                    prev_decendent[j] = 0
                    new_log = parallel_cond_density_log[i,j,vec_to_int(prev_decendent,num_try,step_size)] + log(selectionprob)
                    if indicators[i,j]==1:
                        new_log = new_log + log(gamma) + min(0,acc_ratio_log)
                    if indicators[i,j]==0:
                        new_log = new_log + log(1-gamma*min(1,math.exp(acc_ratio_log)))
                    parallel_cond_density_log[i,j+1,dec_int] = new_log
                else:
                    prev_decendent = decendent.copy()
                    prev_decendent[j] = 0
                    new_log = parallel_cond_density_log[i,j,vec_to_int(prev_decendent,num_try,step_size)] + log(selectionprob)
                    if indicators[i,j]==1:
                        new_log = new_log + log(gamma) + min(0,acc_ratio_log)
                    if indicators[i,j]==0:
                        new_log = new_log + log(1-gamma*min(1,math.exp(acc_ratio_log)))
                    parallel_cond_density_log[i+1,0,dec_int] = new_log

def sim(k1, k2, K, beta, numsamples, m, t, N_gibbs):

    bigarray = np.zeros([k1,k2,2,numsamples])
    bigmatrix = np.zeros([numsamples,2*k1*k2])
    for i in range(numsamples):
        bigarray[:,:,0,i] = Gibbs_sampler(k1, k2, K, beta, N_gibbs)
        bigarray[:,:,1,i] = SCEP_MH_Gibbs(k1, k2, K, beta, bigarray[:,:,0,i], 0.9, m, t) # I tried m=t=1
        bigmatrix[i,0:(k1*k2)] = np.reshape(bigarray[:,:,0,i],k1*k2)
        bigmatrix[i,(k1*k2):(2*k1*k2)] = np.reshape(bigarray[:,:,1,i],k1*k2)

    return(bigmatrix)

# checking correlation matrix and covariance matrix
# k1 = 3
# k2 = 2
# K = 4 # support {0,1,...,K-1}
# beta = 0.2
# numsamples = 100
# bigmatrix = sim(k1, k2, K, beta, numsamples, 1, 1)

#print(np.corrcoef(np.transpose(bigmatrix)))
#print(np.cov(np.transpose(bigmatrix)))
