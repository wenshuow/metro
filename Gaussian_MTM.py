import math
import numpy as np

p = 500 # dimension of the random vector X
numsamples = 10 # number of samples to generate knockoffs for
rhos = [0.6]*(p-1) # the correlations
halfnumtry = 4 # m/half number of candidates
stepsize = 1.5 # step size in the unit of 1/\sqrt((\Sigma)^{-1}_{jj})

def p_marginal_trans_log(j, xj, xjm1):
    if j>p:
        return("error!")
    if j==1:
        return(-xj*xj/2-0.5*math.log(2*math.pi))
    j = j - 1
    return(-(xj-rhos[j-1]*xjm1)**2/(2*(1-rhos[j-1]**2)) - 0.5*math.log(2*math.pi*(1-rhos[j-1]**2)))

def p_marginal_log(x):
  p_dim = len(x)
  res = -x[0]**2/2-0.5*math.log(2*math.pi)
  if p_dim==1:
    return(res)
  for j in list(range(2,p_dim+1)):
    res = res + p_marginal_trans_log(j, x[j-1], x[j-2])
  return(res)

def SCEP_MH_MC(x_obs, gamma, x_grid):
  p = len(x_obs)
  midtry = math.floor(len(x_grid[0])/2)+1
  x_grid = np.delete(x_grid,midtry-1,1)
  indices = [0]*p
  indicators = [0]*p
  tildexs = []
  ws = np.zeros([2*midtry-2,p])
  refws = np.zeros([2*midtry-2,p])
  marg_density_log = p_marginal_log(x_obs)
  parallel_marg_density_log = np.zeros([2*midtry-2,p])
  refs_parallel_marg_density_log = np.zeros([2*midtry-2,p])
  cond_density_log = 0
  parallel_cond_density_log = np.zeros([2*midtry-2,p])
  refs_parallel_cond_density_log = np.zeros([2*midtry-2,p])
  # first do j=1
  ws[:,0] = [x+x_obs[0] for x in x_grid[0]]
  for k in range(2*midtry-2):
      parallel_marg_density_log[k,0] = marg_density_log + p_marginal_trans_log(1,ws[k,0],0) + p_marginal_trans_log(2,x_obs[1],ws[k,0]) - p_marginal_trans_log(1,x_obs[0],0) - p_marginal_trans_log(2,x_obs[1],x_obs[0])
  probs = [0]*(2*midtry-2)
  for k in range(2*midtry-2):
    probs[k] = parallel_marg_density_log[k,0] + parallel_cond_density_log[k,0]
  oriprobs = list(probs)
  probs = [x-max(probs) for x in probs]
  probs = [math.exp(x) for x in probs]
  probs = [x/sum(probs) for x in probs]
  indices[0] = np.random.multinomial(1,probs).tolist().index(1)
  selectionprob = probs[indices[0]]
  refws[:,0] = [x+ws[indices[0],0] for x in x_grid[0]]
  for k in range(2*midtry-2):
      refs_parallel_marg_density_log[k,0] = marg_density_log + p_marginal_trans_log(1,refws[k,0],0) + p_marginal_trans_log(2,x_obs[1],refws[k,0]) - p_marginal_trans_log(1,x_obs[0],0) - p_marginal_trans_log(2,x_obs[1],x_obs[0])
  refprobs = [0]*(2*midtry-2)
  for k in range(2*midtry-2):
    refprobs[k] = refs_parallel_marg_density_log[k,0] + refs_parallel_cond_density_log[k,0]
  remove = max(refprobs + oriprobs)
  refprobs = [x-remove for x in refprobs]
  oriprobs = [x-remove for x in oriprobs]
  acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
  acc_ratio_log = math.log(acc_rg)
  if math.log(np.random.uniform())<=acc_ratio_log+math.log(gamma):
    tildexs.append(ws[indices[0],0])
    indicators[0] = 1
    cond_density_log = cond_density_log + math.log(selectionprob) + math.log(gamma) + min(0,acc_ratio_log)
  else:
    tildexs.append(x_obs[0])
    indicators[0] = 0
    cond_density_log = cond_density_log + math.log(selectionprob) + math.log(1-gamma*min(1,math.exp(acc_ratio_log)))
  for j in range(1,p):
    ws[:,j] = [x+x_obs[j] for x in x_grid[j]] # Wj,k
    for k in range(2*midtry-2):
      # calculating marginal densities if started from Wj,k in stead of Xj
      if j+1==p:
        parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(p,ws[k,j],x_obs[j-1]) - p_marginal_trans_log(p,x_obs[j],x_obs[j-1])
      if j+1>1 and j+1<p:
        parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(j+1,ws[k,j],x_obs[j-1]) + p_marginal_trans_log(j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
    for k in range(2*midtry-2):
      # if started from Wj,k instead of Xj, conditional densities
      probs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # probs[k2] is the conditional density of X1; X2; ...; Wj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        probs[k2] = parallel_cond_density_log[k2,j-1] + parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(j+1,ws[k,j],ws[k2,j-1]) - p_marginal_trans_log(j+1,x_obs[j],ws[k2,j-1])
        if j+2<=p:
          probs[k2] = probs[k2] + p_marginal_trans_log(j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
      oriprobs = probs
      probs = [x-max(probs) for x in probs]
      probs = [math.exp(x) for x in probs]
      probs = [x/sum(probs) for x in probs]
      selectionprobk = probs[indices[j-1]]
      refprobs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # refprobs[k2] is the conditional density of X1; X2; ...; Wrj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        refprobs[k2] = refs_parallel_cond_density_log[k2,j-1] + refs_parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(j+1,ws[k,j],refws[k2,j-1]) - p_marginal_trans_log(j+1,x_obs[j],refws[k2,j-1])
        if j+2<=p:
          refprobs[k2] = refprobs[k2] + p_marginal_trans_log(j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
      remove = max(refprobs+oriprobs)
      refprobs = [x-remove for x in refprobs]
      oriprobs = [x-remove for x in oriprobs]
      acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
      j_acc_ratio_log = math.log(acc_rg)
      if indicators[j-1]==1:
        parallel_cond_density_log[k,j] = parallel_cond_density_log[k,j] + math.log(selectionprobk) + math.log(gamma) + min(0,j_acc_ratio_log)
      if indicators[j-1]==0:
        parallel_cond_density_log[k,j] = parallel_cond_density_log[k,j] + math.log(selectionprobk) + math.log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
    if j+2<=p:
      for ii in range(j+1,p):
        for k in range(2*midtry-2):
          parallel_cond_density_log[k,ii] = cond_density_log
          refs_parallel_cond_density_log[k,ii] = cond_density_log
    for k in range(2*midtry-2):
      # X1; ...; Wjk; ...Xp; tX1; ...; tXj-1
      probs[k] = parallel_marg_density_log[k,j] + parallel_cond_density_log[k,j]
    originalprobs = probs
    probs = [x-max(probs) for x in probs]
    probs = [math.exp(x) for x in probs]
    probs = [x/sum(probs) for x in probs]
    indices[j] = np.random.multinomial(1,probs).tolist().index(1) # indices[j] is the indices of Wjk selected from all the Wjk's
    selectionprob = probs[indices[j]]
    refws[:,j] = [x+ws[indices[j],j] for x in x_grid[j]]
    for k in range(2*midtry-2):
      # density of X1; ...; Wrj,k; ...; Xp
      if j+1==p:
        refs_parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(p,refws[k,j],x_obs[j-1]) - p_marginal_trans_log(p,x_obs[j],x_obs[j-1])
      if j+1>1 and j+1<p:
        refs_parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(j+1,refws[k,j],x_obs[j-1]) + p_marginal_trans_log(j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
    for k in range(2*midtry-2):
      # if started from Wrj,k instead of Xj
      probs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # probs[k2] is X1; ...; Wj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, densities
        probs[k2] = parallel_cond_density_log[k2,j-1] + parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(j+1,refws[k,j],ws[k2,j-1]) - p_marginal_trans_log(j+1,x_obs[j],ws[k2,j-1])
        if j+2<=p:
          probs[k2] = probs[k2] + p_marginal_trans_log(j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
      oriprobs = probs
      probs = [x-max(probs) for x in probs]
      probs = [math.exp(x) for x in probs]
      probs = [x/sum(probs) for x in probs]
      selectionprobk = probs[indices[j-1]]
      refprobs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # refprobs[k2] is X1; ...; Wrj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, conditional densities
        refprobs[k2] = refs_parallel_cond_density_log[k2,j-1] + refs_parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(j+1,refws[k,j],refws[k2,j-1]) - p_marginal_trans_log(j+1,x_obs[j],refws[k2,j-1])
        if j+2<=p:
          refprobs[k2] = refprobs[k2] + p_marginal_trans_log(j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
      remove = max(refprobs+oriprobs)
      refprobs = [x-remove for x in refprobs]
      oriprobs = [x-remove for x in oriprobs]
      acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
      j_acc_ratio_log = math.log(acc_rg)
      if indicators[j-1]==1:
        refs_parallel_cond_density_log[k,j] = refs_parallel_cond_density_log[k,j] + math.log(selectionprobk) + math.log(gamma) + min(0,j_acc_ratio_log)
      if indicators[j-1]==0:
        refs_parallel_cond_density_log[k,j] = refs_parallel_cond_density_log[k,j] + math.log(selectionprobk) + math.log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
    if j+2<=p:
      for ii in range(j+1,p):
        for k in range(2*midtry-2):
          parallel_cond_density_log[k,ii] = cond_density_log
          refs_parallel_cond_density_log[k,ii] = cond_density_log
    refprobs = [0]*(2*midtry-2)
    for k in range(2*midtry-2):
      # X1; ...; Wrjk; ...Xp; tX1; ...; tXj-1
      refprobs[k] = refs_parallel_marg_density_log[k,j] + refs_parallel_cond_density_log[k,j]
    remove = max(originalprobs+refprobs)
    refprobs = [x-remove for x in refprobs]
    originalprobs = [x-remove for x in originalprobs]
    acc_rg = min(1,sum([math.exp(x) for x in originalprobs])/sum([math.exp(x) for x in refprobs]))
    acc_ratio_log = math.log(acc_rg)
    if math.log(np.random.uniform())<=acc_ratio_log+math.log(gamma):
      tildexs.append(ws[indices[j],j])
      indicators[j] = 1
      cond_density_log = cond_density_log + math.log(selectionprob) + math.log(gamma) + min(0,acc_ratio_log)
    else:
      tildexs.append(x_obs[j])
      indicators[j] = 0
      cond_density_log = cond_density_log + math.log(selectionprob) + math.log(1-gamma*min(1,math.exp(acc_ratio_log)))
  return(tildexs)

p = 500
bigmatrix = np.zeros([numsamples,2*p])
quantile_x = np.zeros([p,2*halfnumtry+1])
sds = [0]*p
sds[0] = math.sqrt(1-rhos[0]**2)
for i in range(1,p-1):
  sds[i] = math.sqrt((1-rhos[i-1]**2)*(1-rhos[i]**2)/(1-rhos[i-1]**2*rhos[i]**2))
sds[p-1] = math.sqrt(1-rhos[p-2]**2)
for i in range(p):
  quantile_x[i] = [x*sds[i]*stepsize for x in list(range(-halfnumtry,halfnumtry+1))]
# quantile.x is the candidate set plus the current state, with step size in the unit of 1/\sqrt((\Sigma)^{-1}_{jj})
for i in range(numsamples):
    bigmatrix[i,0] = np.random.normal()
    for j in range(1,p):
        bigmatrix[i,j] = math.sqrt(1-rhos[j-1]**2)*np.random.normal() + rhos[j-1]*bigmatrix[i,j-1]
    bigmatrix[i,p:(2*p)] = SCEP_MH_MC(bigmatrix[i,0:p],0.999,quantile_x)
# bigmatrixresult is an nx2p matrix, each row being an indpendent sample of (X, \tilde X).