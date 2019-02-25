import math
import numpy as np
from scipy.stats import t
from scipy.stats import multivariate_normal
from cvxopt import matrix, solvers

def log(x):
    if x==0:
        return(float("-inf"))
    return(math.log(x))

def dnormlog(x):
    return(-x*x/2-0.5*log(2*math.pi))

def dexplog(x):
    return(-x)

def p_marginal_trans_log(df_t, rhos, p, j, xj, xjm1):
    if j>p:
        return("error!")
    if j==1:
        return(math.log(t.pdf(xj*math.sqrt(df_t/(df_t/2)), df=df_t))+0.5*math.log(df_t)-0.5*math.log(df_t-2))
    j = j - 1
    return(math.log(t.pdf((xj-rhos[j-1]*xjm1)*math.sqrt(df_t/(df_t-2))/math.sqrt(1-rhos[j-1]**2),df=df_t))+0.5*(math.log(df_t)-math.log(df_t-2)-math.log(1-rhos[j-1]**2)))

def p_marginal_log(df_t, rhos, p, x):
  p_dim = len(x)
  res = p_marginal_trans_log(df_t, rhos, p, 1, x[0], 0)
  if p_dim==1:
    return(res)
  for j in list(range(2,p_dim+1)):
    res = res + p_marginal_trans_log(df_t, rhos, p, j, x[j-1], x[j-2])
  return(res)

def SCEP_MH_MC(x_obs, gamma, x_grid, rhos, df_t):
  p = len(x_obs)
  midtry = math.floor(len(x_grid[0])/2)+1
  x_grid = np.delete(x_grid,midtry-1,1)
  indices = [0]*p
  indicators = [0]*p
  tildexs = []
  ws = np.zeros([2*midtry-2,p])
  refws = np.zeros([2*midtry-2,p])
  marg_density_log = p_marginal_log(df_t, rhos, p, x_obs)
  parallel_marg_density_log = np.zeros([2*midtry-2,p])
  refs_parallel_marg_density_log = np.zeros([2*midtry-2,p])
  cond_density_log = 0
  parallel_cond_density_log = np.zeros([2*midtry-2,p])
  refs_parallel_cond_density_log = np.zeros([2*midtry-2,p])
  # first do j=1
  ws[:,0] = [x+x_obs[0] for x in x_grid[0]]
  for k in range(2*midtry-2):
      parallel_marg_density_log[k,0] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, 1,ws[k,0],0) + p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],ws[k,0]) - p_marginal_trans_log(df_t, rhos, p, 1,x_obs[0],0) - p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],x_obs[0])
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
      refs_parallel_marg_density_log[k,0] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, 1,refws[k,0],0) + p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],refws[k,0]) - p_marginal_trans_log(df_t, rhos, p, 1,x_obs[0],0) - p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],x_obs[0])
  refprobs = [0]*(2*midtry-2)
  for k in range(2*midtry-2):
    refprobs[k] = refs_parallel_marg_density_log[k,0] + refs_parallel_cond_density_log[k,0]
  remove = max(refprobs + oriprobs)
  refprobs = [x-remove for x in refprobs]
  oriprobs = [x-remove for x in oriprobs]
  acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
  acc_ratio_log = log(acc_rg)
  if log(np.random.uniform())<=acc_ratio_log+log(gamma):
    tildexs.append(ws[indices[0],0])
    indicators[0] = 1
    cond_density_log = cond_density_log + log(selectionprob) + log(gamma) + min(0,acc_ratio_log)
  else:
    tildexs.append(x_obs[0])
    indicators[0] = 0
    cond_density_log = cond_density_log + log(selectionprob) + log(1-gamma*min(1,math.exp(acc_ratio_log)))
  for j in range(1,p):
    ws[:,j] = [x+x_obs[j] for x in x_grid[j]] # Wj,k
    for k in range(2*midtry-2):
      # calculating marginal densities if started from Wj,k in stead of Xj
      if j+1==p:
        parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, p,ws[k,j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, p,x_obs[j],x_obs[j-1])
      if j+1>1 and j+1<p:
        parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, j+1,ws[k,j],x_obs[j-1]) + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
    for k in range(2*midtry-2):
      # if started from Wj,k instead of Xj, conditional densities
      probs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # probs[k2] is the conditional density of X1; X2; ...; Wj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        probs[k2] = parallel_cond_density_log[k2,j-1] + parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(df_t, rhos, p, j+1,ws[k,j],ws[k2,j-1]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],ws[k2,j-1])
        if j+2<=p:
          probs[k2] = probs[k2] + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
      oriprobs = probs
      probs = [x-max(probs) for x in probs]
      probs = [math.exp(x) for x in probs]
      probs = [x/sum(probs) for x in probs]
      selectionprobk = probs[indices[j-1]]
      refprobs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # refprobs[k2] is the conditional density of X1; X2; ...; Wrj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        refprobs[k2] = refs_parallel_cond_density_log[k2,j-1] + refs_parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(df_t, rhos, p, j+1,ws[k,j],refws[k2,j-1]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],refws[k2,j-1])
        if j+2<=p:
          refprobs[k2] = refprobs[k2] + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
      remove = max(refprobs+oriprobs)
      refprobs = [x-remove for x in refprobs]
      oriprobs = [x-remove for x in oriprobs]
      acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
      j_acc_ratio_log = log(acc_rg)
      if indicators[j-1]==1:
        parallel_cond_density_log[k,j] = parallel_cond_density_log[k,j] + log(selectionprobk) + log(gamma) + min(0,j_acc_ratio_log)
      if indicators[j-1]==0:
        parallel_cond_density_log[k,j] = parallel_cond_density_log[k,j] + log(selectionprobk) + log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
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
        refs_parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, p,refws[k,j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, p,x_obs[j],x_obs[j-1])
      if j+1>1 and j+1<p:
        refs_parallel_marg_density_log[k,j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, j+1,refws[k,j],x_obs[j-1]) + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
    for k in range(2*midtry-2):
      # if started from Wrj,k instead of Xj
      probs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # probs[k2] is X1; ...; Wj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, densities
        probs[k2] = parallel_cond_density_log[k2,j-1] + parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(df_t, rhos, p, j+1,refws[k,j],ws[k2,j-1]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],ws[k2,j-1])
        if j+2<=p:
          probs[k2] = probs[k2] + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
      oriprobs = probs
      probs = [x-max(probs) for x in probs]
      probs = [math.exp(x) for x in probs]
      probs = [x/sum(probs) for x in probs]
      selectionprobk = probs[indices[j-1]]
      refprobs = [0]*(2*midtry-2)
      for k2 in range(2*midtry-2):
        # refprobs[k2] is X1; ...; Wrj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, conditional densities
        refprobs[k2] = refs_parallel_cond_density_log[k2,j-1] + refs_parallel_marg_density_log[k2,j-1] + p_marginal_trans_log(df_t, rhos, p, j+1,refws[k,j],refws[k2,j-1]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],refws[k2,j-1])
        if j+2<=p:
          refprobs[k2] = refprobs[k2] + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],refws[k,j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
      remove = max(refprobs+oriprobs)
      refprobs = [x-remove for x in refprobs]
      oriprobs = [x-remove for x in oriprobs]
      acc_rg = min(1,sum([math.exp(x) for x in oriprobs])/sum([math.exp(x) for x in refprobs]))
      j_acc_ratio_log = log(acc_rg)
      if indicators[j-1]==1:
        refs_parallel_cond_density_log[k,j] = refs_parallel_cond_density_log[k,j] + log(selectionprobk) + log(gamma) + min(0,j_acc_ratio_log)
      if indicators[j-1]==0:
        refs_parallel_cond_density_log[k,j] = refs_parallel_cond_density_log[k,j] + log(selectionprobk) + log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
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
    acc_ratio_log = log(acc_rg)
    if log(np.random.uniform())<=acc_ratio_log+log(gamma):
      tildexs.append(ws[indices[j],j])
      indicators[j] = 1
      cond_density_log = cond_density_log + log(selectionprob) + log(gamma) + min(0,acc_ratio_log)
    else:
      tildexs.append(x_obs[j])
      indicators[j] = 0
      cond_density_log = cond_density_log + log(selectionprob) + log(1-gamma*min(1,math.exp(acc_ratio_log)))
  return(tildexs)



# find SDP
def get_cov_guided(p, rhos):
  cormatrix = np.ones([p,p])
  for i in range(p-1):
      for j in range(i+1,p):
          for k in range(i,j):
              cormatrix[i,j] = cormatrix[i,j]*rhos[k]
          cormatrix[j,i] = cormatrix[i,j]

  G = cormatrix
  Cl1 = np.zeros([1,p*p])
  # Al1 = -np.diag(np.ones([p]))
  Cl2 = np.reshape(np.diag(np.ones([p])),[1,p*p])
  # Al2 = np.diag(np.ones([p]))
  d_As = np.reshape(np.diag(np.ones([p])),[p*p])
  As = np.diag(d_As)
  As = As[np.where(np.sum(As,axis=1)>0),:][0,:,:]
  Al1 = -As.copy()
  Al2 = As.copy()
  Cs = np.reshape(2*G,[1,p*p])
  A = np.concatenate([Al1,Al2,As],axis=1)
  C = np.transpose(np.concatenate([Cl1,Cl2,Cs],axis=1))
  K = {}
  K['s'] = [p,p,p]
  # K['l'] = 2*p
  b = np.ones([p,1])
  A = np.asmatrix(A)
  c = np.asmatrix(C)
  b = np.asmatrix(b)
  result = dsdp(A, b, c, K)
  d_sdp = result['y'] # use SDP for diag(s); other options include ASDP, equicorrelated, etc.

  # starting to calculate the parameters for covariance-guided proposals
  Cov_matrix = np.matrix.copy(cormatrix)
  matrix_diag = d_sdp
  Cov_matrix_off = np.matrix.copy(Cov_matrix)
  for i in range(p):
      Cov_matrix_off[i,i] = Cov_matrix[i,i] - matrix_diag[i]
  correlations = [0]*(p-1)
  for i in range(p-1):
    correlations[i] = Cov_matrix[i,i+1]/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i+1,i+1])
  inverse_all = np.zeros([2*p-1,2*p-1])
  inverse_all[0,0] = (1/(1-correlations[0]**2))/Cov_matrix[0,0]
  inverse_all[0,1] = (-correlations[0]/(1-correlations[0]**2))/math.sqrt(Cov_matrix[0,0]*Cov_matrix[1,1])
  inverse_all[p-1,p-1] = (1/(1-correlations[p-2]**2))/Cov_matrix[p-1,p-1]
  inverse_all[p-1,p-2] = (-correlations[p-2]/(1-correlations[p-2]**2))/math.sqrt(Cov_matrix[p-1,p-1]*Cov_matrix[p-2,p-2])
  if p>=3:
      for i in range(1,p-1):
          inverse_all[i,i-1] = (-correlations[i-1]/(1-correlations[i-1]**2))/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i-1,i-1])
          inverse_all[i,i] = ((1-correlations[i-1]**2*correlations[i]**2)/((1-correlations[i-1]**2)*(1-correlations[i]**2)))/Cov_matrix[i,i]
          inverse_all[i,i+1] = (-correlations[i]/(1-correlations[i]**2))/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i+1,i+1])
      
  temp_mat = Cov_matrix_off @ inverse_all[0:p,0:p]
  prop_mat = temp_mat
  upper_matrix = np.concatenate((Cov_matrix, Cov_matrix_off), axis = 1)
  lower_matrix = np.concatenate((Cov_matrix_off, Cov_matrix), axis = 1)
  whole_matrix = np.concatenate((upper_matrix, lower_matrix), axis = 0)
  cond_means_coeff = []
  cond_vars = [0]*p
  temp_means_coeff = np.reshape(whole_matrix[p,0:p],[1,p]) @ inverse_all[0:p,0:p]
  cond_means_coeff.append(temp_means_coeff)
  cond_vars[0] = (Cov_matrix[0,0] - cond_means_coeff[0] @ np.reshape(whole_matrix[p,0:p],[p,1]))[0,0]
  for il in range(1,p):
    temp_var = Cov_matrix[il-1]
    temp_id = np.zeros([p+il-1,p+il-1])
    temp_row = np.zeros([p+il-1,p+il-1])
    temp_id[il-1,il-1] = 1
    temp_row[il-1,:] = matrix_diag[il-1] * inverse_all[il-1,0:(p+il-1)]
    temp_col = np.matrix.copy(np.transpose(temp_row))
    temp_fourth = matrix_diag[il-1]**2 * np.reshape(inverse_all[il-1,0:(p+il-1)],[p+il-1,1]) @ np.reshape(inverse_all[il-1,0:(p+il-1)],[1,p+il-1])
    temp_numerator = temp_id - temp_row - temp_col + temp_fourth
    temp_denominator = -matrix_diag[il-1] * (2-matrix_diag[il-1]*inverse_all[il-1,il-1])
    temp_remaining = -matrix_diag[il-1]*inverse_all[il-1,0:(p+il-1)]
    temp_remaining[il-1] = 1 + temp_remaining[il-1]
    inverse_all[0:(p+il-1),0:(p+il-1)] = inverse_all[0:(p+il-1),0:(p+il-1)] - (1/temp_denominator)*temp_numerator
    inverse_all[p+il-1,p+il-1] = -1/temp_denominator
    inverse_all[p+il-1,0:(p+il-1)] = 1/temp_denominator * temp_remaining
    inverse_all[0:(p+il-1),p+il-1] = np.matrix.copy(inverse_all[p+il-1,0:(p+il-1)])
    temp_means_coeff = np.reshape(whole_matrix[p+il,0:(p+il)],[1,p+il]) @ inverse_all[0:(p+il),0:(p+il)]
    cond_means_coeff.append(temp_means_coeff)
    cond_vars[il] = (Cov_matrix[il,il] - cond_means_coeff[il] @ np.reshape(whole_matrix[p+il,0:(p+il)],[p+il,1]))[0,0]


  return [cond_means_coeff, cond_vars]


def SCEP_MH_MC_COV(x_obs, gamma, mu_vector, cond_coeff, cond_means_coeff, cond_vars, rhos):
  p = len(x_obs)
  rej = 0
  tildexs = []
  cond_mean = mu_vector + np.reshape(cond_coeff @ np.reshape(x_obs-mu_vector,[p,1]),p)
  cond_cov = Cov_matrix - cond_coeff @ Cov_matrix_off
  ws = multivariate_normal.rvs(cond_mean,cond_cov)
  def q_prop_pdf_log(num_j, vec_j, prop_j):
      num_j = num_j + 1
      if num_j!=(len(vec_j)-p+1):
          return("error")
      temp_mean = (mu_vector[num_j-1] + cond_means_coeff[num_j-1] @ np.reshape(vec_j-np.concatenate([mu_vector,mu_vector[0:(num_j-1)]]),[len(vec_j),1]))[0,0]
      return(-(prop_j-temp_mean)**2/(2*cond_vars[num_j-1])-0.5*math.log(2*math.pi*cond_vars[num_j-1]))
  parallel_chains = np.reshape(x_obs.tolist()*p,[p,p])
  for j in range(p):
    parallel_chains[j,j] = ws[j]
  marg_density_log = p_marginal_log(df_t, rhos, p, x_obs)
  parallel_marg_density_log = [0]*p
  for j in range(p):
    if j==0:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, 1,ws[0],0) + p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],ws[0]) - p_marginal_trans_log(df_t, rhos, p, 1,x_obs[0],0) - p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],x_obs[0])
    if j==p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, p,ws[p-1],x_obs[p-2]) - p_marginal_trans_log(df_t, rhos, p, p,x_obs[p-1],x_obs[p-2])
    if j>0 and j<p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, j+1,ws[j],x_obs[j-1]) + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
  cond_density_log = 0
  parallel_cond_density_log = [0]*p
  for j in range(p):
    true_vec = np.concatenate([x_obs,ws[0:j]])
    alter_vec = np.concatenate([x_obs,ws[0:j]])
    alter_vec[j] = ws[j]
    acc_ratio_log = q_prop_pdf_log(j, alter_vec, x_obs[j]) + parallel_marg_density_log[j] + parallel_cond_density_log[j] - (marg_density_log + cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]))
    if math.log(np.random.uniform())<=acc_ratio_log + math.log(gamma):
        tildexs.append(ws[j])
        cond_density_log = cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]) + log(gamma) + min(0,acc_ratio_log)
        if j+2<=p:
            true_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
            alter_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
            alter_vec_j[j] = ws[j]
            j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(df_t, rhos, p, j+2,ws[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j])
            if j+3<=p:
                j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],x_obs[j+1])
            j_acc_ratio_log = j_acc_ratio_log - (parallel_cond_density_log[j+1] + parallel_marg_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]))
            parallel_cond_density_log[j+1] = parallel_cond_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]) + log(gamma) + min(0,j_acc_ratio_log)
        if j+3<=p:
            for ii in range(j+2,p):
                parallel_cond_density_log[ii] = cond_density_log
    else:
      rej = rej + 1
      tildexs.append(x_obs[j])
      cond_density_log = cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]) + log(1-gamma*min(1,math.exp(acc_ratio_log)))
      if j+2<=p:
          true_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
          alter_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
          alter_vec_j[j] = ws[j]
          j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(df_t, rhos, p, j+2,ws[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j])
          if j+3<=p:
              j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],x_obs[j+1])
          j_acc_ratio_log = j_acc_ratio_log - (parallel_cond_density_log[j+1] + parallel_marg_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]))
          parallel_cond_density_log[j+1] = parallel_cond_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]) + log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
      if j+3<=p:
          for ii in range(j+2,p):
              parallel_cond_density_log[ii] = cond_density_log
  tildexs.append(rej)
  return(tildexs)


#generates the AR covariance matrix from a vector of rhos
def ar_cov_matrix(rhos):
    p = np.shape(rhos)[0] + 1
    cormatrix = np.ones([p,p])
    for i in range(p-1):
        for j in range(i+1,p):
            for k in range(i,j):
                cormatrix[i,j] = cormatrix[i,j]*rhos[k]
            cormatrix[j,i] = cormatrix[i,j]

    return cormatrix

# solves the SDP
def sdp_solver(cormatrix):
    p = np.shape(cormatrix)[0]
    c = -np.ones(p)
    
    G1 = np.zeros((p, p*p))
    for i in range(p):
        G1[i, i*p + i] = 1.0

    c = matrix(c)
    Gs = [matrix(G1.T)] + [matrix(G1.T)]
    hs = [matrix(2 * cormatrix)] + [matrix(np.identity(p))]
    G0 = matrix(-(np.identity(p)))
    h0 = matrix(np.zeros(p))

    sol = solvers.sdp(c, G0, h0, Gs, hs)
    return np.array(sol['x'])


# find the parameters of the proposal dist
def compute_proposals(rhos):
    p = np.shape(rhos)[0] + 1
    cormatrix = ar_cov_matrix(rhos)
    s_sdp = sdp_solver(cormatrix)
    print(np.mean(1-s_sdp))
    matrix_diag = np.reshape(s_sdp,p)
    Cov_matrix = np.matrix.copy(cormatrix)
    Cov_matrix_off = np.matrix.copy(Cov_matrix)
    for i in range(p):
        Cov_matrix_off[i,i] = Cov_matrix[i,i] - matrix_diag[i]
    correlations = [0]*(p-1)
    for i in range(p-1):
      correlations[i] = Cov_matrix[i,i+1]/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i+1,i+1])
    inverse_all = np.zeros([2*p-1,2*p-1])
    inverse_all[0,0] = (1/(1-correlations[0]**2))/Cov_matrix[0,0]
    inverse_all[0,1] = (-correlations[0]/(1-correlations[0]**2))/math.sqrt(Cov_matrix[0,0]*Cov_matrix[1,1])
    inverse_all[p-1,p-1] = (1/(1-correlations[p-2]**2))/Cov_matrix[p-1,p-1]
    inverse_all[p-1,p-2] = (-correlations[p-2]/(1-correlations[p-2]**2))/math.sqrt(Cov_matrix[p-1,p-1]*Cov_matrix[p-2,p-2])
    if p>=3:
        for i in range(1,p-1):
            inverse_all[i,i-1] = (-correlations[i-1]/(1-correlations[i-1]**2))/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i-1,i-1])
            inverse_all[i,i] = ((1-correlations[i-1]**2*correlations[i]**2)/((1-correlations[i-1]**2)*(1-correlations[i]**2)))/Cov_matrix[i,i]
            inverse_all[i,i+1] = (-correlations[i]/(1-correlations[i]**2))/math.sqrt(Cov_matrix[i,i]*Cov_matrix[i+1,i+1])
        
    temp_mat = Cov_matrix_off @ inverse_all[0:p,0:p]
    prop_mat = temp_mat
    upper_matrix = np.concatenate((Cov_matrix, Cov_matrix_off), axis = 1)
    lower_matrix = np.concatenate((Cov_matrix_off, Cov_matrix), axis = 1)
    whole_matrix = np.concatenate((upper_matrix, lower_matrix), axis = 0)
    cond_means_coeff = []
    cond_vars = [0]*p
    temp_means_coeff = np.reshape(whole_matrix[p,0:p],[1,p]) @ inverse_all[0:p,0:p]
    cond_means_coeff.append(temp_means_coeff)
    cond_vars[0] = (Cov_matrix[0,0] - cond_means_coeff[0] @ np.reshape(whole_matrix[p,0:p],[p,1]))[0,0]
    for il in range(1,p):
        temp_var = Cov_matrix[il-1]
        temp_id = np.zeros([p+il-1,p+il-1])
        temp_row = np.zeros([p+il-1,p+il-1])
        temp_id[il-1,il-1] = 1
        temp_row[il-1,:] = matrix_diag[il-1] * inverse_all[il-1,0:(p+il-1)]
        temp_col = np.matrix.copy(np.transpose(temp_row))
        temp_fourth = matrix_diag[il-1]**2 * np.reshape(inverse_all[il-1,0:(p+il-1)],[p+il-1,1]) @ np.reshape(inverse_all[il-1,0:(p+il-1)],[1,p+il-1])
        temp_numerator = temp_id - temp_row - temp_col + temp_fourth
        temp_denominator = -matrix_diag[il-1] * (2-matrix_diag[il-1]*inverse_all[il-1,il-1])
        temp_remaining = -matrix_diag[il-1]*inverse_all[il-1,0:(p+il-1)]
        temp_remaining[il-1] = 1 + temp_remaining[il-1]
        inverse_all[0:(p+il-1),0:(p+il-1)] = inverse_all[0:(p+il-1),0:(p+il-1)] - (1/temp_denominator)*temp_numerator
        inverse_all[p+il-1,p+il-1] = -1/temp_denominator
        inverse_all[p+il-1,0:(p+il-1)] = 1/temp_denominator * temp_remaining
        inverse_all[0:(p+il-1),p+il-1] = np.matrix.copy(inverse_all[p+il-1,0:(p+il-1)])
        temp_means_coeff = np.reshape(whole_matrix[p+il,0:(p+il)],[1,p+il]) @ inverse_all[0:(p+il),0:(p+il)]
        cond_means_coeff.append(temp_means_coeff)
        cond_vars[il] = (Cov_matrix[il,il] - cond_means_coeff[il] @ np.reshape(whole_matrix[p+il,0:(p+il)],[p+il,1]))[0,0]
        cond_vars = np.clip(cond_vars, 0.000000001, 1)
    return([Cov_matrix, matrix_diag, prop_mat,cond_means_coeff,cond_vars])



def SCEP_MH_COV(x_obs, gamma, mu_vector, df_t, rhos, param_list):
  Cov_matrix, matrix_diag, cond_coeff, cond_means_coeff, cond_vars = param_list
  p = len(x_obs)
  Cov_matrix_off = np.matrix.copy(Cov_matrix)
  for i in range(p):
      Cov_matrix_off[i,i] = Cov_matrix[i,i] - matrix_diag[i]
  rej = 0
  tildexs = []
  cond_mean = mu_vector + np.reshape(cond_coeff @ np.reshape(x_obs-mu_vector,[p,1]),p)
  cond_cov = Cov_matrix - cond_coeff @ Cov_matrix_off
  ws = multivariate_normal.rvs(cond_mean,cond_cov)
  def q_prop_pdf_log(num_j, vec_j, prop_j):
      num_j = num_j + 1
      if num_j!=(len(vec_j)-p+1):
          return("error")
      temp_mean = (mu_vector[num_j-1] + cond_means_coeff[num_j-1] @ np.reshape(vec_j-np.concatenate([mu_vector,mu_vector[0:(num_j-1)]]),[len(vec_j),1]))[0,0]
      return(-(prop_j-temp_mean)**2/(2*cond_vars[num_j-1])-0.5*math.log(2*math.pi*cond_vars[num_j-1]))
  parallel_chains = np.reshape(x_obs.tolist()*p,[p,p])
  for j in range(p):
    parallel_chains[j,j] = ws[j]
  marg_density_log = p_marginal_log(df_t,rhos,p,x_obs)
  parallel_marg_density_log = [0]*p
  for j in range(p):
    if j==0:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, 1,ws[0],0) + p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],ws[0]) - p_marginal_trans_log(df_t, rhos, p, 1,x_obs[0],0) - p_marginal_trans_log(df_t, rhos, p, 2,x_obs[1],x_obs[0])
    if j==p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, p,ws[p-1],x_obs[p-2]) - p_marginal_trans_log(df_t, rhos, p, p,x_obs[p-1],x_obs[p-2])
    if j>0 and j<p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(df_t, rhos, p, j+1,ws[j],x_obs[j-1]) + p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],x_obs[j])
  cond_density_log = 0
  parallel_cond_density_log = [0]*p
  for j in range(p):
    true_vec = np.concatenate([x_obs,ws[0:j]])
    alter_vec = np.concatenate([x_obs,ws[0:j]])
    alter_vec[j] = ws[j]
    acc_ratio_log = q_prop_pdf_log(j, alter_vec, x_obs[j]) + parallel_marg_density_log[j] + parallel_cond_density_log[j] - (marg_density_log + cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]))
    if math.log(np.random.uniform())<=acc_ratio_log + math.log(gamma):
        tildexs.append(ws[j])
        cond_density_log = cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]) + log(gamma) + min(0,acc_ratio_log)
        if j+2<=p:
            true_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
            alter_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
            alter_vec_j[j] = ws[j]
            j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(df_t, rhos, p, j+2,ws[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j])
            if j+3<=p:
                j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],x_obs[j+1])
            j_acc_ratio_log = j_acc_ratio_log - (parallel_cond_density_log[j+1] + parallel_marg_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]))
            parallel_cond_density_log[j+1] = parallel_cond_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]) + log(gamma) + min(0,j_acc_ratio_log)
        if j+3<=p:
            for ii in range(j+2,p):
                parallel_cond_density_log[ii] = cond_density_log
    else:
      rej = rej + 1
      tildexs.append(x_obs[j])
      cond_density_log = cond_density_log + q_prop_pdf_log(j, true_vec, ws[j]) + log(1-gamma*min(1,math.exp(acc_ratio_log)))
      if j+2<=p:
          true_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
          alter_vec_j = np.concatenate([parallel_chains[j+1,:], ws[0:j]])
          alter_vec_j[j] = ws[j]
          j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(df_t, rhos, p, j+2,ws[j+1],ws[j]) - p_marginal_trans_log(df_t, rhos, p, j+2,x_obs[j+1],ws[j])
          if j+3<=p:
              j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(df_t, rhos, p, j+3,x_obs[j+2],x_obs[j+1])
          j_acc_ratio_log = j_acc_ratio_log - (parallel_cond_density_log[j+1] + parallel_marg_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]))
          parallel_cond_density_log[j+1] = parallel_cond_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]) + log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
      if j+3<=p:
          for ii in range(j+2,p):
              parallel_cond_density_log[ii] = cond_density_log
  #tildexs.append(rej)
  return(tildexs)
