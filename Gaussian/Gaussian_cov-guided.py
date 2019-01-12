# This script provides a sanity check for covariance-guided proposals.
# When X follows multivariate normal and the true covariance matrix is known,
# we should always accept and never reject.
import math
import numpy as np
from scipy.stats import multivariate_normal
from pydsdp.dsdp5 import dsdp

p = 5 # dimension of the random vector X
numsamples = 10000 # number of samples to generate knockoffs for
rhos = [0.6]*(p-1) # the correlations

def log(x):
    if x==0:
        return(float("-inf"))
    return(math.log(x))
    
def p_marginal_trans_log(j, xj, xjm1):
    if j>p:
        return("error!")
    if j==1:
        return(-xj*xj/2-0.5*math.log(2*math.pi))
    j = j - 1
    return(-(xj-rhos[j-1]*xjm1)**2/(2*(1-rhos[j-1]**2)) - 0.5*math.log(2*math.pi*(1-rhos[j-1]**2)))

def p_marginal_log(x):
  p_dim = len(x)
  res = p_marginal_trans_log(1, x[0], 0)
  if p_dim==1:
    return(res)
  for j in list(range(2,p_dim+1)):
    res = res + p_marginal_trans_log(j, x[j-1], x[j-2])
  return(res)
  
cormatrix = np.ones([p,p])
for i in range(p-1):
    for j in range(i+1,p):
        for k in range(i,j):
            cormatrix[i,j] = cormatrix[i,j]*rhos[k]
        cormatrix[j,i] = cormatrix[i,j]

G = cormatrix
Cl1 = np.zeros([1,p*p])
Cl2 = np.reshape(np.diag(np.ones([p])),[1,p*p])
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

def SCEP_MH_MC(x_obs, gamma, mu_vector, cond_coeff, cond_means_coeff, cond_vars):
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
  marg_density_log = p_marginal_log(x_obs)
  parallel_marg_density_log = [0]*p
  for j in range(p):
    if j==0:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(1,ws[0],0) + p_marginal_trans_log(2,x_obs[1],ws[0]) - p_marginal_trans_log(1,x_obs[0],0) - p_marginal_trans_log(2,x_obs[1],x_obs[0])
    if j==p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(p,ws[p-1],x_obs[p-2]) - p_marginal_trans_log(p,x_obs[p-1],x_obs[p-2])
    if j>0 and j<p-1:
        parallel_marg_density_log[j] = marg_density_log + p_marginal_trans_log(j+1,ws[j],x_obs[j-1]) + p_marginal_trans_log(j+2,x_obs[j+1],ws[j]) - p_marginal_trans_log(j+1,x_obs[j],x_obs[j-1]) - p_marginal_trans_log(j+2,x_obs[j+1],x_obs[j])
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
            j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(j+2,ws[j+1],ws[j]) - p_marginal_trans_log(j+2,x_obs[j+1],ws[j])
            if j+3<=p:
                j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(j+3,x_obs[j+2],x_obs[j+1])
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
          j_acc_ratio_log = q_prop_pdf_log(j, alter_vec_j, x_obs[j]) + parallel_cond_density_log[j] + parallel_marg_density_log[j] + p_marginal_trans_log(j+2,ws[j+1],ws[j]) - p_marginal_trans_log(j+2,x_obs[j+1],ws[j])
          if j+3<=p:
              j_acc_ratio_log = j_acc_ratio_log + p_marginal_trans_log(j+3,x_obs[j+2],ws[j+1]) - p_marginal_trans_log(j+3,x_obs[j+2],x_obs[j+1])
          j_acc_ratio_log = j_acc_ratio_log - (parallel_cond_density_log[j+1] + parallel_marg_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]))
          parallel_cond_density_log[j+1] = parallel_cond_density_log[j+1] + q_prop_pdf_log(j, true_vec_j, ws[j]) + log(1-gamma*min(1,math.exp(j_acc_ratio_log)))
      if j+3<=p:
          for ii in range(j+2,p):
              parallel_cond_density_log[ii] = cond_density_log
  tildexs.append(rej)
  return(tildexs)

bigmatrix = np.zeros([numsamples,2*p])
rejections = 0
for i in range(numsamples):
    bigmatrix[i,0] = np.random.normal()
    for j in range(1,p):
        bigmatrix[i,j] = math.sqrt(1-rhos[j-1]**2)*np.random.normal() + rhos[j-1]*bigmatrix[i,j-1]
    knockoff_scep = SCEP_MH_MC(bigmatrix[i,0:p],1,np.zeros([p]),prop_mat,cond_means_coeff, cond_vars)
    bigmatrix[i,p:(2*p)] = knockoff_scep[0:p]
    rejections = rejections + knockoff_scep[p]
# bigmatrix is an nx2p matrix, each row being an indpendent sample of (X, \tilde X).
print("The rejection rate is "+str(rejections/(p*numsamples))+".")