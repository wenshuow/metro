library(knockoff)
p = 500
numsamples = 2000
rho = 0.99
#############
rhos = runif(p-1,-rho,rho)
# Use following line for my set of rhos
# rhos = read.csv(sprintf("cors_%d_1.csv",p),header=FALSE)[,1]*rho
#############
halfnumtry = 3 # this is the number m in Section 3.2.2
stepsize = 0.4 # this is the number s in Section 3.2.2


p.marginal.trans.log = function(j, xj, xjm1){
  # this is the log of the transition density from Xj-1 to Xj; if j=1, it's just the marginal log density of X1
  if(j>p)
    return("error!")
  if(j==1)
    return(-xj^2/2-0.5*log(2*pi))
  return(-(xj-rhos[j-1]*xjm1)^2/(2*(1-rhos[j-1]^2)) - 0.5*log(2*pi*(1-rhos[j-1]^2)))
}

p.marginal.log = function(x){
  # this is the marginal density of the entire random vector, calculated sequentially using the previous function p.marginal.trans.log
  p.dim = length(x)
  res = p.marginal.trans.log(1, x[1], 0)
  if(p.dim==1)
    return(res)
  for(j in 2:length(x)){
    res = res + p.marginal.trans.log(j, x[j], x[j-1])
  }
  return(res)
}





SCEP.MH.MC = function(x.obs, alpha, x.grid){
  # this is the main function that samples knockoffs
  # x.obs is the real covariates
  # alpha is as in Section 3.2.2
  # x.grid is the grid we want to use for MTM sampling; it always contains the relative 2m candidates and 0
  p = length(x.obs) # p is the length of x.obs and number of covariates
  midtry = floor(dim(x.grid)[2]/2)+1 # this is the index of the center of x.grid, i.e., the position of the value 0
  x.grid = x.grid[,-midtry] # this is to remove the value 0, as it is not one of the candidates
  indices = rep(0,p) # this vector to record with candidate is selected for each j from 1 to p; initialized as a 0 vector
  indicators = indices # this vector to record acceptance or rejection for each j from 1 to p; initialized as a 0 vector
  tildexs = c() # this vector to record the knockoffs, the final output of the algorithm
  ws = matrix(0,ncol=p,nrow=2*midtry-2) # this is all the candidate proposals, with total number p*(2m), stored in px2m matrix, intialized to be a 0 matrix
  refws = ws # this is the reference proposals, used to ensure reversibility
  marg.density.log = p.marginal.log(x.obs)
  parallel.marg.density.log = matrix(0,ncol=p,nrow=2*midtry-2)
  refs.parallel.marg.density.log = parallel.marg.density.log
  cond.density.log = 0
  parallel.cond.density.log = parallel.marg.density.log
  refs.parallel.cond.density.log = parallel.cond.density.log
  ws[,1] = x.grid[1,] + x.obs[1]
  for(k in 1:(2*midtry-2)){
      parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,ws[k,1],0) + p.marginal.trans.log(2,x.obs[2],ws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  probs = rep(0,(2*midtry-2))
  for(k in 1:(2*midtry-2)){
    probs[k] = parallel.marg.density.log[k,1] + parallel.cond.density.log[k,1]
  }
  oriprobs = probs
  probs = probs-max(probs)
  probs = exp(probs)
  probs = probs/sum(probs)
  indices[1] = which(rmultinom(1,1,probs)==1)
  selectionprob = probs[indices[1]]
  refws[,1] = x.grid[1,] + ws[indices[1],1]
  for(k in 1:(2*midtry-2)){
      refs.parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,refws[k,1],0) + p.marginal.trans.log(2,x.obs[2],refws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  refprobs = rep(0,(2*midtry-2))
  for(k in 1:(2*midtry-2)){
    refprobs[k] = refs.parallel.marg.density.log[k,1] + refs.parallel.cond.density.log[k,1]
  }
  remove = max(c(refprobs,oriprobs))
  refprobs = refprobs-remove
  oriprobs = oriprobs-remove
  acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs)))
  acc.ratio.log = log(acc.rg)
  if(log(runif(1))<=acc.ratio.log+log(alpha)){
    tildexs = c(tildexs,ws[indices[1],1])
    indicators[1] = 1
    cond.density.log = cond.density.log + log(selectionprob) + log(alpha) + min(0,acc.ratio.log)
  } else{
    tildexs = c(tildexs,x.obs[1])
    indicators[1] = 0
    cond.density.log = cond.density.log + log(selectionprob) + log(1-alpha*min(1,exp(acc.ratio.log)))
  }
  for(j in 2:p){
    ws[,j] = x.grid[j,] + x.obs[j] # Wj,k
    for(k in 1:(2*midtry-2)){
      # calculating marginal densities if started from Wj,k in stead of Xj
      if(j==p){
        parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(p,ws[k,p],x.obs[p-1]) - p.marginal.trans.log(p,x.obs[p],x.obs[p-1])
      }
      if(j>1&&j<p){
        parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(j,ws[k,j],x.obs[j-1]) + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j,x.obs[j],x.obs[j-1]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
    }
    for(k in 1:(2*midtry-2)){
      # if started from Wj,k instead of Xj, conditional densities
      probs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # probs[k2] is the conditional density of X1; X2; ...; Wj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        probs[k2] = parallel.cond.density.log[k2,j-1] + parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],ws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],ws[k2,j-1])
        if(j+1<=p)
          probs[k2] = probs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      oriprobs = probs
      probs = probs-max(probs)
      probs = exp(probs)
      probs = probs/sum(probs)
      selectionprobk = probs[indices[j-1]]
      refprobs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # refprobs[k2] is the conditional density of X1; X2; ...; Wrj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        refprobs[k2] = refs.parallel.cond.density.log[k2,j-1] + refs.parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],refws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],refws[k2,j-1])
        if(j+1<=p)
          refprobs[k2] = refprobs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      remove = max(c(refprobs,oriprobs))
      refprobs = refprobs-remove
      oriprobs = oriprobs-remove
      acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs)))
      j.acc.ratio.log = log(acc.rg)
      if(indicators[j-1]==1){
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(alpha) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-alpha*min(1,exp(j.acc.ratio.log)))
      }
    }
    if(j+1<=p){
      for(ii in (j+1):p){
        for(k in 1:(2*midtry-2)){
          parallel.cond.density.log[k,ii] = cond.density.log
          refs.parallel.cond.density.log[k,ii] = cond.density.log
        }
      }
    }
    # probability of selecting Wj,k as candidate
    for(k in 1:(2*midtry-2)){
      # X1; ...; Wjk; ...Xp; tX1; ...; tXj-1
      probs[k] = parallel.marg.density.log[k,j] + parallel.cond.density.log[k,j]
    }
    originalprobs = probs
    probs = probs-max(probs)
    probs = exp(probs)
    probs = probs/sum(probs)
    indices[j] = which(rmultinom(1,1,probs)==1) # indices[j] is the indices of Wjk selected from all the Wjk's
    selectionprob = probs[indices[j]]
    refws[,j] = x.grid[j,] + ws[indices[j],j]
    for(k in 1:(2*midtry-2)){
      # density of X1; ...; Wrj,k; ...; Xp
      if(j==p){
        refs.parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(p,refws[k,p],x.obs[p-1]) - p.marginal.trans.log(p,x.obs[p],x.obs[p-1])
      }
      if(j>1&&j<p){
        refs.parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(j,refws[k,j],x.obs[j-1]) + p.marginal.trans.log(j+1,x.obs[j+1],refws[k,j]) - p.marginal.trans.log(j,x.obs[j],x.obs[j-1]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
    }
    for(k in 1:(2*midtry-2)){
      # if started from Wrj,k instead of Xj
      probs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # probs[k2] is X1; ...; Wj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, densities
        probs[k2] = parallel.cond.density.log[k2,j-1] + parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,refws[k,j],ws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],ws[k2,j-1])
        if(j+1<=p)
          probs[k2] = probs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],refws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      oriprobs = probs
      probs = probs-max(probs)
      probs = exp(probs)
      probs = probs/sum(probs)
      selectionprobk = probs[indices[j-1]]
      refprobs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # refprobs[k2] is X1; ...; Wrj-1,k2; Wrj,k; ...; Xp; tX1; ...; tXj-2, conditional densities
        refprobs[k2] = refs.parallel.cond.density.log[k2,j-1] + refs.parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,refws[k,j],refws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],refws[k2,j-1])
        if(j+1<=p)
          refprobs[k2] = refprobs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],refws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      remove = max(c(refprobs,oriprobs))
      refprobs = refprobs-remove
      oriprobs = oriprobs-remove
      acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs)))
      j.acc.ratio.log = log(acc.rg)
      if(indicators[j-1]==1){
        refs.parallel.cond.density.log[k,j] = refs.parallel.cond.density.log[k,j] + log(selectionprobk) + log(alpha) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        refs.parallel.cond.density.log[k,j] = refs.parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-alpha*min(1,exp(j.acc.ratio.log)))
      }
    }
    if(j+1<=p){
      for(ii in (j+1):p){
        for(k in 1:(2*midtry-2)){
          parallel.cond.density.log[k,ii] = cond.density.log
          refs.parallel.cond.density.log[k,ii] = cond.density.log
        }
      }
    }
    refprobs = rep(0,(2*midtry-2))
    for(k in 1:(2*midtry-2)){
      # X1; ...; Wrjk; ...Xp; tX1; ...; tXj-1
      refprobs[k] = refs.parallel.marg.density.log[k,j] + refs.parallel.cond.density.log[k,j]
    }
    remove = max(c(originalprobs,refprobs))
    refprobs = refprobs-remove
    originalprobs = originalprobs-remove
    acc.rg = min(1,sum(exp(originalprobs))/sum(exp(refprobs)))
    acc.ratio.log = log(acc.rg)
    if(log(runif(1))<=acc.ratio.log+log(alpha)){
      tildexs = c(tildexs,ws[indices[j],j])
      indicators[j] = 1
      cond.density.log = cond.density.log + log(selectionprob) + log(alpha) + min(0,acc.ratio.log)
    } else{
      tildexs = c(tildexs,x.obs[j])
      indicators[j] = 0
      cond.density.log = cond.density.log + log(selectionprob) + log(1-alpha*min(1,exp(acc.ratio.log)))
    }
  }
  return(tildexs)
}



bigmatrix = matrix(0,ncol=2*p,nrow=numsamples)
for(i in 1:numsamples){
  bigmatrix[i,1] = rnorm(1,0,1)
  for(j in 2:p)
    bigmatrix[i,j] =rnorm(1,0,sqrt(1-rhos[j-1]^2)) + rhos[j-1]*bigmatrix[i,j-1]
}

#############
# This is the grids, with the jth row for the relative grid for the Xj, including 0.
# So it's a px(2m+1) matrix, with 2m being the number of proposals
# For example, if we want to propose [Xj-2,Xj-1,Xj+1,Xj+2] for Xj, the jth row of quantile.x would be [-2, -1, 0, 1, 2]
quantile.x = matrix(0,ncol=2*halfnumtry+1,nrow=p)
for(i in 1:p){
  quantile.x[i,] = ((-halfnumtry):halfnumtry)*stepsize
}
#############

for(i in 1:numsamples){
  bigmatrix[i,(p+1):(2*p)] = SCEP.MH.MC(bigmatrix[i,1:p],0.999,quantile.x)
}
cor.measure = c()
for(i in 1:p){
  cor.measure = c(cor.measure,abs(cor(bigmatrix[,i],bigmatrix[,i+p])))
}
print(mean(cor.measure))

