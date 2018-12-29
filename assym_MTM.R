library(knockoff)
library(foreach)
library(doParallel)
registerDoParallel(makeCluster(8)) # number of cores for parallelization

asymean = 1/sqrt(2*pi)-1/2
variance = 1.5-(1/sqrt(2*pi)-1/2)^2
p = 500 # dimension of the random vector X
numsamples = 10 # number of samples to generate knockoffs for
rhos = rep(0.6,p-1) # the correlations
halfnumtry = 4 # m/half number of candidates
stepsize = 1.5 # step size in the unit of 1/\sqrt((\Sigma)^{-1}_{jj})


p.marginal.trans.log = function(j, xj, xjm1){
  if(j==1){
    if(xj*sqrt(variance)+asymean>=0){
      return(dnorm(xj*sqrt(variance)+asymean,sd=1,log=TRUE)+0.5*log(variance)+log(2))
    } else{
      return(dexp(-(xj*sqrt(variance)+asymean),log=TRUE))+0.5*log(variance)
    }
  }
  if((xj-rhos[j-1]*xjm1)*sqrt(variance)/sqrt(1-rhos[j-1]^2)+asymean>=0){
    return(dnorm((xj-rhos[j-1]*xjm1)*sqrt(variance)/sqrt(1-rhos[j-1]^2)+asymean,sd=1,log=TRUE)+0.5*log(variance)-0.5*log(1-rhos[j-1]^2)+log(2))
  } else{
    return(dexp(-((xj-rhos[j-1]*xjm1)*sqrt(variance)/sqrt(1-rhos[j-1]^2)+asymean),log=TRUE)+0.5*log(variance)-0.5*log(1-rhos[j-1]^2))
  }
}

p.marginal.log = function(x){
  p.dim = length(x)
  res = p.marginal.trans.log(1, x[1], 0)
  if(p.dim==1)
    return(res)
  for(j in 2:length(x)){
    res = res + p.marginal.trans.log(j, x[j], x[j-1])
  }
  return(res)
}




SCEP.MH.MC = function(x.obs, gamma, x.grid){
  p = length(x.obs)
  midtry = floor(dim(x.grid)[2]/2)+1
  gridwithoutzero = x.grid[,-midtry]
  x.grid = x.grid[,-midtry]
  indices = rep(0,p)
  indicators = indices
  tildexs = c()
  ws = matrix(0,ncol=p,nrow=2*midtry-2)
  refws = ws
  marg.density.log = p.marginal.log(x.obs)
  parallel.marg.density.log = matrix(0,ncol=p,nrow=2*midtry-2)
  refs.parallel.marg.density.log = parallel.marg.density.log
  cond.density.log = 0
  parallel.cond.density.log = parallel.marg.density.log
  refs.parallel.cond.density.log = parallel.cond.density.log
  # first do j=1
  ws[,1] = x.grid[1,] + x.obs[1]
  #print(ws)
  for(k in 1:(2*midtry-2)){
    #print(marg.density.log)
    #print(p.marginal.trans.log(1,ws[k,1],0))
    parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,ws[k,1],0) + p.marginal.trans.log(2,x.obs[2],ws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  #print(parallel.marg.density.log)
  probs = rep(0,(2*midtry-2))
  for(k in 1:(2*midtry-2)){
    probs[k] = parallel.marg.density.log[k,1] + parallel.cond.density.log[k,1]
  }
  oriprobs = probs
  probs = probs-max(probs)
  probs = exp(probs)
  probs = probs/sum(probs)
  #print(probs)
  indices[1] = which(rmultinom(1,1,probs)==1)
  selectionprob = probs[indices[1]]
  refws[,1] = x.grid[1,] + ws[indices[1],1]
  for(k in 1:(2*midtry-2)){
    refs.parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,refws[k,1],0) + p.marginal.trans.log(2,x.obs[2],refws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  refprobs = rep(0,(2*midtry-2))
  #print(ws[,1])
  #print(indices[1])
  #print(refws[,1])
  for(k in 1:(2*midtry-2)){
    refprobs[k] = refs.parallel.marg.density.log[k,1] + refs.parallel.cond.density.log[k,1]
  }
  remove = max(c(refprobs,oriprobs))
  refprobs = refprobs-remove
  oriprobs = oriprobs-remove
  acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs)))
  acc.ratio.log = log(acc.rg)
  if(log(runif(1))<=acc.ratio.log+log(gamma)){
    tildexs = c(tildexs,ws[indices[1],1])
    indicators[1] = 1
    cond.density.log = cond.density.log + log(selectionprob) + log(gamma) + min(0,acc.ratio.log)
  } else{
    tildexs = c(tildexs,x.obs[1])
    indicators[1] = 0
    cond.density.log = cond.density.log + log(selectionprob) + log(1-gamma*min(1,exp(acc.ratio.log)))
  }
  #print(acc.rg)
  #print(indicators)
  #print(marg.density.log)
  #print(cond.density.log)
  #print(parallel.marg.density.log)
  #print(parallel.cond.density.log)
  #print(refs.parallel.marg.density.log)
  #print(refs.parallel.cond.density.log)
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
      if(parallel.marg.density.log[k,j]==-Inf){
        parallel.cond.density.log[k,j] = -Inf
        next
      }
      # if started from Wj,k instead of Xj, conditional densities
      probs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # probs[k2] is the conditional density of X1; X2; ...; Wj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        if(parallel.marg.density.log[k2,j-1]==-Inf){
          probs[k2]=-Inf
        } else{
          probs[k2] = parallel.cond.density.log[k2,j-1] + parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],ws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],ws[k2,j-1])
          if(j+1<=p)
            probs[k2] = probs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
        }
        #print(parallel.cond.density.log[k2,j-1])
        #print(parallel.marg.density.log[k2,j-1])
        #print(p.marginal.trans.log(j,ws[k,j],ws[k2,j-1]))
        #print(p.marginal.trans.log(j,x.obs[j],ws[k2,j-1]))
        #print(probs[k2])
      }
      #print("starting")
      #print(probs)
      oriprobs = probs
      probs = probs-max(probs)
      probs = exp(probs)
      probs = probs/sum(probs)
      selectionprobk = probs[indices[j-1]]
      refprobs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        if(refs.parallel.marg.density.log[k2,j-1]==-Inf){
          refprobs[k2] = -Inf
          next
        }
        # refprobs[k2] is the conditional density of X1; X2; ...; Wrj-1,k2; Wj,k; ...; Xp; tX1; ...; tXj-2
        refprobs[k2] = refs.parallel.cond.density.log[k2,j-1] + refs.parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],refws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],refws[k2,j-1])
        if(j+1<=p)
          refprobs[k2] = refprobs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      remove = max(c(refprobs,oriprobs))
      #print(refprobs)
      #print(oriprobs)
      refprobs = refprobs-remove
      oriprobs = oriprobs-remove
      acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs)))
      #print("accrat")
      #print(acc.rg)
      #print(indicators[j-1])
      #print(selectionprobk)
      j.acc.ratio.log = log(acc.rg)
      if(indicators[j-1]==1){
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(gamma) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-gamma*min(1,exp(j.acc.ratio.log)))
      }
      #print(parallel.cond.density.log[k,j])
    }
    if(j+1<=p){
      for(ii in (j+1):p){
        for(k in 1:(2*midtry-2)){
          #print("current k")
          #print(k)
          #print(cond.density.log)
          parallel.cond.density.log[k,ii] = cond.density.log
          refs.parallel.cond.density.log[k,ii] = cond.density.log
          #print(parallel.marg.density.log)
          #print(parallel.cond.density.log)
        }
      }
    }
    # probability of selecting Wj,k as candidate
    #probs = rep(0,(2*midtry-2))
    for(k in 1:(2*midtry-2)){
      # X1; ...; Wjk; ...Xp; tX1; ...; tXj-1
      probs[k] = parallel.marg.density.log[k,j] + parallel.cond.density.log[k,j]
    }
    #print(parallel.marg.density.log[,j])
    #print(parallel.cond.density.log[,j])
    originalprobs = probs
    #print(probs)
    #print(j)
    #print(indicators)
    probs = probs-max(probs)
    probs = exp(probs)
    probs = probs/sum(probs)
    #print(probs)
    indices[j] = which(rmultinom(1,1,probs)==1) # indices[j] is the indices of Wjk selected from all the Wjk's
    selectionprob = probs[indices[j]]
    refws[,j] = x.grid[j,] + ws[indices[j],j]
    #print(ws[,1])
    #print(indices[1])
    #print(refws[,1])
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
      if(refs.parallel.marg.density.log[k,j]==-Inf){
        refs.parallel.cond.density.log[k,j] = -Inf
        next
      }
      # if started from Wrj,k instead of Xj
      probs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        if(parallel.marg.density.log[k2,j-1]==-Inf){
          probs[k2] = -Inf
          next
        }
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
        if(refs.parallel.marg.density.log[k2,j-1]==-Inf){
          refprobs[k2] = -Inf
          next
        }
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
        refs.parallel.cond.density.log[k,j] = refs.parallel.cond.density.log[k,j] + log(selectionprobk) + log(gamma) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        refs.parallel.cond.density.log[k,j] = refs.parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-gamma*min(1,exp(j.acc.ratio.log)))
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
    #print(originalprobs)
    #print(refprobs)
    refprobs = refprobs-remove
    originalprobs = originalprobs-remove
    acc.rg = min(1,sum(exp(originalprobs))/sum(exp(refprobs)))
    acc.ratio.log = log(acc.rg)
    #print(marg.density.log)
    #print(cond.density.log)
    #print(parallel.marg.density.log)
    #print(parallel.cond.density.log)
    #print(refs.parallel.marg.density.log)
    #print(refs.parallel.cond.density.log)
    if(log(runif(1))<=acc.ratio.log+log(gamma)){
      tildexs = c(tildexs,ws[indices[j],j])
      indicators[j] = 1
      cond.density.log = cond.density.log + log(selectionprob) + log(gamma) + min(0,acc.ratio.log)
    } else{
      tildexs = c(tildexs,x.obs[j])
      indicators[j] = 0
      cond.density.log = cond.density.log + log(selectionprob) + log(1-gamma*min(1,exp(acc.ratio.log)))
    }
    #print(acc.rg)
    #print(indicators)
    #print(indices)
    #print(tildexs)
  }
  #print(indices)
  #print(ws)
  #print(refws)
  #print(marg.density.log)
  #print(cond.density.log)
  #print(parallel.marg.density.log)
  #print(parallel.cond.density.log)
  #print(refs.parallel.marg.density.log)
  #print(refs.parallel.cond.density.log)
  return(tildexs)
}

p = 500
quantile.x = matrix(0,ncol=2*halfnumtry+1,nrow=p)
sds = rep(0,p)
sds[1] = sqrt(1-rhos[1]^2)
for(i in 2:(p-1)){
  sds[i] = sqrt((1-rhos[i-1]^2)*(1-rhos[i]^2)/(1-rhos[i-1]^2*rhos[i]^2))
}
sds[p] = sqrt(1-rhos[p-1]^2)
for(i in 1:p){
  quantile.x[i,] = ((-halfnumtry):halfnumtry)*sds[i]*stepsize
}
# quantile.x is the candidate set plus the current state, with step size in the unit of 1/\sqrt((\Sigma)^{-1}_{jj})

bigmatrix = matrix(0,ncol=2*p,nrow=numsamples)
bigmatrixresult = foreach(i=1:numsamples,.combine='rbind') %dopar% {
  if(runif(1)>=0.5){
    bigmatrix[i,1] = (abs(rnorm(1))-asymean)/sqrt(variance)
  } else{
    bigmatrix[i,1] = (-rexp(1)-asymean)/sqrt(variance)
  }
  for(j in 2:p){
    if(runif(1)>=0.5){
      bigmatrix[i,j] = ((abs(rnorm(1))-asymean)/sqrt(variance))*sqrt(1-rhos[j-1]^2) + rhos[j-1]*bigmatrix[i,j-1]
    } else{
      bigmatrix[i,j] = ((-rexp(1)-asymean)/sqrt(variance))*sqrt(1-rhos[j-1]^2) + rhos[j-1]*bigmatrix[i,j-1]
    }
  }
  bigmatrix[i,(p+1):(2*p)] = SCEP.MH.MC(bigmatrix[i,1:p],0.999,quantile.x)
  bigmatrix[i,]
}
# bigmatrixresult is an nx2p matrix, each row being an indpendent sample of (X, \tilde X).