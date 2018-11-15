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
# in the end, the above two functions should be supplied as arguments of the below function




SCEP.MH.MC = function(x.obs, alpha, x.grid){
  # this is the main function that samples knockoffs
  # x.obs is the real covariates
  # alpha is as in Section 3.2.2
  # x.grid is the grid we want to use for MTM sampling; x.grid[j,] is the relative grid we use for Xj, e.g., [-2,1,0,1,2]
  p = length(x.obs) # p is the length of x.obs and number of covariates
  midtry = floor(dim(x.grid)[2]/2)+1 # this is the index of the center of x.grid, i.e., the position of the value 0; midtry-1=m
  x.grid = x.grid[,-midtry] # this is to remove the value 0, as it is not one of the candidates
  indices = rep(0,p) # this vector to record with candidate is selected for each j from 1 to p; initialized as a 0 vector
  indicators = indices # this vector to record acceptance or rejection for each j from 1 to p; initialized as a 0 vector
  tildexs = c() # this vector to record the knockoffs, the final output of the algorithm
  ws = matrix(0,ncol=p,nrow=2*midtry-2) # this is all the candidate proposals, with total number p*(2m), stored in px2m matrix, intialized to be a 0 matrix
  # adding to the above line, for each Xj, we will have 2m candidates, and they are ws[j,1] to ws[j,2m]; m+1=midtry
  refws = ws # this is the reference proposals, used to ensure reversibility
  # adding to the above line, for each Xj and selected Yj from the 2m candidates, we will have 2m references, and they are refws[j,1] to refws[j,2m]
  # adding to the above line, note that refws will depend on the actual choice of Yj, which is not known now
  marg.density.log = p.marginal.log(x.obs) # the log density of the vector x.obs
  parallel.marg.density.log = matrix(0,ncol=p,nrow=2*midtry-2) # the jk'th entry of the matrix is the marginal log density of the modified x.obs, with Xj changed to ws[j,k]
  refs.parallel.marg.density.log = parallel.marg.density.log # the jk'th entry of the matrix is the marginal log density of the modified x.obs, with Xj changed to refws[j,k]
  cond.density.log = 0 # this is the log of the conditional density of the entire knockoff sampling history (this includes selecting candidates, accepting and rejecting), conditional on the obeserved x being x.obs
  # right now we don't have anything other than the observed x, so this is 0
  parallel.cond.density.log = parallel.marg.density.log # the jk'th entry is the log conditional density similar to above, but conditional on the observed x being x.obs with Xj changed to ws[j,k]
  refs.parallel.cond.density.log = parallel.cond.density.log # the jk'th entry is the log conditional density similar to above, but conditional on the observed x being x.obs with Xj changed to refws[j,k]
  ws[,1] = x.grid[1,] + x.obs[1] # ws[,1] is the set of candidates for X1, which is obtained by adding x.obs[1] to the relative grid x.grid[1,]
  for(k in 1:(2*midtry-2)){
      # calculating the marginal log density for (ws[k,1] x.obs[2] x.obs[3] ... x.obs[p]), by simply removing and adding terms to the log density of (x.obs)
      parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,ws[k,1],0) + p.marginal.trans.log(2,x.obs[2],ws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  probs = rep(0,(2*midtry-2)) # this is meant to store the likelihoods at the 2m candidates, i.e., ws[,1], which will be used to select one
  for(k in 1:(2*midtry-2)){
    # probs[k] is now set to be the log density of (ws[k,1] x.obs[2] x.obs[3] ... x.obs[p]); the part after + is zero, but is kept here because in the future it should be included, so we I want to keep a same format
    probs[k] = parallel.marg.density.log[k,1] + parallel.cond.density.log[k,1]
  }
  oriprobs = probs # oriprobs is a backup copy of probs
  probs = probs-max(probs) # to get the selection probabilities, we need to exponentiate and normarlize; to avoid overflow, first remove the max
  probs = exp(probs) # exponentiate
  probs = probs/sum(probs) # normarlize
  indices[1] = which(rmultinom(1,1,probs)==1) # select a candidate using the now normarlized probabilities
  selectionprob = probs[indices[1]] # record the probability of selecting it in selectionprob; this will translates to a term in the log conditional density in the future
  refws[,1] = x.grid[1,] + ws[indices[1],1] # now we have a proposal for X1, we can get the reference set by adding the relative grid to that proposal, which is ws[indices[1],1] (see two lines above)
  for(k in 1:(2*midtry-2)){
      # calculating the marginal log density for (refws[k,1] x.obs[2] x.obs[3] ... x.obs[p]), by simply removing and adding terms to the log density of (x.obs)
      refs.parallel.marg.density.log[k,1] = marg.density.log + p.marginal.trans.log(1,refws[k,1],0) + p.marginal.trans.log(2,x.obs[2],refws[k,1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
  }
  # to calculate the acceptance ratio, we need the likelihoods at the reference candidates; let's do that now
  refprobs = rep(0,(2*midtry-2)) # initialze
  for(k in 1:(2*midtry-2)){
    # refprobs[k] is now set to be the log density of (refws[k,1] x.obs[2] x.obs[3] ... x.obs[p]); the part after + is zero, but is kept here because in the future it should be included, so we I want to keep a same format
    refprobs[k] = refs.parallel.marg.density.log[k,1] + refs.parallel.cond.density.log[k,1]
  }
  # recall that we had a backup copy as oriprobs; we need to exponentiate and then sum oriprobs and refprobs, take their ratio, which gives acceptance ratio
  remove = max(c(refprobs,oriprobs)) # to avoid overflow, first find the maximum
  refprobs = refprobs-remove # remove the maximum
  oriprobs = oriprobs-remove # remove the maximum
  acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs))) # here is the acceptance ratio
  acc.ratio.log = log(acc.rg) # the log acceptance ratio
  if(log(runif(1))<=acc.ratio.log+log(alpha)){ # we accept if Unif(0,1)<=alpha*acceptance raio; here is the log transformed version; usually alpha is 1 but here we can't (see paper)
    tildexs = c(tildexs,ws[indices[1],1]) # in case of acceptance, append ws[indices[1],1] to tilexs; this is the first term in tildexs
    indicators[1] = 1 # update the indicator, representing we accept for X1 & tilde X1
    cond.density.log = cond.density.log + log(selectionprob) + log(alpha) + min(0,acc.ratio.log) # update the conditional density
    # adding the log density of (selecting that proposal from the candidates and accepting it)
  } else{ # this is a rejection
    tildexs = c(tildexs,x.obs[1]) # in case of rejection, append x.obs[1] to tilexs
    indicators[1] = 0 # update the indicator, representing we reject for X1
    cond.density.log = cond.density.log + log(selectionprob) + log(1-alpha*min(1,exp(acc.ratio.log))) # update the conditional density
    # adding the log density of (selecting that proposal from the candidates and reject); you can see had we chosen alpha=1, we might get a -Inf here
  }
  for(j in 2:p){
    # now we deal with Xj, j from 2 to p
    ws[,j] = x.grid[j,] + x.obs[j] # by adding x.obs[j] to the relative grid, we get the candidates for Xj
    for(k in 1:(2*midtry-2)){
      # calculating marginal log densities of (x.obs[1:j-1] ws[k,j] x,obs[j+1:p]), by adding and removing terms from log density of (x.obs)
      if(j==p){
        parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(p,ws[k,p],x.obs[p-1]) - p.marginal.trans.log(p,x.obs[p],x.obs[p-1])
      }
      if(j>1&&j<p){
        parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(j,ws[k,j],x.obs[j-1]) + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j,x.obs[j],x.obs[j-1]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
    }
    for(k in 1:(2*midtry-2)){
      # calculating the conditional log density of the knockoff sampling history given the observed X is a modified x.obs with Xj changed to ws[k,j]
      # in other words, log of p(knockoff sampling history|observe (x.obs[1:j-1] ws[k,j] x.obs[j+1:p]))
      # note that a change of Xj does not affect the knockoff sampling until we sample knockoff for Xj-1
      # so the only new thing we need to calculate is how likely is it p(step j-1 of history|observe (x.obs[1:j-1] ws[k,j] x.obs[j+1:p]))
      # you will see in this iteration, the jth observation is always ws[k,j] in place of x.obs[j]
      probs = rep(0,(2*midtry-2)) # we need to know the selection probability at step j-1, which has changed
      for(k2 in 1:(2*midtry-2)){
        # probs[k2] is the log density of (x.obs[1:j-2] ws[k2,j-1] ws[k,j] x.obs[j+1:p]) plus log of p(history until step j-2|observe (x.obs[1:j-2] ws[k2,j-1] ws[k,j] x.obs[j+1:p]))
        # the first term written above can be obtained easily from log density of (x.obs[1:j-2] ws[k2,j-1] x.obs[j] x.obs[j+1:p])
        # the second term is the same as log of p(history until step j-2|observe (x.obs[1:j-2] ws[k2,j-1] x.obs[j] x.obs[j+1:p])), which is parallel.cond.density.log[k2,j-1]
        # these should explain the below operations
        probs[k2] = parallel.cond.density.log[k2,j-1] + parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],ws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],ws[k2,j-1])
        if(j+1<=p)
          probs[k2] = probs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      oriprobs = probs # again, take a back up for probs
      probs = probs-max(probs) # rescale
      probs = exp(probs) # exponentiate
      probs = probs/sum(probs) # normarlize
      selectionprobk = probs[indices[j-1]] # probability of selecting the same candidate for step j-1 (which is the indices[j-1]th one)
      refprobs = rep(0,(2*midtry-2)) # now we need the likelihoods at reference candidates to find the acceptance ratio
      for(k2 in 1:(2*midtry-2)){
        # this might be a little confusing, but I am restarting a new loop, so the k2 is not the same as above
        # refprobs[k2] is the log density of (x.obs[1:j-2] refws[k2,j-1] ws[k,j] x.obs[j+1:p]) plus log of p(history until step j-2|observe (x.obs[1:j-2] refws[k2,j-1] ws[k,j] x.obs[j+1:p]))
        # the logic is exactly the same as the previous loop; just add "ref" to everything
        refprobs[k2] = refs.parallel.cond.density.log[k2,j-1] + refs.parallel.marg.density.log[k2,j-1] + p.marginal.trans.log(j,ws[k,j],refws[k2,j-1]) - p.marginal.trans.log(j,x.obs[j],refws[k2,j-1])
        if(j+1<=p)
          refprobs[k2] = refprobs[k2] + p.marginal.trans.log(j+1,x.obs[j+1],ws[k,j]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
      remove = max(c(refprobs,oriprobs)) # prepare for rescaling
      refprobs = refprobs-remove # rescale refprobs
      oriprobs = oriprobs-remove # rescale oriprobs
      acc.rg = min(1,sum(exp(oriprobs))/sum(exp(refprobs))) # acceptance ratio
      j.acc.ratio.log = log(acc.rg) # log of the acceptance raio; the prefix "j" indicates it's not the same as the one we get when we start from x.obs
      if(indicators[j-1]==1){
        # we have accepted at step j-1, so the conditional density should be updated accordingly
        # this means adding the log of selection probability and log of acceptance probability
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(alpha) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        # we have rejected at step j-1, so the conditional density should be updated accordingly
        # this means adding the log of selection probability and log of rejection probability
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-alpha*min(1,exp(j.acc.ratio.log)))
      }
      # now parallel.cond.density.log[k,j] is log of p(history till step j-1|observe (x.obs[1:j-1] ws[k,j] x.obs[j+1:p]))
    }
    if(j+1<=p){
      for(ii in (j+1):p){
        # obviously, p(history till step j-1|observe (x.obs[1:l] ws[k,l] x.obs[l+1:p])) for l>=j+1 is the same as p(history till step j-1|observe (x.obs[1:l] x.obs[l] x.obs[l+1:p]))
        # this is due to Markov property; thus update them to be the same as cond.density.log
        for(k in 1:(2*midtry-2)){
          parallel.cond.density.log[k,ii] = cond.density.log
          refs.parallel.cond.density.log[k,ii] = cond.density.log
        }
      }
    }
    # now let's calculate the likelihoods when x.obs[j] is changed to ws[k,j]
    for(k in 1:(2*midtry-2)){
      # just log of marginal density of (x.obs[1:j-1] ws[k,j] x.obs[j+1:p]) + log of p(history till step j-1|observe (x.obs[1:j-1] ws[k,j] x.obs[j+1:p]))
      # we have both terms by now; the second one has just been calculated
      probs[k] = parallel.marg.density.log[k,j] + parallel.cond.density.log[k,j]
    }
    originalprobs = probs # a backup for probs
    probs = probs-max(probs) # rescale
    probs = exp(probs) # exponentiate
    probs = probs/sum(probs) # normarlize
    indices[j] = which(rmultinom(1,1,probs)==1) # indices[j] is the index of the selected candidate (see line 75)
    selectionprob = probs[indices[j]] # recored the probability of selecting it
    # starting from the below line, it's almost identical as the part starting from line 107, with ws changed to refws; I will thus make only key comments
    refws[,j] = x.grid[j,] + ws[indices[j],j] # add the selected candidate to the relative grid, to get the reference candidates
    for(k in 1:(2*midtry-2)){
      # calculating marginal log densities of (x.obs[1:j-1] refws[k,j] x,obs[j+1:p]), by adding and removing terms from log density of (x.obs)
      if(j==p){
        refs.parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(p,refws[k,p],x.obs[p-1]) - p.marginal.trans.log(p,x.obs[p],x.obs[p-1])
      }
      if(j>1&&j<p){
        refs.parallel.marg.density.log[k,j] = marg.density.log + p.marginal.trans.log(j,refws[k,j],x.obs[j-1]) + p.marginal.trans.log(j+1,x.obs[j+1],refws[k,j]) - p.marginal.trans.log(j,x.obs[j],x.obs[j-1]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
      }
    }
    for(k in 1:(2*midtry-2)){
      # calculating the conditional log density of the knockoff sampling history given the observed X is a modified x.obs with Xj changed to refws[k,j]
      # you will see in this iteration, the jth observation is always refws[k,j] in place of x.obs[j]
      probs = rep(0,(2*midtry-2))
      for(k2 in 1:(2*midtry-2)){
        # probs[k2] is the log density of (x.obs[1:j-2] ws[k2,j-1] refws[k,j] x.obs[j+1:p]) plus log of p(history until step j-2|observe (x.obs[1:j-2] ws[k2,j-1] refws[k,j] x.obs[j+1:p]))
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
        # refprobs[k2] is the log density of (x.obs[1:j-2] refws[k2,j-1] refws[k,j] x.obs[j+1:p]) plus log of p(history until step j-2|observe (x.obs[1:j-2] refws[k2,j-1] refws[k,j] x.obs[j+1:p]))
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
      # now refs.parallel.cond.density.log[k,j] is log of p(history till step j-1|observe (x.obs[1:j-1] refws[k,j] x.obs[j+1:p]))
    }
    if(j+1<=p){
      for(ii in (j+1):p){
        for(k in 1:(2*midtry-2)){
          parallel.cond.density.log[k,ii] = cond.density.log
          refs.parallel.cond.density.log[k,ii] = cond.density.log
        }
      }
    }
    # line 232 to line 239 seems useless, since I already did it from line 164 to line 173, and cond.density.log hasn't changed in between...
    # if that's true, we will delete them
    # now we have everything ready to actually make decisions for the jth step
    # picking up from line 185, we now need to get the sum of the likelihoods at the reference candidates
    refprobs = rep(0,(2*midtry-2))
    for(k in 1:(2*midtry-2)){
      # just log of marginal density of (x.obs[1:j-1] refws[k,j] x.obs[j+1:p]) + log of p(history till step j-1|observe (x.obs[1:j-1] refws[k,j] x.obs[j+1:p]))
      refprobs[k] = refs.parallel.marg.density.log[k,j] + refs.parallel.cond.density.log[k,j]
    }
    remove = max(c(originalprobs,refprobs)) # recall the backup we made in line 180; get the maximum
    refprobs = refprobs-remove # rescale
    originalprobs = originalprobs-remove # rescale
    acc.rg = min(1,sum(exp(originalprobs))/sum(exp(refprobs))) # acceptance ratio
    acc.ratio.log = log(acc.rg) # log of acceptance ratio
    if(log(runif(1))<=acc.ratio.log+log(alpha)){
      # in case of acceptance
      tildexs = c(tildexs,ws[indices[j],j]) # append ws[indices[j],j] to tildexs
      indicators[j] = 1 # update indicators[j] to mean we accept at step j
      cond.density.log = cond.density.log + log(selectionprob) + log(alpha) + min(0,acc.ratio.log) # conditional log density reflects we select and accept
    } else{
      # in case of rejection
      tildexs = c(tildexs,x.obs[j]) # append x.obs[j] to tildexs
      indicators[j] = 0 # update indicators[j] to mean we reject at step j
      cond.density.log = cond.density.log + log(selectionprob) + log(1-alpha*min(1,exp(acc.ratio.log))) # conditional log density reflects we select and reject
    }
  }
  return(tildexs) # at the end, report the tildexs
}



bigmatrix = matrix(0,ncol=2*p,nrow=numsamples)
for(i in 1:numsamples){
  # each row is a independent sample
  bigmatrix[i,1] = rnorm(1,0,1) # bigmatrix[i,1] is normal
  for(j in 2:p)
    bigmatrix[i,j] =rnorm(1,0,sqrt(1-rhos[j-1]^2)) + rhos[j-1]*bigmatrix[i,j-1] # afterwards it's AR(1)
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
  bigmatrix[i,(p+1):(2*p)] = SCEP.MH.MC(bigmatrix[i,1:p],0.999,quantile.x) # using alpha=0.999
}
cor.measure = c()
for(i in 1:p){
  cor.measure = c(cor.measure,abs(cor(bigmatrix[,i],bigmatrix[,i+p])))
}
print(mean(cor.measure))

