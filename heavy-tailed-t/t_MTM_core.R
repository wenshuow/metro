library(knockoff)
library(foreach)
library(doParallel)

p.marginal.trans.log = function(j, xj, xjm1){
  if(j>p)
    return("error!")
  if(j==1)
    return(dt(xj*sqrt(df.t/(df.t-2)), df.t, log=TRUE)+0.5*log(df.t)-0.5*log(df.t-2))
  return(dt((xj-rhos[j-1]*xjm1)*sqrt(df.t/(df.t-2))/sqrt(1-rhos[j-1]^2), df.t, log=TRUE)+0.5*(log(df.t)-log(df.t-2)-log(1-rhos[j-1]^2)))
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

p.joint.log = function(x, tildex, w)
  return(c(FALSE,p.marginal.log(x)))

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
  if(log(runif(1))<=acc.ratio.log+log(gamma)){
    tildexs = c(tildexs,ws[indices[1],1])
    indicators[1] = 1
    cond.density.log = cond.density.log + log(selectionprob) + log(gamma) + min(0,acc.ratio.log)
  } else{
    tildexs = c(tildexs,x.obs[1])
    indicators[1] = 0
    cond.density.log = cond.density.log + log(selectionprob) + log(1-gamma*min(1,exp(acc.ratio.log)))
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
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(gamma) + min(0,j.acc.ratio.log)
      }
      if(indicators[j-1]==0){
        parallel.cond.density.log[k,j] = parallel.cond.density.log[k,j] + log(selectionprobk) + log(1-gamma*min(1,exp(j.acc.ratio.log)))
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
    refprobs = refprobs-remove
    originalprobs = originalprobs-remove
    acc.rg = min(1,sum(exp(originalprobs))/sum(exp(refprobs)))
    acc.ratio.log = log(acc.rg)
    if(log(runif(1))<=acc.ratio.log+log(gamma)){
      tildexs = c(tildexs,ws[indices[j],j])
      indicators[j] = 1
      cond.density.log = cond.density.log + log(selectionprob) + log(gamma) + min(0,acc.ratio.log)
    } else{
      tildexs = c(tildexs,x.obs[j])
      indicators[j] = 0
      cond.density.log = cond.density.log + log(selectionprob) + log(1-gamma*min(1,exp(acc.ratio.log)))
    }
  }
  return(tildexs)
}


SCEP.MH.MC.COV = function(x.obs, alpha, mu.vector, cond.coeff, cond.means.coeff, cond.vars){
  p = length(mu.vector)
  rej = 0
  tildexs = c()
  cond.mean = mu.vector + (cond.coeff %*% matrix(x.obs-mu.vector,ncol=1))[,1]
  cond.cov = Cov.matrix - cond.coeff %*% Cov.matrix.off
  ws = mvrnorm(1,cond.mean,cond.cov)
  q.prop.pdf.log = function(num.j, vec.j, prop.j){
    if(num.j!=(length(vec.j)-p+1))
      return("error")
    temp.mean = mu.vector[num.j] + cond.means.coeff[[num.j]] %*% matrix(vec.j-c(mu.vector,mu.vector[0:(num.j-1)]),ncol=1)
    return(dnorm(prop.j, temp.mean, sqrt(cond.vars[num.j]),log=TRUE))
  }
  parallel.chains = matrix(rep(x.obs,p),ncol=p,byrow = TRUE)
  for(j in 1:p){
    parallel.chains[j,j] = ws[j]
  }
  marg.density.log = p.marginal.log(x.obs)
  parallel.marg.density.log = rep(0,p)
  for(j in 1:p){
    if(j==1)
      parallel.marg.density.log[j] = marg.density.log + p.marginal.trans.log(1,ws[1],0) + p.marginal.trans.log(2,x.obs[2],ws[1]) - p.marginal.trans.log(1,x.obs[1],0) - p.marginal.trans.log(2,x.obs[2],x.obs[1])
    if(j==p)
      parallel.marg.density.log[j] = marg.density.log + p.marginal.trans.log(p,ws[p],x.obs[p-1]) - p.marginal.trans.log(p,x.obs[p],x.obs[p-1])
    if(j>1&&j<p)
      parallel.marg.density.log[j] = marg.density.log + p.marginal.trans.log(j,ws[j],x.obs[j-1]) + p.marginal.trans.log(j+1,x.obs[j+1],ws[j]) - p.marginal.trans.log(j,x.obs[j],x.obs[j-1]) - p.marginal.trans.log(j+1,x.obs[j+1],x.obs[j])
  }
  cond.density.log = 0
  parallel.cond.density.log = rep(0,p)
  for(j in 1:p){
    true.vec = c(x.obs, ws[0:(j-1)])
    alter.vec = true.vec
    alter.vec[j] = ws[j]
    acc.ratio.log = q.prop.pdf.log(j, alter.vec, x.obs[j]) + parallel.marg.density.log[j] + parallel.cond.density.log[j] - (marg.density.log + cond.density.log + q.prop.pdf.log(j, true.vec, ws[j]))
    if(log(runif(1))<=acc.ratio.log + log(alpha)){
      tildexs = c(tildexs, ws[j])
      cond.density.log = cond.density.log + q.prop.pdf.log(j, true.vec, ws[j]) + log(alpha) + min(0,acc.ratio.log) # perhaps accelaratable
      if(j+1<=p){
        true.vec.j = c(parallel.chains[j+1,], ws[0:(j-1)])
        alter.vec.j = true.vec.j
        alter.vec.j[j] = ws[j]
        j.acc.ratio.log = q.prop.pdf.log(j, alter.vec.j, x.obs[j]) + parallel.cond.density.log[j] + parallel.marg.density.log[j] + p.marginal.trans.log(j+1,ws[j+1],ws[j]) - p.marginal.trans.log(j+1,x.obs[j+1],ws[j])
        if(j+2<=p)
          j.acc.ratio.log = j.acc.ratio.log + p.marginal.trans.log(j+2,x.obs[j+2],ws[j+1]) - p.marginal.trans.log(j+2,x.obs[j+2],x.obs[j+1])
        j.acc.ratio.log = j.acc.ratio.log - (parallel.cond.density.log[j+1] + parallel.marg.density.log[j+1] + q.prop.pdf.log(j, true.vec.j, ws[j]))
        parallel.cond.density.log[j+1] = parallel.cond.density.log[j+1] + q.prop.pdf.log(j, true.vec.j, ws[j]) + log(alpha) + min(0,j.acc.ratio.log)
      }
      if(j+2<=p){
        for(ii in (j+2):p)
          parallel.cond.density.log[ii] = cond.density.log
      }
    } else{
      rej = rej + 1
      tildexs = c(tildexs, x.obs[j])
      cond.density.log = cond.density.log + q.prop.pdf.log(j, true.vec, ws[j]) + log(1-alpha*min(1,exp(acc.ratio.log)))
      if(j+1<=p){
        true.vec.j = c(parallel.chains[j+1,], ws[0:(j-1)])
        alter.vec.j = true.vec.j
        alter.vec.j[j] = ws[j]
        j.acc.ratio.log = q.prop.pdf.log(j, alter.vec.j, x.obs[j]) + parallel.cond.density.log[j] + parallel.marg.density.log[j] + p.marginal.trans.log(j+1,ws[j+1],ws[j]) - p.marginal.trans.log(j+1,x.obs[j+1],ws[j])
        if(j+2<=p)
          j.acc.ratio.log = j.acc.ratio.log + p.marginal.trans.log(j+2,x.obs[j+2],ws[j+1]) - p.marginal.trans.log(j+2,x.obs[j+2],x.obs[j+1])
        j.acc.ratio.log = j.acc.ratio.log - (parallel.cond.density.log[j+1] + parallel.marg.density.log[j+1] + q.prop.pdf.log(j, true.vec.j, ws[j]))
        parallel.cond.density.log[j+1] = parallel.cond.density.log[j+1] + q.prop.pdf.log(j, true.vec.j, ws[j]) + log(1-alpha*min(1,exp(j.acc.ratio.log)))
      }
      if(j+2<=p){
        for(ii in (j+2):p)
          parallel.cond.density.log[ii] = cond.density.log
      }
    }
  }
  return(c(tildexs,rej))
}




cormatrix = matrix(1,ncol=p,nrow=p)
for(i in 1:(p-1)){
  for(j in (i+1):p){
    cormatrix[i,j] = prod(rhos[i:(j-1)])
    cormatrix[j,i] = cormatrix[i,j]
  }
}

d.mat.emp = create.solve_sdp(cormatrix) # the vector s by SDP
# starting to calculate the parameters for covariance-guided proposals
Cov.matrix = cormatrix
matrix.diag = d.mat.emp
Cov.matrix.off = Cov.matrix
for(i in 1:p)
  Cov.matrix.off[i,i] = Cov.matrix[i,i] - matrix.diag[i]
correlations = rep(0,p-1)
for(i in 1:(p-1))
  correlations[i] = Cov.matrix[i,i+1]/sqrt(Cov.matrix[i,i]*Cov.matrix[i+1,i+1])
inverse.all = matrix(0,ncol=2*p-1,nrow=2*p-1)
inverse.all[1,1] = (1/(1-correlations[1]^2))/Cov.matrix[1,1]
inverse.all[1,2] = (-correlations[1]/(1-correlations[1]^2))/sqrt(Cov.matrix[1,1]*Cov.matrix[2,2])
inverse.all[p,p] = (1/(1-correlations[p-1]^2))/Cov.matrix[p,p]
inverse.all[p,p-1] = (-correlations[p-1]/(1-correlations[p-1]^2))/sqrt(Cov.matrix[p,p]*Cov.matrix[p-1,p-1])
if(p>=3){
  for(i in 2:(p-1)){
    inverse.all[i,i-1] = (-correlations[i-1]/(1-correlations[i-1]^2))/sqrt(Cov.matrix[i,i]*Cov.matrix[i-1,i-1])
    inverse.all[i,i] = ((1-correlations[i-1]^2*correlations[i]^2)/((1-correlations[i-1]^2)*(1-correlations[i]^2)))/Cov.matrix[i,i]
    inverse.all[i,i+1] = (-correlations[i]/(1-correlations[i]^2))/sqrt(Cov.matrix[i,i]*Cov.matrix[i+1,i+1])
  }
}
temp.mat = Cov.matrix.off %*% inverse.all[1:p,1:p]
prop.mat = temp.mat
upper.matrix = cbind(Cov.matrix, Cov.matrix.off)
lower.matrix = cbind(Cov.matrix.off, Cov.matrix)
whole.matrix = rbind(upper.matrix, lower.matrix)
cond.means.coeff = vector("list",p)
cond.vars = rep(0,p)
cond.means.coeff[[1]] = whole.matrix[p+1,1:(p+1-1),drop=FALSE] %*% inverse.all[1:p,1:p]
cond.vars[1] = Cov.matrix[1,1] - cond.means.coeff[[1]] %*% as.matrix(whole.matrix[p+1,1:(p+1-1)])
for(il in 2:p){
  temp.var = Cov.matrix[il-1]
  temp.id = matrix(0,ncol=p+il-2,nrow=p+il-2)
  temp.row = temp.id
  temp.id[il-1,il-1] = 1
  temp.row[il-1,] = matrix.diag[il-1] * inverse.all[il-1,1:(p+il-2)]
  temp.col = t(temp.row)
  temp.fourth = matrix.diag[il-1]^2 * matrix(inverse.all[il-1,1:(p+il-2)],ncol=1) %*% matrix(inverse.all[il-1,1:(p+il-2)],nrow=1)
  temp.numerator = temp.id - temp.row - temp.col + temp.fourth
  temp.denominator = -matrix.diag[il-1] * (2-matrix.diag[il-1]*inverse.all[il-1,il-1])
  temp.remaining = -matrix.diag[il-1]*inverse.all[il-1,1:(p+il-2)]
  temp.remaining[il-1] = 1 + temp.remaining[il-1]
  inverse.all[1:(p+il-2),1:(p+il-2)] = inverse.all[1:(p+il-2),1:(p+il-2)] - (1/temp.denominator)*temp.numerator
  inverse.all[p+il-1,p+il-1] = -1/temp.denominator
  inverse.all[p+il-1,1:(p+il-2)] = 1/temp.denominator * temp.remaining
  inverse.all[1:(p+il-2),p+il-1] = inverse.all[p+il-1,1:(p+il-2)]
  cond.means.coeff[[il]] = whole.matrix[p+il,1:(p+il-1),drop=FALSE] %*% inverse.all[1:(p+il-1),1:(p+il-1)]
  cond.vars[il] = Cov.matrix[il,il] - cond.means.coeff[[il]] %*% as.matrix(whole.matrix[p+il,1:(p+il-1)])
}


