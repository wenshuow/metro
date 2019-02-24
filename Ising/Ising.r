library(bayess)
library(purrr)

# Take an integer 1 to 2^k and represent as a k2-dim vector of +- 1
# Arguments:
#   int: an integer 1 to 2^k2
#   k2: the dimension of the resulting vector
int_to_binary = function(int, k2) {
  int = int - 1
  out = rep(0, k2)
  for(i in 1:k2) {
    if(int %% 2 == 1) {
      out[i] = 1  
    } else {
      out[i] = -1
    }
    
    int = int %/% 2
  }
  
  return(out)
}
# int_to_binary(1, 2)
# int_to_binary(2, 2)
# int_to_binary(3, 2)
# int_to_binary(4, 2)

# Take an vector of +=1 and return an integer in 1:2^k2.
# This is the inverse of int_to_binary()
# 
# Arguments:
#   bin_vec: a k2-dimensional vector of +- 1
#   k2: the dimension of the vector
bin_to_int = function(bin_vec, k2) {
  out = 0
  for(i in 1:length(bin_vec)) {
    if(bin_vec[i] == 1)
      out = out + 2^(i-1)
  }
  
  return(out + 1)
}
#check that this inverts the int_to_bin function
# for(int in 1:8) {
#   print(bin_to_int(int_to_binary(int, 3), 3) == int)
# }

# Check if (i,j) is within a k1 x k2 grid
check_valid = function(i, j, k1, k2) {
  return(i >= 1 && i <= k1 && j >= 1 && j <= k2)
}
# check_valid(1, 1, 3, 3)
# check_valid(0, 1, 3, 3)
# check_valid(0, 4, 3, 3)


scep_adjust = function(x, p, scep_param) {
  if(scep_param == 0) {
    return(p)
  } else if(scep_param > 0) {
    return((x == 1) * scep_param + (1 - scep_param) * p)
  } else {
    anti = 0.5
    if(p < .5 && x == 1) {
      anti = 0
    }
    if(p > .5 && x == 1) {
      anti = (2*p - 1) / p
    }
    if(p < .5 && x == -1) {
      anti = p / (1 - p)
    }
    if(p > .5 && x == -1) {
      anti = 1
    }
    
    return((1+scep_param) * p + (-scep_param) * anti)
  }
}




# Generate an ising model for X with parameter k1.
#
# Arguments:
#   X: The variables, grid of +- 1
#   k1: first dimension of grid
#   k2: second dimension of grid
#   alpha: Ising model parameter
#   seed: optional random seed for reproducibility
#   scep_param: number in (0,1) that parameterizes scep_construction
#   B: k1 x k2 matrix of field effects given as [0,1] probabilities
#
# Returns:
#   The sampled knockoff: a k1 x k2 grid of +- 1
#
# Note:
#   k1 should be larger than or equal to k2 for fastest computation.
#   Knockoffs are generated along second axis first.
ising_knockoff = function(X, k1, k2, alpha, seed = NULL, scep_param = 0, B = NULL) {
  
  if(!is.null(seed)) {
    set.seed(seed)
  }
  if(k1 < k2) {
    print("Warning: k1 should be >= k2 for best efficiency.")
  }
  
  A = array(0, c(k1, k2, 2^k2)) #Store P(Xk_ij = 1 | X_-ij, Xk_{1:ij}) as function of k2 desendents
  Xk = array(0, dim(X)) #matrix of knockoffs
  
  #loop across rows
  for(i in 1:k1) {
    #sweep across rows
    for(j in 1:k2) {
      
      # Sweep across JT descendent configs.
      # Goal of each iteration: populate A[i, j, bin_vec]: 
      for(bin_int in 1:2^k2) {
        bin_vec = int_to_binary(bin_int, k2) #configuration of D_ij (the junction-tree descendents), a vector of +- 1
                                             #bin_vec[k] is the hypothetical state of the unique JT descendent in column k
                                             #e.g. for i=1, j=1, bin_vec[1] the state of (2,1) and bin_vec[2] is the state of (1,2).
        

        #Step 1: Account for P(X_ij | X_{-ij}) terms (up to 4 neighbors)
        
        #include boundry terms or field effect
        adjust = 0
        if(!is.null(B)) {
          adjust = B[i, j]
        }
        
        # Account for P(X_ij | X_{-ij}) terms (up to 4 neighbors)
        sum_neighbors = 0
        if(check_valid(i, j-1, k1, k2)) {
          sum_neighbors = sum_neighbors + X[i, j-1]
        }
        if(check_valid(i-1, j, k1, k2)) {
          sum_neighbors = sum_neighbors + X[i-1, j]
        }
        if(check_valid(i, j+1, k1, k2)) {
          sum_neighbors = sum_neighbors + bin_vec[j+1]
        } 
        if(check_valid(i+1, j, k1, k2)) {
          sum_neighbors = sum_neighbors + bin_vec[j]
        }
        p_base = exp(alpha * sum_neighbors + adjust) / (exp(alpha * sum_neighbors + adjust) + exp(-alpha * sum_neighbors + adjust))
        p = p_base
        
        # Step 2: Account for knockoff propogation terms 
        # Scan across previous terms (i1, j1) where (i,j) is in the active node of the Junction tree 
        # when Xk_{i1, j1} was sampled.
        p_pos = 1 #probability of observed outcome when X_ij = +1
        p_neg = 1 #probability of observed outcome when X_ij = -1
        for(j1 in 1:k2) {
          if(j1 >= j) {
            i1 =  i-1
          } else {
            i1 = i
          }
          if(check_valid(i1, j1, k1, k2)) {
            #find the descendent vector for (i1, j1)
            bin_vec1 = rep(0, k2) 
            if(i1 == i - 1) {
              bin_vec1[j1:k2] = X[i1, j1:k2]
              bin_vec1[1:j] = X[i, 1:j]
              if(j+1 <= j1) {
                bin_vec1[(j+1):j1] = bin_vec[(j+1):j1]
              }
            } 
            if(i1 == i) {
              bin_vec1[(j1+1):j] = X[i, (j1+1):j] 
              if(j < k2) {
                bin_vec1[(j+1):k2] = bin_vec[(j+1):k2]
              }
              bin_vec1[1:j1] = bin_vec[1:j1]
            }
            
            #create versions of the descendent vector for X_ij = +=1
            bin_vec1_pos = bin_vec1
            bin_vec1_neg = bin_vec1
            bin_vec1_pos[j] = 1
            bin_vec1_neg[j] = -1
            
            #incorporate this correction term from the Xk_{i1,j1} knockoff sampling
            if(Xk[i1, j1] == 1) {
              p_pos = p_pos * A[i1, j1, bin_to_int(bin_vec1_pos, k2)]
              p_neg = p_neg * A[i1, j1, bin_to_int(bin_vec1_neg, k2)]
            } else {
              p_pos = p_pos * (1 - A[i1, j1, bin_to_int(bin_vec1_pos, k2)])
              p_neg = p_neg * (1 - A[i1, j1, bin_to_int(bin_vec1_neg, k2)])
            }
            
          }
        }
        p = p * p_pos / (p * p_pos + (1 - p) * p_neg)
        p_scep = scep_adjust(X[i, j], p, scep_param)

        A[i, j, bin_to_int(bin_vec, k2)] = p_scep
      }
      
      # sample Xk_ij (using the observed descendent configuration)
      desc_config = rep(0, k2)
      if(i + 1 <= k1) {
        desc_config[1:j] = X[i + 1, 1:j]
      }
      if(j < k2) {
        desc_config[(j+1):k2] = X[i, (j+1):k2]
      }
      p = A[i, j, bin_to_int(desc_config, k2)]
      if(is.na(p)) {
        print(paste0(i,j, "na probability"))
      }

      Xk[i, j] = 2*rbinom(1, 1, p) - 1
    }
  }
  
  return(Xk)
}

ising_knockoff_slice = function(X, k1, k2, alpha, seed = NULL, scep_param = 0, B = NULL, slices = NULL, max_width = NULL) {
  if(is.null(slices)) {
    if(is.null(max_width)) {max_width = 10}
    slices = get_slices(k2, max_width)
    #print(paste0("slices: ", slices))
  }
  
  Xk = X
  if(is.null(B)) {
    B = 0 * X
  }
  
  for(i in 1:(length(slices)+1)) {
    if(i == 1) {
      start = 1
      end = slices[i] - 1
    } else if (i == (length(slices) + 1)) {
      start = slices[i-1] + 1
      end = k2
    } else {
      start = slices[i-1] + 1
      end = slices[i] - 1
    }
    width = end - start + 1
    if(width <= 0 ) {
      print("Error in slicing")
      print(slices)
      print(k2)
    }
    
    #start with initial field
    #print(paste0("start: ", start, ", end: ", end))
    B_temp = B[, start:end]
    # adjust for fixed variables
    if(i != 1) {
      B_temp[, 1] = B_temp[, 1] + alpha * X[, start - 1]
    }
    if(i != (length(slices) + 1)) {
      B_temp[, width] = B_temp[, width] + X[, end + 1]
    }
    
    Xk[, start:end] = ising_knockoff(X[, start:end], k1, width, alpha, scep_param = scep_param, B = B_temp)
  }
  
  return(Xk)
}

get_slices = function(k2, max_width = 10) {
  start = 2 + rdunif(1, 1, max_width - 1)
  return(seq.int(from = start, to = k2 - 2, by = max_width + 1))
}



# Sample frFom the conditional distribution of X_{ij} | X_{-ij}
#
# Arguments:
#   X: original variable state
#   i,j: coordinate to sample
#   k1,k2: dimensions of grid
#   alpha: Ising model parameter
# 
# Returns:
#   Sampled value for X_ij | X_{-ij} (a random quantity in +- 1)
marginal_knockoff = function(i, j, X, k1, k2, alpha, verbose = FALSE) {
  p_base = .5 #can modify to include boundry terms
  
  # Account for P(X_ij | X_{-ij}) terms (up to 4 neighbors)
  sum_neighbors = 0
  if(check_valid(i, j-1, k1, k2)) {
    sum_neighbors = sum_neighbors + X[i, j-1]
  }
  if(check_valid(i-1, j, k1, k2)) {
    sum_neighbors = sum_neighbors + X[i-1, j]
  }
  if(check_valid(i, j+1, k1, k2)) {
    sum_neighbors = sum_neighbors + X[i, j+1]
  } 
  if(check_valid(i+1, j, k1, k2)) {
    sum_neighbors = sum_neighbors + X[i+1, j]
  }
  p_base = p_base * exp(alpha * sum_neighbors) / (p_base * exp(alpha * sum_neighbors) + (1 - p_base) * exp(-alpha * sum_neighbors))
  
  if(verbose == TRUE) {
    print(sum_neighbors)
  }
  
  return(rbinom(1, 1, p_base))  
}

# Sample from the conditional distribution of X_{ij} | X_{-ij}
#   for all i,j. This is *not* a valid knockoff. It is useful
#   a lower bound for the quality of a valid knockoff at each 
#   coordinate.
#
# Arguments:
#   X: original variable state
#   k1,k2: dimensions of grid
#   alpha: Ising model parameter
# 
# Returns:
#   Xk: a k1 x k2 grid of +- 1
ising_marginal_knockoff = function(X, k1, k2, alpha) {
  Xkm = X
  for(i in 1:k1) {
    for(j in 1:k2) {
      Xkm[i, j] = marginal_knockoff(i, j, X, k1, k2, alpha)
    }
  }
  
  return(Xkm)
}

# Generates an sample from an Ising model, its knockoff, and a marginal knockoff.
#
# Arguments:
#   k1,k2: dimensions of grid
#   alpha: Ising model parameter
#   niter: number of MH iterations for original Ising sampler.
#   seed: optional seed for reproducibility
# 
# Returns:
#   An 3 x k1 x k2 array (X, Xk, Xkm) where each element is +- 1
#     X: a sample from an Ising model
#     Xk: a knockoff for X
#     Xkm: a k1 x k2 grid of elementwise knockoffs (This is not a valid joint knockoff)
generate_ising_pair = function(k1, k2, alpha = 1, niter = 1000, seed = NULL, scep_param = 0, slice = FALSE, max_width = 10) {
  X = isinghm(niter, k1, k2, beta = 2 * alpha)
  X[X == 0] = -1
  if(slice) {
    Xk = ising_knockoff_slice(X, k1, k2, alpha, max_width = max_width)
  } else {
    Xk = ising_knockoff(X, k1, k2, alpha, seed, scep_param)
  }
  Xkm = ising_marginal_knockoff(X, k1, k2, alpha)
  A = array(dim =c(3, k1, k2))
  A[1,,] = X
  A[2,,] = Xk
  A[3,,] = Xkm
  return(A)
}

sdp_lb = function(S, max_size = 500) {
  s = c()
  while(ncol(S) > max_size) {
    S_prime = S[1:max_size, 1:max_size]
    s_prime = creat.solve_sdp(S_prime)
    s = c(s, s_prime)
    S = S[(n+1):ncol(S), (n+1):ncol(S)]
    n = ncol(S)
  }
  s_prime = creat.solve_sdp(S)
  s = c(s, s_prime)  
  
  mean(abs(cor(1 - s)))
}


t1 = Sys.time()
generate_ising_pair(10, 10)
t2 = Sys.time()
t2 - t1
