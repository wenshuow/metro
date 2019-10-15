''' 
	The metropolized knockoff sampler for an arbitrary probability density
	and graphical structure.

	See https://arxiv.org/abs/1903.00434 for a description of the algorithm
	and proof of validity and runtime.

	Author: Stephen Bates, October 2019
'''

import numpy as np
import itertools

def gaussian_proposal(j, xj):
	''' Sample proposal by adding independent Gaussian noise.

	'''

	return xj + np.random.normal()

def single_metro(lf, x, order, active_frontier, sym_proposal = gaussian_proposal, gamma = .99):
	''' Samples a knockoff using the Metro algorithm, using an 
			arbitrary ordering of the variables.

		Args:
			lf (function that takes a 1-d numpy array) : the log probability 
				density, only needs to be specified up to an additive constant
			x (1-dim numpy array, length p) : the observed sample
			order (1-dim numpy array, length p) : ordering to sample the variables.
				Should be a vector with unique entries 0,...,p-1.
			active_fontier (list of lists) : a list of length p where
				entry j is the set of entries > j that are in V_j. This specifies
				the conditional independence structure of the distribution given by
				lf. See page 34 of the paper.
			sym_proposal (function that takes two scalars) : symmetric proposal function

		Returns:
			xk: a vector of length d, the sampled knockoff

	'''
	#reindex to sample variables in ascending order
	inv_order = order.copy()
	for i, j in enumerate(order):
		inv_order[j] = i

	def lf2(x):
		return lf(x[inv_order])

	active_frontier2 = []
	for i in range(len(order)):
	    active_frontier2 += [[inv_order[j] for j in active_frontier[i]]]

	def sym_proposal2(j, xj):
		return sym_proposal(order[j], xj)

	# call the metro function that samples variables in ascending order
	return ordered_metro(lf2, x[order], active_frontier2, sym_proposal2, gamma)[inv_order]



def ordered_metro(lf, x, active_frontier, sym_proposal = gaussian_proposal, gamma = .99):
	''' Samples a knockoff using the Metro algorithm, moving from variable 1
		to variable n.

		Args:
			lf (function that takes a 1-d numpy array) : the log probability 
				density, only needs to be specified up to an additive constant
			x (1-dim numpy array, length p) : the observed sample
			active_fontier (list of lists) : a list of length p where
				entry j is the set of entries > j that are in V_j. This specifies
				the conditional independence structure of the distribution given by
				lf. See page 34 of the paper.
			sym_proposal (function that takes two scalars) : symmetric proposal function

		Returns:
			xk: a vector of length d, the sampled knockoff

	'''

	# locate the previous terms that affected by variable j
	affected_vars = [[] for k in range(len(x))]
	for j, j2 in itertools.product(range(len(x)), range(len(x))):
		if j in active_frontier[j2]:
			affected_vars[j] += [j2]

	# store dynamic programming intermediate results
	dp_dicts = [{} for j in range(len(x))] 

	x_prop = np.zeros(len(x)) #proposals
	x_prop[:] = np.nan
	acc = np.zeros(len(x)) #pattern of acceptances

	#loop across variables
	for j in range(len(x)):
		# sample proposal)
		x_prop[j] = sym_proposal(j, x[j])

		# compute accept/reject probability and sample
		acc_prob = compute_acc_prob(lf, x, x_prop, acc, j, active_frontier, affected_vars, dp_dicts, gamma)
		acc[j] = np.random.binomial(1, acc_prob)

	xk = x.copy()
	xk[acc == 1.0] = x_prop[acc == 1.0]
	
	return xk


def compute_acc_prob(lf, x, x_prop, acc, j, active_frontier, affected_vars, dp_dicts, gamma = .99):
	''' Computes the acceptance probability at step j. Intended for use only as
		a subroutine of the "ordered_metro" function.

		This calculation is based on the observed sequence of proposals and 
		accept/rejects of the steps before j, and the configuration of variables after j
		specified by the acceptances after j.

		Args:
			lf (function that takes a 1-dim numpy array) : the log probability density
			x (1-dim numpy array, length p) : the observed sample
			x_prop (1-dim numpy array, length p) : the sequence of proposals.
				Only needs to be filled up to the last nonzero entry of 'acc'.
			acc (1-dim 0-1 numpy array, length p) : the sequence of acceptances (1) 
				and rejections (0).
			j (int) : the active index
			active_fontier (list of lists) : as above
			affected_vars (list of lists) : entry j gives the set of variables occuring
				before j that are affected by j's value. This is a pre-processed version of 
				"active frontier" that contains the same information in a convenient form.
			dp_dicts (list of dicts) : stores the results of calls to this function.
			gamma (float) : multiplier for the acceptance probability, between 0 and 1.

	'''

	# return entry if previously computed
	key = acc[active_frontier[j]].tostring()
	if key in dp_dicts[j]:
		return(dp_dicts[j][key])
	
	# otherwise, compute the entry
	acc0 = acc.copy() #rejection pattern if we reject at j
	acc1 = acc.copy() #rejection pattern if we accept at j
	acc0[j] = 0
	acc1[j] = 1

	# compute terms from the query to the density
	x_temp1 = x.copy()
	x_temp1[acc == 1] = x_prop[acc == 1] # new point to query
	x_temp1[0:j] = x[0:j]
	x_temp1[j] = x[j]
	x_temp2 = x_temp1.copy()
	x_temp2[j] = x_prop[j] #new point to query if proposal accepted
	ld_obs = lf(x_temp1) #log-density with observed point at j
	ld_prop = lf(x_temp2) #log-density with proposed point at j

	#loop across history to adjust for the observed knockoff sampling pattern
	for j2 in affected_vars[j]:
		p0 = compute_acc_prob(lf, x, x_prop, acc0, j2, active_frontier, affected_vars, dp_dicts, gamma)
		p1 = compute_acc_prob(lf, x, x_prop, acc1, j2, active_frontier, affected_vars, dp_dicts, gamma)
		if(acc[j2] == 1):
			ld_obs += np.log(p0)
			ld_prop += np.log(p1)
		else:
			ld_obs += np.log(1 - p0)
			ld_prop += np.log(1 - p1)

	#probability of acceptance at step j, given past rejection pattern
	acc_prob = gamma * min(1, np.exp(ld_prop - ld_obs)) 
	dp_dicts[j][acc[active_frontier[j]].tostring()] = acc_prob #store result

	return acc_prob

