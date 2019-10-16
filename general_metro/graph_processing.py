''' 
    Processes a graph of representing the structure of a random variable
    for use with metropolized knockoff sampling.

    Author: Stephen Bates, October 2019
'''

import numpy as np
import networkx as nx

def get_ordering(T):
    ''' Takes a junction tree and returns a variable ordering for the metro
        knockoff sampler.

        Args:
            T: A networkx graph that is a junction tree.
                Nodes must be sets with elements 0,...,p-1.
                e.g.: width, T = treewidth_decomp(G)

        Returns:
            order : a numpy array with unique elements 0,...,p-1
            active_frontier (list of lists) : the set of variables active
                at future steps. Intended as input for the "single_metro" function.

    '''

    T = T.copy()
    order = []
    active_frontier = []
    
    while(T.number_of_nodes() > 0):
        gen = (x for x in T.nodes() if T.degree(x) <= 1)
        active_node = next(gen)
        parents = list(T[active_node].keys())
        if(len(parents) == 0):
            active_vars = set(active_node)
            activated_set = active_vars.copy()
        else:
            active_vars = set(active_node.difference(parents[0]))
            activated_set = active_vars.union(parents[0]).difference(set(order))
        for i in active_vars:
            order += [i]
            frontier = list(activated_set.difference(set(order)))
            active_frontier += [frontier]
        T.remove_node(active_node)
    
    return [np.array(order), active_frontier]