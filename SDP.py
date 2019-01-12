# This is a script for solving the SDP in second-order knockoff construction
# and it could be used for covariance-guided prposals in MH-SCEP.
# The R function can be found in the "knockoff" package.
import numpy as np
from pydsdp.dsdp5 import dsdp

# creating correlation matrix
# replace it with your own correlation matrix if needed
p = 10
rhos = [0.6] * (p-1)
cormatrix = np.ones([p,p])
for i in range(p-1):
    for j in range(i+1,p):
        for k in range(i,j):
            cormatrix[i,j] = cormatrix[i,j]*rhos[k]
        cormatrix[j,i] = cormatrix[i,j]
G = cormatrix
# finished creating correlation matrix

# solve SDP for G
p = np.size(G[0,:])
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
print(result)
# result['y'] is the vector s we want
