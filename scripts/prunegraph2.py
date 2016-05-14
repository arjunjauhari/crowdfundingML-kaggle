import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster.bicluster import SpectralCoclustering as SCC
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy import linalg as la
import pdb
import sklearn

def intersect(a, b):
	return list(set(a) & set(b))

graph = np.genfromtxt('../data/graph.csv', delimiter = ',')

row = graph[:,0]
col = graph[:,1]
val = graph[:,2]

graph_sparse = coo_matrix((val, (row, col)), shape=(1829,146983197))

## no of donations of all donors
donor_donations 	= graph_sparse.sum(0).getA1()

## no of donations received by each project
project_donations 	= graph_sparse.sum(1).getA1()

## project and donor indices
donor_idx 	= donor_donations.nonzero()[0]
project_idx	= np.arange(1829)

# release donor_donations
donor_donations = 0

#print len(donor_idx)
#print max(donor_donations)

## print degree distribution
#print max(project_donations)
#plt.show()

graph_sparse = graph_sparse.tocsr()
#pruneG = graph_sparse[:,(graph_sparse != 0).sum(axis=0) != 0]
pruneG = graph_sparse[:,donor_idx]

### reindex donors
#col2 = np.zeros(len(col))
#
#for i in xrange(len(col)):
#	donor = col[i]
#	idx = np.where(donor_idx==donor)[0][0]
#	col2[i] = idx
#
### write to file
#col2 = col2 + 0
#graph_pruned = np.vstack((row, col2, val))
#graph_pruned = np.transpose(graph_pruned)
#np.savetxt("graph_pruned.csv", graph_pruned, delimiter=',')
#np.savetxt("pruneG.csv", pruneG, delimiter=',')

## create new pruned graph
#g = coo_matrix((val, (row, col2)), shape=(1829,289344))

pruneG = pruneG.toarray()
## count number of non-zero values
print np.count_nonzero(pruneG)

# Perform random projection
K = 100
np.random.seed(1)
d = pruneG.shape[1]
WRp = np.sign(np.random.randn(d,K))/np.sqrt(K)  # dxK
YRp = np.dot(pruneG, WRp)     # NxK
# Perform PCA
#uXPca,sXPca,wXPca = la.svd(YRp, full_matrices=False)
#K=20
#wXPca = wXPca[0:K,:]    # Kxd
#muXPca = np.mean(YRp,0)
#YPca = np.dot(YRp-muXPca, wXPca.T)    # NxK

# Perform random projection
#K = 2
#d = pruneG.shape[1]
#WRp = np.sign(np.random.randn(d,K))/np.sqrt(K)  # dxK
#YRp = np.dot(pruneG, WRp)     # NxK
#
#np.savetxt("pruneGRp.csv", YRp, delimiter=',')

# Get adjacency matrix
#adjRBF = sklearn.metrics.pairwise.pairwise_kernels(YPca,metric='rbf')
adjRBF = sklearn.metrics.pairwise.pairwise_kernels(YRp,metric='rbf',gamma=0.3)
#adjRBF = sklearn.metrics.pairwise.pairwise_kernels(project_donations.reshape(-1,1),metric='rbf',gamma=0.00001)
#temp = np.array(xrange(1829))+1
#adjRBF = np.hstack((temp[:,np.newaxis],adjRBF))
#temp = np.array(xrange(1830))
#adjRBF = np.vstack((temp,adjRBF))
#np.savetxt("aff.csv", adjRBF, delimiter=';')
# scatter plot X
#plt.scatter(YPca[:,0], YPca[:,1])
#plt.show()
#
### perform spectral clustering
##sc = SC(n_clusters=2)
## do kmeans on compressed data represented by Y
#k_means = cluster.KMeans(2)
#
#k_means.fit(YPca)
#
#print k_means.labels_
#
spec = cluster.SpectralClustering(2,affinity='precomputed')
#spec = cluster.SpectralClustering(2,affinity='nearest_neighbors')
#spec.fit(YPca)
spec.fit(adjRBF)

print spec.labels_
proj = [12,1419,865,146,1653,1176]

#print k_means.labels_[proj]
print spec.labels_[proj]
print np.count_nonzero(spec.labels_)

# dump affinity matrix
#spec.affinity_matrix_.toarray()
#np.savetxt("aff.csv", spec.affinity_matrix_.toarray(), delimiter=',')
#np.savetxt("aff.csv", spec.affinity_matrix_, delimiter=',')

# save result
numProj = 1829
indices = list(xrange(numProj))
#np.savetxt("SC_200.csv", zip(indices,cdonors), header="index, is_successful", delimiter = ',')
np.savetxt("out_pd.csv", zip(indices,spec.labels_),fmt='%d', header="index,is_successful", comments='', delimiter = ',')
