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
import sklearn
import hinton
import heapq
from sklearn.mixture import GMM

def centroid_all(X):
    dim = np.shape(X)[1]
    centroids = np.zeros((2,dim))
    idx_1 = [12,1419,865]
    idx_0 = [146,1653,1176]
    centroids[0] = centroid(X,idx_0)
    centroids[1] = centroid(X,idx_1)
    return centroids

def centroid(X,i):
    dim = np.shape(X)[1]
    sum = np.zeros(dim)
    sum = X[i[0]] + X[i[1]] + X[i[2]]
    return sum/3

## print labels
def print_labels(label):
    idx_1 = [12,1419,865]
    idx_0 = [146,1653,1176]
    print "label 1 :",
    print label[idx_1]
    print "label 0 :",
    print label[idx_0]

def intersect(a, b):
    return list(set(a) & set(b))

graph = np.genfromtxt('../data/graph.csv', delimiter = ',')
social = np.genfromtxt('../data/social_and_evolution.csv', delimiter = ',')

row = graph[:,0]
col = graph[:,1]
val = graph[:,2]

graph_sparse = coo_matrix((val, (row, col)), shape=(1829,146983197))

## no of donations of all donors
donor_donations 	= graph_sparse.sum(0).getA1()

## no of donations received by each project
project_donations 	= graph_sparse.sum(1).getA1()

# n largest
#print heapq.nlargest(50, project_donations)

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
pruneG = graph_sparse[:,donor_idx]

pruneG = pruneG.toarray()
## count number of non-zero values
print np.count_nonzero(pruneG)

# one more level of pruning
pruneGl2 = pruneG[:,pruneG.sum(0) != 1]

pruneFl2set = pruneGl2.sum(1)

print pruneGl2.shape

numProj = 1829
adMat = np.zeros((numProj,numProj))
for i in xrange(pruneGl2.shape[0]):
    edge = np.nonzero(pruneGl2[:,pruneGl2[i,:] != 0])[0]
    edge = edge[edge != i]
    adMat[i,edge] = 1

    #for k in xrange(len(edge)):
    #    #adMat[i,edge[k]] += 1.0/len(edge)
    #    #adMat[i,edge[k]] += 1
    #    adMat[i,edge[k]] += 1.0/min(pruneFl2set[i],pruneFl2set[edge[k]])

# normalizing, needed??
adMat = adMat/np.max(adMat)
print np.count_nonzero(adMat)

#hinton.hinton(adMat,"adMat")
#np.fill_diagonal(adMat,0)
temp = np.array(xrange(1829))+1
adDup = np.hstack((temp[:,np.newaxis],adMat))
temp = np.array(xrange(1830))
adDup = np.vstack((temp,adDup))
np.savetxt("aff.csv", adDup, delimiter=';')

# social similarity matrix
socialsim = sklearn.metrics.pairwise.pairwise_kernels(social,metric='rbf',gamma=0.8)

adMat = adMat+socialsim

# Make D matrix
D = np.diag(adMat.sum(1))

# Make laplacian
Dinv = la.fractional_matrix_power(D,-0.5)
L = np.eye(numProj) - np.dot(np.dot(Dinv,adMat),Dinv)

#L = D - adMat

# Cal eigenvector of laplacian
eigval,eigvec = np.linalg.eigh(L)

# Sort from smallest to largest
idx = np.argsort(eigval)
eigval = eigval[idx]
eigvec = eigvec[:,idx]

 Perform PCA
uXPca,sXPca,wXPca = la.svd(eigvec, full_matrices=False)
K=500
wXPca = wXPca[0:K,:]    # Kxd
muXPca = np.mean(eigvec,0)
YPca = np.dot(eigvec-muXPca, wXPca.T)    # NxK
for k in xrange(numProj):
    print "k = %d" % k
    print eigval[2*k];
    print eigval[2*k+1];

    # take the K eigenvector corresponding to K smallest eigenvalue
    Y = eigvec[:,[2*k,2*k+1]]#[:,1:100]
    #Y = eigvec[:,[0,1]]
    plt.scatter(Y[:,0],Y[:,1])
    for i in xrange(Y[:,0].size):
        plt.annotate('{0}'.format(i), xy=(Y[i,0],Y[i,1]))
    plt.axis('equal')
    plt.show()

Y = eigvec#[:,0:20]
# do kmeans on compressed data represented by Y
k_means = cluster.KMeans(60)

k_means.fit(Y)

print k_means.labels_

proj = [12,1419,865,146,1653,1176]
#for k in xrange(1,numProj):
#    print "k = %d ======================" % k
#    # do kmeans on compressed data represented by Y
#    k_means = cluster.KMeans(2)
#
#    k_means.fit(eigvec[:,0:k+1])
#
#    print k_means.labels_
#    # Spectral clustering
#    spec = cluster.SpectralClustering(2)#,affinity='precomputed')
#    spec.fit(eigvec[:,0:k+1])
#
#    print spec.labels_
#
#    # GMM
#    gmm = GMM(n_components=2)
#    gmmlabels = gmm.fit_predict(eigvec[:,0:k+1])
#
#    print k_means.labels_[proj]
#    print spec.labels_[proj]
#    print gmmlabels[proj]
#    print np.count_nonzero(k_means.labels_)
#    print np.count_nonzero(spec.labels_)
#    print np.count_nonzero(gmmlabels)

    # save result
    #indices = list(xrange(numProj))
    #np.savetxt("out_t1.csv", zip(indices,gmmlabels),fmt='%d', header="index,is_successful", comments='', delimiter = ',')
    #np.savetxt("out_t1.csv", zip(indices,gmmlabels),fmt='%d', header="index,is_successful", comments='', delimiter = ',')
    #np.savetxt("out_t1.csv", zip(indices,gmmlabels),fmt='%d', header="index,is_successful", comments='', delimiter = ',')
## GMM
#gmm = GMM(n_components=2)
#gmmlabels = gmm.fit_predict(eigvec[:,0:200])
#
#proj = [12,1419,865,146,1653,1176]
#
#print spec.labels_[proj]
#print gmmlabels[proj]
##print k_means.labels_[proj]
#print np.count_nonzero(spec.labels_)
#print np.count_nonzero(gmmlabels)
##print np.count_nonzero(k_means.labels_)
#
##print project_donations[np.nonzero(spec.labels_)]
#
eigvec = sklearn.manifold.spectral_embedding(adMat, n_components=210)
print eigvec.shape
# GMM
gmm = GMM(n_components=2)
gmmlabels = gmm.fit_predict(eigvec)
print np.count_nonzero(gmmlabels)
## save result
indices = list(xrange(numProj))
print gmmlabels[proj]
np.savetxt("gmm_210.csv", zip(indices,gmmlabels),fmt='%d', header="index,is_successful", comments='', delimiter = ',')

