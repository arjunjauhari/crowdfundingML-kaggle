import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.cluster.bicluster import SpectralCoclustering as SCC
from scipy import linalg as la
import matplotlib.pyplot as plt
import sklearn
import scipy

## calculate centroids
def centroid_all(X):
    dim = np.shape(X)[1]
    centroids = np.zeros((13,dim))
    idx_0 = [80,452,1688]
    idx_1 = [262,449,1714]
    idx_2 = [396,1032,1261]
    idx_3 = [226,1189,1200]
    idx_4 = [501,1361,1740]
    idx_5 = [810,1412,1173]
    idx_6 = [1553,1576,1650]
    idx_7 = [270,1798,1800]
    idx_8 = [1385,1453,1686]
    idx_9 = [598,670,684]
    idx_10 = [900,926,1191]
    idx_11 = [232,643,1539]
    idx_12 = [491,594,1168]
    centroids[0] = centroid(X,idx_0)
    centroids[1] = centroid(X,idx_1)
    centroids[2] = centroid(X,idx_2)
    centroids[3] = centroid(X,idx_3)
    centroids[4] = centroid(X,idx_4)
    centroids[5] = centroid(X,idx_5)
    centroids[6] = centroid(X,idx_6)
    centroids[7] = centroid(X,idx_7)
    centroids[8] = centroid(X,idx_8)
    centroids[9] = centroid(X,idx_9)
    centroids[10] = centroid(X,idx_10)
    centroids[11] = centroid(X,idx_11)
    centroids[12] = centroid(X,idx_12)
    return centroids

def centroid(X,i):
    dim = np.shape(X)[1]
    sum = np.zeros(dim)
    sum = X[i[0]] + X[i[1]] + X[i[2]]
    return sum/3

## change labels function
def change_labels(label, new_labels):
    label2 = np.zeros((1829,))
    for i in xrange(13):
        z = np.where(label==i)[0]
        for j in xrange(len(z)):
            label2[z[j]] = new_labels[i]
    return label2

def print_labels(label,change=0):
    idx_0 = [80,452,1688]
    idx_1 = [262,449,1714]
    idx_2 = [396,1032,1261]
    idx_3 = [226,1189,1200]
    idx_4 = [501,1361,1740]
    idx_5 = [810,1412,1173]
    idx_6 = [1553,1576,1650]
    idx_7 = [270,1798,1800]
    idx_8 = [1385,1453,1686]
    idx_9 = [598,670,684]
    idx_10 = [900,926,1191]
    idx_11 = [232,643,1539]
    idx_12 = [491,594,1168]
    print "label 0 :",
    print label[idx_0]
    print "label 1 :",
    print label[idx_1]
    print "label 2 :",
    print label[idx_2]
    print "label 3 :",
    print label[idx_3]
    print "label 4 :",
    print label[idx_4]
    print "label 5 :",
    print label[idx_5]
    print "label 6 :",
    print label[idx_6]
    print "label 7 :",
    print label[idx_7]
    print "label 8 :",
    print label[idx_8]
    print "label 9 :",
    print label[idx_9]
    print "label 10 :",
    print label[idx_10]
    print "label 11 :",
    print label[idx_11]
    print "label 12 :",
    print label[idx_12]
    if change == 1:
        nl = np.zeros(13)
        nl[scipy.stats.mode(label[idx_0])[0][0]] = 0
        nl[scipy.stats.mode(label[idx_1])[0][0]] = 1
        nl[scipy.stats.mode(label[idx_2])[0][0]] = 2
        nl[scipy.stats.mode(label[idx_3])[0][0]] = 3
        nl[scipy.stats.mode(label[idx_4])[0][0]] = 4
        nl[scipy.stats.mode(label[idx_5])[0][0]] = 5
        nl[scipy.stats.mode(label[idx_6])[0][0]] = 6
        nl[scipy.stats.mode(label[idx_7])[0][0]] = 7
        nl[scipy.stats.mode(label[idx_8])[0][0]] = 8
        nl[scipy.stats.mode(label[idx_9])[0][0]] = 9
        nl[scipy.stats.mode(label[idx_10])[0][0]] = 10
        nl[scipy.stats.mode(label[idx_11])[0][0]] = 11
        nl[scipy.stats.mode(label[idx_12])[0][0]] = 12
        newla = change_labels(label,nl)
        return newla

def tellError(label):
    idx = {}
    idx[0] = [80,452,1688]
    idx[1] = [262,449,1714]
    idx[2] = [396,1032,1261]
    idx[3] = [226,1189,1200]
    idx[4] = [501,1361,1740]
    idx[5] = [810,1412,1173]
    idx[6] = [1553,1576,1650]
    idx[7] = [270,1798,1800]
    idx[8] = [1385,1453,1686]
    idx[9] = [598,670,684]
    idx[10] = [900,926,1191]
    idx[11] = [232,643,1539]
    idx[12] = [491,594,1168]
    err = 0
    for key in idx:
        err += np.count_nonzero(label[idx[key]] != key)
    return err/39.0


description = np.genfromtxt('../data/description.csv', delimiter = ',')

row = description[:,0]
col = description[:,1]
val = description[:,2]

X = coo_matrix((val, (row, col)), shape=(1829,8000)).todense() # description

# Sweep RP
#np.random.seed(1)
#d = np.shape(X)[1]
#temp = np.sign(np.random.randn(d,1010))
#for k in xrange(990,1010,1):
#    K = k
#    #np.random.seed(1)
#    #d = np.shape(X)[1]
#    #WRp = np.sign(np.random.randn(d,K))/np.sqrt(K)  # dxK
#    WRp = temp[:,0:K]
#    YRp = np.dot(X, WRp)     # NxK
#    # Do Kmeans
#    cen = centroid_all(YRp)
#    model = KMeans(n_clusters = 13,init=cen)
#    label = model.fit_predict(YRp)
#    #print_labels(label)
#    print "%d : %f" % (K,tellError(label))
#    indices = list(xrange(1829))
#    np.savetxt("task2_RP%dsd1_KMeans_init.csv" % K, zip(indices,label),fmt='%d', header="index,category",comments='', delimiter = ',')
#
# Sweep PCA
# Perform PCA
for k in xrange(100,2100,100):
    uXPca,sXPca,wXPca = la.svd(X, full_matrices=False)
    K=k
    wXPca = wXPca[0:K,:]    # Kxd
    muXPca = np.mean(X,0)
    YPca = np.dot(X-muXPca, wXPca.T)    # NxK
    # Perform Kmeans


    # Do Kmeans
    cen = centroid_all(YPca)
    model = KMeans(n_clusters = 13,init=cen)
    label = model.fit_predict(YPca)
    #print_labels(label)
    #tellError(label)
    print "%d : %f" % (K,tellError(label))
    indices = list(xrange(1829))
    np.savetxt("task2_PCA%d_KMeans_init.csv" % K, zip(indices,label),fmt='%d', header="index,category",comments='', delimiter = ',')

## Perform PCA
#uXPca,sXPca,wXPca = la.svd(X, full_matrices=False)
#K=1500
#wXPca = wXPca[0:K,:]    # Kxd
#muXPca = np.mean(X,0)
#YPca = np.dot(X-muXPca, wXPca.T)    # NxK
## Perform Kmeans
#
#
## Do Kmeans
#cen = centroid_all(YPca)
#model = KMeans(n_clusters = 13,init=cen)
#label = model.fit_predict(YPca)
#print_labels(label)
#tellError(label)
#
## Do SC
#cen = centroid_all(X)
#model = SC(n_clusters = 13,init=cen)
#label = model.fit_predict(X)
#print_labels(label)
#
## Do AF
##af = AffinityPropagation(preference=-1,max_iter=500).fit(X)
##cluster_centers_indices = af.cluster_centers_indices_
##label = af.labels_
##print_labels(label)
##n_clusters_ = len(cluster_centers_indices)
##
##print('Estimated number of clusters: %d' % n_clusters_)
#
## Make similarity matrix
#sim = sklearn.metrics.pairwise.laplacian_kernel(X)
#sim = sklearn.metrics.pairwise.pairwise_kernels(X,metric='cosine')
#sim = sklearn.metrics.pairwise.pairwise_kernels(X,metric='linear')
#
#sim = sim-np.amin(sim)
#
#sim = sklearn.metrics.pairwise.pairwise_kernels(X,metric='rbf',gamma=0.9)
## Make spectral embedding
#eigvec = sklearn.manifold.spectral_embedding(sim, n_components=250)
#
## Do Kmeans
#cen = centroid_all(eigvec)
#model = KMeans(n_clusters = 13,init=cen)
#label = model.fit_predict(eigvec)
#nlabel = print_labels(label)
##print_labels(nlabel,change=0)
##
### Do SC
#model = SC(n_clusters = 13)
#label = model.fit_predict(eigvec)
#print_labels(label)
#
## Do AF
#af = AffinityPropagation(preference=-1,max_iter=500).fit(eigvec)
#cluster_centers_indices = af.cluster_centers_indices_
#label = af.labels_
#print_labels(label)
#n_clusters_ = len(cluster_centers_indices)
#
#print('Estimated number of clusters: %d' % n_clusters_)
#
## GMM
#gmm = GMM(n_components=6)
#gmmlabels = gmm.fit_predict(eigvec)
#print_labels(gmmlabels)
#
#indices = list(xrange(1829))
#np.savetxt("task2_PCA1500sd10_KMeans_init.csv", zip(indices,label),fmt='%d', header="index,category",comments='', delimiter = ',')
##
##
