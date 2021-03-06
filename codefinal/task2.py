import numpy as np
import scipy
import sklearn
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from sklearn.manifold import spectral_embedding as SE
from scipy import linalg as la


##################### Helper Functions #########################

def cosine(a,b):
	return np.sum(np.multiply(a,b))

## print labels function
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

## change labels function
def change_labels(label, new_labels):
	label2 = np.zeros((1829,))
	for i in xrange(13):
		z = np.where(label==i)[0]
		for j in xrange(len(z)):
			label2[z[j]] = new_labels[i]
	return label2

## change labels function
def change_labels_n(label, new_labels, n):
	label2 = np.zeros((1829,))
	for i in xrange(n):
		z = np.where(label==i)[0]
		for j in xrange(len(z)):
			label2[z[j]] = new_labels[i]
	return label2

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


#################### End of Helper Functions #####################

## read and construct description graph
print "reading description graph..."

description = np.genfromtxt('description.csv', delimiter = ',')

row = description[:,0]
col = description[:,1]
val = description[:,2]

X = coo_matrix((val, (row, col)), shape=(1829,8000)).todense() # description

print "description graph constructed"

## construct similarity matrix
adj_mat = np.zeros((1829,1829))

for i in xrange(0,1829):
	for j in xrange(i+1, 1829):
		a = X[i]
		b = X[j]
		c = cosine(a,b)
		adj_mat[i][j] = adj_mat[j][i] = c
		if c < 0:
			adj_mat[i][j] = adj_mat[j][i] = 0


## perform PCA (Method 6)

print "performing pca with k=1500 dimensions..."

uXPca,sXPca,wXPca = la.svd(X, full_matrices=False)
K=1500
wXPca = wXPca[0:K,:]    # Kxd
muXPca = np.mean(X,0)
YPca = np.dot(X-muXPca, wXPca.T)    # NxK

## perform SVD

#mean = np.mean(X, 0)
#X_centered = X - mean
#U, s, V = np.linalg.svd(X_centered)

## Dimensionality reduction

#k = 1200
#ratio = sum(np.power(s[0:k],2)) / sum(np.power(s,2))
#print ratio
#W = V.T[:,0:k]
#Y = np.dot(X_centered, W)
#Y = np.dot( U[:,0:k], np.diag(s[0:k]) )

## Perform random projection

#K = 1200
#np.random.seed(1)
#d = np.shape(X)[1]
#WRp = np.sign(np.random.randn(d,K))/np.sqrt(K)  # dxK
#YRp = np.dot(X, WRp)     # NxK


## Perform KMeans with given centroids (Method 6)

print "performing KMeans with explicit centroids..."
centroids = centroid_all(YPca)

model = KMeans(n_clusters = 13, init=centroids)
label_kmeans = model.fit_predict(YPca)
print_labels(label_kmeans)

## perform spectral embedding
print "performing spectral embedding..."
se = SE(adj_mat, n_components=200, drop_first=True)

## Perform GMM

print "performing GMM on spectral embedded data.."
gmm = GMM(n_components=13,covariance_type='diag')
label_gmm = gmm.fit_predict(se)
print_labels(label_gmm)


# write to file KMeans prediction (method 6)
indices = list(xrange(1829))
#np.savetxt("task2_Centroids_KMeans_PCA_1500.csv", zip(indices,label_kmeans), header="index,category", 
#	comments='', delimiter = ',')


