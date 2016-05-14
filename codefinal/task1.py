import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering as SC
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from scipy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding as SE

##################### Helper Functions #########################

def intersect(a, b):
	return float(len( a & b ))

def union(a,b):
	return float(len( a | b ))

def minimum(a,b):
	return float(min( len(a),len(b) ))

def product(a,b):
	return float(len(a) * len(b))

## Print label of given semi-supervised projects
def print_labels(label):
	idx_1 = [12,1419,865]
	idx_0 = [146,1653,1176]
	print "label 1 :",
	print label[idx_1]
	print "label 0 :",
	print label[idx_0]

## Calculate centroids
def centroid_all(X):
	dim = np.shape(X)[1]
	centroids = np.zeros((2,dim))
	idx_1 = [12,1419,865]
	idx_0 = [146,1653,1176]
	centroids[0] = centroid(X,idx_0)
	centroids[1] = centroid(X,idx_1)
	return centroids

## Centroid helper
def centroid(X,i):
	dim = np.shape(X)[1]
	sum = np.zeros(dim)
	sum = X[i[0]] + X[i[1]] + X[i[2]]
	return sum/3


#################### Reading and cleaning the data ####################

## Read social and evolution data but we don't use it in this code
print "reading social_and_evolution data..."
social = np.genfromtxt('social_and_evolution.csv', delimiter = ',')
print "read social_and_evolution data"

## Read and clean the graph
print "reading the backer network..."
graph = np.genfromtxt('graph.csv', delimiter = ',')

row = graph[:,0]
col = graph[:,1]
val = graph[:,2]

graph_sparse = coo_matrix((val, (row, col)), shape=(1829,146983197))

print "cleaing the graph..."
## No. of donations of all donors
donor_donations 	= graph_sparse.sum(0).getA1()

## No. of donations received by each project
project_donations 	= graph_sparse.sum(1).getA1()

## Project and donor indices
donor_idx 	= donor_donations.nonzero()[0]
project_idx	= np.arange(1829)

## Pruning graph matrix by just keeping
## the donors who made some contribution
graph_sparse = graph_sparse.tocsr()
pruneG = graph_sparse[:,donor_idx]

pruneG = pruneG.toarray()

## One more level of pruning
## removing all the donors with only 1 donations
## as these won't contribute in similarity matrix
#pruneGl2 = pruneG[:,pruneG.sum(0) != 1]
#pruneFl2set = pruneGl2.sum(1)

g = pruneG

print "clean graph created"

#################### Constructing similarity matrix ##############
## Initialize an empty 1829x1829 matrix
adj_mat = np.zeros((1829,1829))

sets = list()
for i in xrange(0,1829):
	a = g[i,:]
	sets.append(set(a))

for i in xrange(0,1829):
	for j in xrange(i+1, 1829):
		a = sets[i]
		b = sets[j]
		#adj_mat[i][j] = adj_mat[j][i] = intersect(a,b) / minimum(a,b) #as mentioned in report
		#adj_mat[i][j] = adj_mat[j][i] = intersect(a,b) / union(a,b) #as mentioned in report
		if intersect(a,b) > 0:
			adj_mat[i][j] = adj_mat[j][i] = 1

## Perform spectral embedding(Manually)
## We tried it but we don't use it in this code
se = SE(adj_mat, n_components=100, drop_first=True)
# Make D matrix
D = np.diag(adj_mat.sum(1))

# Make laplacian
Dinv = LA.fractional_matrix_power(D,-0.5)
L = np.eye(1829) - np.dot(np.dot(Dinv,adj_mat),Dinv)

# Calculate eigenvector of laplacian
eigval,eigvec = np.linalg.eig(L)

# Sort from smallest to largest
idx = np.argsort(eigval)
eigval = eigval[idx]
eigvec = eigvec[:,idx]

## Perform spectral embedding (Using library function)
## We use this instead
print "Performing spectral embedding"
se = SE(adj_mat, n_components=202, drop_first=True)

## Perform Kmeans with explicit centroids
centroids = centroid_all(se)

# Flip centroids (Method 5, Best)
centroids_new = np.zeros((2,202))
centroids_new[0] = centroids[1]
centroids_new[1] = centroids[0]

print "Performing KMeans.."
model = KMeans(n_clusters = 2)
label_kmeans = model.fit_predict(se)

print_labels(label_kmeans)

# GMM (Method 5, not best)
print "Performing GMM.."
gmm = GMM(n_components=2)
gmmlabels = gmm.fit_predict(se)

print_labels(gmmlabels)

indices = list(xrange(1829))
#np.savetxt("task1_kmeans_init_.csv", zip(indices,label_kmeans), header="index,is_successful",
#	comments='', delimiter = ',')


