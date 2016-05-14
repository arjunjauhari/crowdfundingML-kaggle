import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.cluster.bicluster import SpectralCoclustering as SCC
from scipy import linalg as la
import matplotlib.pyplot as plt

## print labels function
def print_labels(label):
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

description = np.genfromtxt('../data/description.csv', delimiter = ',')

row = description[:,0]
col = description[:,1]
val = description[:,2]

X = coo_matrix((val, (row, col)), shape=(1829,8000)).todense() # description

## perform PCA

#pca = PCA(n_components=200)
#Y = pca.fit_transform(X.todense())

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

K = 1200
np.random.seed(1)
d = np.shape(X)[1]
WRp = np.sign(np.random.randn(d,K))/np.sqrt(K)  # dxK
YRp = np.dot(X, WRp)     # NxK

# Perform PCA
uXPca,sXPca,wXPca = la.svd(YRp, full_matrices=False)
K=20
wXPca = wXPca[0:K,:]    # Kxd
muXPca = np.mean(YRp,0)
YPca = np.dot(YRp-muXPca, wXPca.T)    # NxK
# Perform Kmeans

model = KMeans(n_clusters = 13)
label = model.fit_predict(YPca)
print_labels(label)

## Perform GMM

gmm = GMM(n_components=13)
label = gmm.fit_predict(YPca)
print_labels(label)

## Visualize in first two dimensions

for k in xrange(1829):
    # take the K eigenvector corresponding to K smallest eigenvalue
    Y = X[:,[2*k,2*k+1]]#[:,1:100]
    #Y = eigvec[:,[0,1]]
    plt.scatter(Y[:,0],Y[:,1])
    #for i in xrange(Y[:,0].size):
    #    plt.annotate('{0}'.format(i), xy=(Y[i,0],Y[i,1]))
    plt.axis('equal')
    plt.show()

## perform spectral clustering
sc = SC(n_clusters=13,gamma=0.25)
label = sc.fit_predict(Y)

## semi-supervised labels
print_labels(label)

## perform spectral coclustering
#scc = SCC(n_clusters=13)
#scc.fit(Y)


indices = list(xrange(1829))
np.savetxt("task2_KMeans_RP_100.csv", zip(indices,label2), header="index,category",
	comments='', delimiter = ',')


