import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn import cluster

def show_graph(adjacency_matrix, labels):
    # given an adjacency matrix use networkx and matlpotlib to plot the graph
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    labels = list(labels)
    for i in xrange(len(labels)):
        if labels[i]:
            labels[i]='r'
        else:
            labels[i]='c'
    gr.add_edges_from(edges)
    # nx.draw(gr) # edited to include labels
    #nx.spring_layout(gr, k =0.7, iterations=50)
    nx.draw(gr, cmap=plt.get_cmap('jet'), node_color=labels, with_labels=True)
    # now if you decide you don't want labels because your graph
    # is too busy just do: nx.draw_networkx(G,with_labels=False)
    plt.show()

# Generate random adjacency matrix
np.random.seed(50)
N = 10
#adjMat = np.random.random_integers(0,1,(N,N))
adjMat = np.random.rand(N,N)
thresh = 0.3
adjMat[adjMat > thresh] = 1
adjMat[adjMat <= thresh] = 0
adjMat = adjMat.astype(int)
#print adjMat
adjMat = (adjMat + adjMat.T)/2

np.fill_diagonal(adjMat,0)
#print(adjMat)

N2 = 10
adjMat2 = np.zeros((N2,N2))
adjMat2 = adjMat2.astype(int)
# rest of the part
for i in xrange(N2-1):
    adjMat2[i,i+1] = 1
    adjMat2[i+1,i] = 1

# concatenate
#temp1 = np.vstack((adjMat, np.zeros((5,25)).astype(int)))
#temp2 = np.vstack((np.zeros((25,5)).astype(int), adjMat2))
#adjMat = np.hstack((temp1,temp2))
#print(adjMat)
#temp1 = np.vstack((adjMat, np.zeros((20,10)).astype(int)))
#temp2 = np.vstack((np.zeros((10,10)).astype(int), adjMat, np.zeros((10,10)).astype(int)))
#temp3 = np.vstack((np.zeros((20,10)).astype(int), adjMat))
#adjMat = np.hstack((temp1,temp2,temp3))
adjMat = np.kron(np.eye(3),adjMat)

print(adjMat)
print(np.sum(adjMat, axis=1))
## Adding edges: experimental
adjMat[6,12] = 1
adjMat[12,6] = 1
adjMat[14,5] = 1
adjMat[5,14] = 1
adjMat[16,26] = 1
adjMat[26,16] = 1
#adjMat[9,4] = 1
#adjMat[4,9] = 1
#adjMat[20,22] = 1
#adjMat[7,13] = 1
#adjMat[13,7] = 1
#adjMat[4,18] = 1
#adjMat[18,4] = 1
#adjMat[18,20] = 1
#adjMat[20,18] = 1
#adjMat[9,4] = 1
#adjMat[4,9] = 1
#adjMat[20,22] = 1
#adjMat[22,20] = 1
## dump
#np.savetxt('aSpectral1.csv',adjMat,fmt='%d',delimiter=',')
#################################
#adjMat[18,29] = 1
#adjMat[29,18] = 1
#adjMat[18,25] = 1
#adjMat[25,18] = 1
#np.savetxt('aSpectral2.csv',adjMat,fmt='%d',delimiter=',')
#adjMat[6,29] = 1
#adjMat[29,6] = 1
#adjMat[6,26] = 1
#adjMat[26,6] = 1
#adjMat[9,2] = 1
#adjMat[2,9] = 1
#adjMat[5,0] = 1
#adjMat[0,5] = 1
#adjMat[1,8] = 1
#adjMat[8,1] = 1

#print adjMat
#print(np.sum(adjMat))
#show_graph(adjMat)

# Create diagonal matrix
diagMat = np.diag(adjMat.sum(axis=0))
#print diagMat

# Create Laplacian matrix
diagMatinv = la.fractional_matrix_power(diagMat,-0.5)

lapMat = np.eye(2*N+N2) - np.dot(np.dot(diagMatinv,adjMat),diagMatinv)

# Unnormalized
#lapMat = diagMat - adjMat

#print lapMat

# Cal eigenvector of laplacian
eigval,eigvec = np.linalg.eig(lapMat)

# Sort from smallest to largest
idx = np.argsort(eigval)
eigval = eigval[idx]
eigvec = eigvec[:,idx]

#print eigval
#print eigvec

# take the K eigenvector corresponding to K smallest eigenvalue
Y = eigvec[:,[0,1]]
plt.scatter(Y[:,0],Y[:,1])
for i in xrange(Y[:,0].size):
    plt.annotate('{0}'.format(i), xy=(Y[i,0],Y[i,1]))
plt.axis('equal')
plt.show()

# do kmeans on compressed data represented by Y
k_means = cluster.KMeans(2)

k_means.fit(Y)

print k_means.labels_

# do spectral
spec = cluster.SpectralClustering(2,affinity='precomputed')
spec.fit(adjMat)

print spec.labels_
#print spec.affinity_matrix_

show_graph(adjMat,k_means.labels_)
