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
import hinton
import heapq

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

#FIXME: add weight
numProj = 1829
adMat = np.zeros((numProj,numProj))
for i in xrange(pruneGl2.shape[0]):
    edge = np.nonzero(pruneGl2[:,pruneGl2[i,:] != 0])[0]
    edge = edge[edge != i]
    #adMat[i,edge] = 1

    for k in xrange(len(edge)):
        #adMat[i,edge[k]] += 1.0/len(edge)
        #adMat[i,edge[k]] += 1
        adMat[i,edge[k]] += 1.0/min(pruneFl2set[i],pruneFl2set[edge[k]])

pdb.set_trace()

out = np.ones(numProj)
#out[np.nonzero(adMat[12])[0]] = 1
#out[np.nonzero(adMat[1419])[0]] = 1
#out[np.nonzero(adMat[865])[0]] = 1
#out[np.nonzero(adMat[13])[0]] = 1
#out[np.nonzero(adMat[964])[0]] = 1
#out[np.nonzero(adMat[252])[0]] = 1

haack = np.array([1, 1539, 1028, 5, 1739, 1544, 684, 1111, 524, 1313, 527, 1553, 531, 1655, 533, 1558, 1559, 24, 1050, 539, 1564, 1825, 517, 33, 34, 1058, 36, 1061, 1062, 39, 1231, 860, 42, 555, 598, 946, 46, 560, 50, 1075, 53, 566, 57, 570, 59, 1084, 573, 574, 1087, 576, 1059, 1741, 1603, 1595, 1008, 1163, 1095, 1548, 586, 1099, 1622, 1102, 79, 592, 1204, 595, 1108, 597, 86, 87, 1112, 601, 90, 92, 1630, 95, 1632, 1121, 1367, 613, 102, 615, 1128, 1777, 1569, 1644, 109, 622, 1572, 112, 1640, 626, 628, 630, 119, 632, 1145, 122, 1147, 124, 639, 1154, 1643, 644, 646, 1671, 1057, 139, 140, 1165, 1166, 655, 656, 536, 1683, 1471, 1174, 152, 153, 1690, 1179, 668, 1695, 672, 1563, 1188, 1222, 678, 1649, 1192, 681, 1706, 171, 172, 177, 692, 183, 696, 185, 1055, 700, 191, 1312, 1730, 707, 197, 710, 199, 201, 1738, 631, 205, 206, 717, 1232, 888, 722, 723, 1525, 1557, 216, 1314, 1242, 219, 1244, 733, 222, 1759, 1761, 1763, 1769, 235, 1775, 753, 1781, 1780, 757, 1048, 1065, 1272, 249, 1275, 1229, 554, 254, 1645, 1793, 1283, 772, 1494, 262, 1287, 1819, 1294, 271, 272, 1810, 1748, 277, 791, 792, 281, 1306, 795, 286, 800, 801, 290, 1317, 295, 1320, 1826, 812, 1459, 1328, 1331, 1335, 312, 607, 318, 835, 1348, 1349, 838, 1400, 841, 844, 1458, 1398, 1421, 1654, 337, 1716, 1364, 1081, 856, 1631, 348, 1338, 862, 864, 1168, 571, 356, 871, 360, 1596, 1386, 364, 365, 1390, 1392, 881, 882, 1581, 1598, 886, 376, 889, 1356, 1405, 1685, 898, 747, 388, 901, 1414, 903, 1417, 1070, 1420, 397, 664, 1426, 916, 1432, 409, 413, 1590, 1093, 913, 1442, 1480, 422, 423, 941, 1456, 434, 947, 1460, 951, 440, 959, 448, 1696, 966, 968, 969, 973, 1215, 1489, 469, 982, 761, 472, 868, 1103, 988, 477, 1502, 995, 996, 999, 490, 1517, 1518, 496, 545, 1621])
out[np.nonzero(adMat[146])[0]] = 0
out[np.nonzero(adMat[1653])[0]] = 0
out[np.nonzero(adMat[1176])[0]] = 0
out[haack] = 0
print np.count_nonzero(out)
# save result
indices = list(xrange(numProj))
np.savetxt("haack.csv", zip(indices,out),fmt='%d', header="index,is_successful", comments='', delimiter = ',')
