#!usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
import heapq

# read data from csv
data = np.genfromtxt('graph.csv', delimiter = ',')

row = data[:,0]
col = data[:,1]
val = data[:,2]

indices = set(col)

# the number of donations per donor
sumdonor = {}

for i in xrange(len(col)):
    if col[i] in sumdonor:
        sumdonor[col[i]] += 1
    else:
        sumdonor[col[i]] = 1

# Max
maxdo = max(sumdonor.values())
# n largest
print heapq.nlargest(50, sumdonor.values())

print "Max: %d" % maxdo

#print indices
print len(indices)

topdonor = []
thresh = 1
for key in sumdonor:
    if sumdonor[key] > thresh:
        topdonor.append(key)
        #print "%d : %d" % (key, sumdonor[key])

print len(topdonor)

print sumdonor[1453409]

np.savetxt("indices.csv", list(indices),fmt='%d', delimiter = ',')
