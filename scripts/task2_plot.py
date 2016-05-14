import numpy as np
import matplotlib.pyplot as plt

f1 = open("task2_PCA100-2000_Kmeans.csv","r")

idx = []
tr = []
pubs = []
pris = []
cnt = 0
for line in f1:
    llist = line.split(',')
    idx.append(float(llist[0]))
    tr.append(1.0 - float(llist[1]))
    pubs.append(float(llist[2]))
    pris.append(float(llist[3]))

    cnt+=1

plt.plot(idx,tr,'-r*',label='Training')
plt.plot(idx,pubs,'-b+',label='Public Score')
plt.plot(idx,pris,'-gD',label='Private Score')
plt.xlabel("K [PCA]")
plt.ylabel("Accuracy")
plt.title("Accuracy vs PCA Dimension Sweep")
plt.legend(loc="upper left")
plt.show()
