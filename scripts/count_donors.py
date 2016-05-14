#!usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import heapq

g = open('graph.csv', 'r')

numProj = 1829
cdonors = np.zeros(numProj)
for line in g:
    lspl = line.split(',')
    cdonors[int(lspl[0])] += 1

#print "minimum"
#print heapq.nsmallest(500, cdonors)
proj = [12,1419,865,146,1653,1176]
for i in xrange(len(proj)):
    print "Project %d : %d" % (proj[i],cdonors[proj[i]])

#n,bins,patch = plt.hist(cdonors,bins=[0,10,20,30,40,50,60,70,80,90,100],cumulative=True,normed=False,orientation='vertical')
#bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,500]
#bins = [0,10,20,75,100,125]
bins = [0,10,20,30,40,50,60,70,80,90,100]
n,bins,patch = plt.hist(cdonors,bins=bins,cumulative=True,normed=False,orientation='vertical')
print n
print bins
cumper = (n/float(numProj))*100
print cumper
for i in xrange(cumper.size):
    plt.annotate('%.2f'%cumper[i], xy=(bins[i]+1,n[i]))
# plot
plt.xlabel("Number of donors", size=10, style='italic')
plt.ylabel("Number of Proj", size=10, style='italic')
plt.show()

thresh = 1
cdonors[cdonors <= thresh] = 0
cdonors[cdonors > thresh] = 1
cdonors = cdonors.astype(int)

print np.sum(cdonors)
#np.savetxt('out1.csv',cdonors,fmt='%d',delimiter=',')
indices = list(xrange(numProj))
#np.savetxt("SC_200.csv", zip(indices,cdonors), header="index, is_successful", delimiter = ',')
np.savetxt("SC_200.csv", zip(indices,cdonors), header="index,is_successful", comments='', delimiter = ',')
