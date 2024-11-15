# (c) 2024 Patrick Nieman and Varun Sahay
# Converts rock path data to weighted float one-hot form

import numpy as np

def onehot(i,n):
    a=np.zeros(n)
    a[i]=1
    return a

rocks=74 #number of classes

#Read rock path data
path="/Applications/CS230 Data/pathData20.csv"
data=[]
with open(path,"r") as f:
    l=f.readline()
    while l:
        data.append([int(i) for i in l.replace("\n","").split(",")])
        l=f.readline()
out=np.zeros((len(data),rocks))

#Convert to normalized one-hot vectors and save
i=0
for datum in data:
    line=np.zeros((rocks))
    for j in datum:
        line+=onehot(j,rocks)
    out[i,:]=line/np.sum(line)
    i+=1
np.savetxt("/Applications/CS230 Data/pathData201h.csv",out,delimiter=",",fmt='%s')