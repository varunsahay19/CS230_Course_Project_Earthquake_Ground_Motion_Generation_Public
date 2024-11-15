# (c) 2024 Patrick Nieman and Varun Sahay
# Compiles X and Y datasets for training linear ground motion model, from a subset of the full dataset if specified


import numpy as np
import shutil
import os
import json

#Utility
def pad(a,n):
    out=""
    for i in range(n-len(a)):
        out=out+"0"
    return out+a

n=7477 #Total dataset size
max=1024 #desired records; number of X, Y entries will be max*angles
angles=10 #Per record file
indicesI=np.arange(n)

mapPath="Rock Query/requests.csv"
pathPath="/Applications/CS230 Data/pathData201h.csv"
metadataPath="/Applications/CS230 Data/metadata.csv"
tsmdPath="/Applications/CS230 Data/timeSeriesMetadata.json"

#Get ordered record IDs for available records
with open(mapPath,"r") as f:
    list=np.array(f.read().replace("\n",",").split(","))
rsnsI=list[indicesI]
indices=[]
rsns=[]

#Determine for which records full data has been preprocessed
k=0
count=0
for i in rsnsI:
    if os.path.isfile("/Applications/CS230 Data/Rotated Records/%s.npy"%pad(i,5)):
        rsns.append(i)
        indices.append(indicesI[k])
        count+=1
    k+=1
    if count==max:
        break

#Load rock paths for each record
input=[]
with open("/Applications/CS230 Data/Export/paths1h.csv","w") as o:
    with open(pathPath,"r") as f:
        data=f.read().split("\n")
        for i in indices:
            o.write(data[i])
            o.write("\n")
            input.append(data[i])

#Load event and station metadata for each record
with open("/Applications/CS230 Data/Export/metadata.dsv","w") as o:
    metadata=[]
    with open(metadataPath,"r") as f:
        mdata={}
        data=f.read().split("\n")
        for d in data:
            mdata[d.split("||")[0]]=d
        k=0
        for i in rsns:
            o.write(mdata[str(float(i))].split("||",1)[-1])
            o.write("\n")
            input[k]+=","+mdata[str(float(i))].replace(",",".").replace("||",",").split(",",1)[-1]
            k+=1

#Expand inputs for each rotated angle
with open("/Applications/CS230 Data/Export/inputExpanded.csv","w") as o2:
    with open("/Applications/CS230 Data/Export/input.csv","w") as o:
        for i in input:
            o.write(i)
            o.write("\n")
            for j in range(angles):
                o2.write(i)
                o2.write(",%s"%j)
                o2.write("\n")
with open("/Applications/CS230 Data/Export/timeSeriesDTsExpanded.csv","w") as o2:
    with open("/Applications/CS230 Data/Export/timeSeriesDTs.csv","w") as o:
        with open(tsmdPath,"r") as f:
            tsMetadata=json.loads(f.read())
            for i in rsns:
                text="%s\n"%tsMetadata[str(pad(i,5))]["dt"][0]
                o.write(text)
                for j in range(angles):
                    o2.write(text)

#Concatenate all rotated records into one array
ri=0
for i in range(len(rsns)):
    records=np.load("/Applications/CS230 Data/Rotated Records/%s.npy"%pad(rsns[i],5))
    na=records.shape[0]
    if i==0:
        output=np.zeros((na*len(rsns),records.shape[1]))
    output[ri:ri+na,:]=records
    ri+=na

#Save Y and copy over useful files
np.save("/Applications/CS230 Data/Export/output.npy",output)
shutil.copy("/Applications/CS230 Data/metadataHeaders.csv","/Applications/CS230 Data/Export/metadataHeaders.csv")
shutil.copy("/Applications/CS230 Data/rocks.csv","/Applications/CS230 Data/Export/rocks.csv")