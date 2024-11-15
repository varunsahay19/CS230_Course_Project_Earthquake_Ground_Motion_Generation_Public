# (c) 2024 Patrick Nieman and Varun Sahay
# Processes event and station metadata for records from the NGA-West2 database flatfile

import numpy as np
import os

#parameters for filtering
columnQualityThreshold=0.96
rn=8169
rowQualityThreshold=1


sourceFolder="Sources"
outputFolder="Rock Query"
finalDataFolder="/Applications/CS230 Data/"
rows=[]

#Utility
def getF(i):
    try:
        return float(i)
    except:
        return i

#Read in flatfile
with open(os.path.join(sourceFolder,"nga050.csv"),"r") as f:
    rows=f.read().replace("\n"," ").split("$&$")
    headers=rows[0].split("\t")
    nc=len(headers)
    rows=rows[1:-1]
    nr=len(rows)
    for i in range(nr):
        rows[i]=[getF(a) for a in rows[i].split("\t")]
headers=np.array(headers)
data=np.array(rows)
data=data[:rn,:]
print(data.shape)


#Determine the metadata columns that most records have data for
scoresC=np.sum(data!='-999.0',axis=0)/rn
goodCols=np.nonzero(scoresC>columnQualityThreshold)
headersOld=headers[goodCols]
dataReduced=np.squeeze(data[:,goodCols])
nc=dataReduced.shape[1]

#Determine records for which most metadata are avaliable
scoresR=np.sum(dataReduced!='-999.0',axis=1)/nc
goodRows=np.nonzero(scoresR>=rowQualityThreshold)
dataReduced=np.squeeze(dataReduced[goodRows,:])
print(dataReduced.shape)

#Filter most relevant columns (manually selected) from filtered metadata
selected=np.array([0,7,11,12,16,18,19,20,21,23,24,25,26,29,30,31,32,34,188])
goodCols=np.array(goodCols)[:,selected]
dataReducedNew=np.squeeze(data[:,goodCols])
dataReducedNew=np.squeeze(dataReducedNew[goodRows,:])
headersNew=headers[goodCols]
print(dataReducedNew.shape)

#Save metadata and extract coordinates, which are used for rock path computation but not for model training
hypLatitudes=np.nonzero(headersOld=="Hypocenter Latitude (deg)")
hypLongitudes=np.nonzero(headersOld=="Hypocenter Longitude (deg)")
staLatitudes=np.nonzero(headersOld=="Station Latitude")
staLongitudes=np.nonzero(headersOld=="Station Longitude")
magnitude=np.nonzero(headersOld=="Earthquake Magnitude")
coordinates=np.squeeze(dataReduced[:,np.array([hypLatitudes,hypLongitudes,staLatitudes,staLongitudes])])
np.savetxt(os.path.join(outputFolder,"coordinates.csv"),coordinates,fmt="%s",delimiter=",")
np.savetxt(os.path.join(finalDataFolder,"metadata.csv"),dataReducedNew,fmt="%s",delimiter="||")
np.savetxt(os.path.join(finalDataFolder,"metadataExpanded.csv"),data,fmt="%s",delimiter="||")
np.savetxt(os.path.join(finalDataFolder,"metadataHeaders.csv"),headersNew.T,fmt="%s")

#Write request IDs in batches of 100 in descending magnitude order
dataSorted=np.squeeze(dataReducedNew[dataReducedNew[:, int(np.squeeze(magnitude))].argsort()])
with open(os.path.join(outputFolder,"requests.csv"),"w") as f:
    i=0
    for rsn in dataSorted[:,0]:
        f.write("%s"%int(float(rsn)))
        if i%100==0 and i>0:
            f.write("\n")
        else:
            f.write(",")
        i+=1
