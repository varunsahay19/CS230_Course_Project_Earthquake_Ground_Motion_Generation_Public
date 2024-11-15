# (c) 2024 Patrick Nieman and Varun Sahay
# Rotates record pairs to arbitrary series of orientations, interpolates to desired timestep, and pads with zeros to specified length

import os
import json
import numpy as np
import math

metadataPath="/Applications/CS230 Data/timeSeriesMetadata.json"
recordPath="/Applications/CS230 Data/Records"
rotatePath="/Applications/CS230 Data/Rotated Records"

#Load record metadata
with open(metadataPath,"r") as f:
    database=json.loads(f.read())
count=0

#Generate desired angles and padding
nAngles=10 #Number of angles to be rotated to (uniformely spaced)
angles=np.atleast_2d(np.arange(0.0,1.0001*math.pi/2,math.pi/(2*(nAngles-1)))).T
specifiedDuration=150
specifiedDt=0.005
t=np.arange(0,specifiedDuration,specifiedDt)
nt=max(t.shape)

#For each record
for k in database:
    r=database[k]
    d=[0,0]
    dn=[0,0]
    #Read x, y directional data from file
    for i in range(2):
        d[i]=np.atleast_2d(np.loadtxt(os.path.join(recordPath,f'{k}-{r["d"][i]}.csv')))

    #Interpolate to desired timestep (linearly)
    for i in range(2):
        dpi=d[i][:,:min(d[0].shape[1],d[1].shape[1])]
        ti=np.arange(0.0,dpi.shape[1]*r["dt"][i],r["dt"][i])
        if i==0:
            dp1=np.interp(t,ti[:max(dpi.shape)],np.squeeze(dpi),left=0,right=0)
        else:
            dp2=np.interp(t,ti[:max(dpi.shape)],np.squeeze(dpi),left=0,right=0)

    #Rotate records
    dn=np.zeros((2,nt))
    baseAngle=r["d"][0 if r["d"][0]<r["d"][1] else 1]
    dn[0,:]=np.cos(-baseAngle)*dp1+np.sin(-baseAngle)*dp2
    dn[1,:]=np.sin(-baseAngle)*dp1+np.cos(-baseAngle)*dp2
    values=np.dot(np.cos(angles),np.atleast_2d(dn[0,:]))+np.dot(np.sin(angles),np.atleast_2d(dn[1,:]))

    #Save rotated record
    np.save(os.path.join(rotatePath,f'{k}.npy'),np.array(values))
    
    count+=1
    if count%100==0:
        print(count)
