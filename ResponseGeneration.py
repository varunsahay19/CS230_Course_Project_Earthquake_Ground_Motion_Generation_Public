# (c) 2024 Patrick Nieman and Varun Sahay
# Calculates response spectra for records, to create training data for the response spectrum model

import numpy as np
import NiemanEQM as eq
import os

kmax=2996
g=386.09 #gravity
xi=0.05 #damping

#Custom period range to emphasize most important spectral shape features, and save for future use
Tr=[0.05,0.15,0.25,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9,1,1.25,1.5,2,2.5,3,5]
path="/Applications/CS230 Data/Rotated Records"
np.save("/Applications/CS230 Data/Export/spectraPeriods1.npy",np.array(Tr))

k=0
ri=0
#For each rotated record
for _,_,files in os.walk(path,topdown=True):
    ne=min(kmax,len(files))
    for file in files:
        #Load records
        records=np.load(os.path.join(path,file))
        na=records.shape[0]

        if k==0: #Initialize array
            input=np.zeros((na*ne,records.shape[1]))
        input[ri:ri+na,:]=records
        
        #Calculate response spectrum explicitly
        _,Sa=eq.responseSpectrumR(records,0.005,Tr,xi)

        if k==0: #Initialize array
            output=np.zeros((na*ne,Sa.shape[1]))
        output[ri:ri+na,:]=Sa

        ri+=na
        k+=1
        print(k)
        if k==kmax:
            break

#Save input and output data for model training
np.save("/Applications/CS230 Data/Export/spectraInput.npy",input)
np.save("/Applications/CS230 Data/Export/spectraOutput.npy",output)