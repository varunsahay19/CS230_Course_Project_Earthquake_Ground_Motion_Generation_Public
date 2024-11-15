# (c) 2024 Patrick Nieman and Varun Sahay
# Plots spectra and records predicted by linear model

import numpy as np
import NiemanEQM as eq
from matplotlib import pyplot as p
import tensorflow as tf
from random import randrange
from loss import *

#Redefine losses to allow easy loading of keras model
responsePath="/Applications/CS230 Data/responseModel.keras"
responseModel=tf.keras.models.load_model(responsePath)
responseModel.trainable=False

g=386.09
casesEvaluated=800

#Load linear model and test data
Tr=np.load("/Applications/CS230 Data/Export/spectraPeriods.npy")
model=tf.keras.models.load_model('/Applications/CS230 Data/model.keras',custom_objects={"responseLoss": responseLoss,"spectrum":spectrum,"arias":arias,"motion":motion,"smearedMotion":smearedMotion,"recordPeak":recordPeak})
model.trainable=False
yTest=np.load("/Applications/CS230 Data/Export/output.npy")
inputs=np.loadtxt("/Applications/CS230 Data/Export/inputExpanded.csv",dtype=float,delimiter=",")

#Shuffle data and predict time histories
shuffle=np.random.shuffle(np.arange(yTest.shape[0]))
yTest=yTest[shuffle,:]
yTest=yTest[:,:casesEvaluated,:]
inputs=inputs[shuffle,:]
inputs=inputs[:,:casesEvaluated,:]
inputs=np.squeeze(inputs)
yTest=np.squeeze(yTest)
predictions=model.predict(inputs)

#Normalize for plotting 
yTest*=g
predictions*=g

#Predict response spectra directly or with response model
#_,Sa=eq.responseSpectrumR(np.atleast_2d(yTest),0.005,Tr[1:],0.05)
#_,SaHat=eq.responseSpectrumR(np.atleast_2d(predictions),0.005,Tr[1:],0.05)
Sa=responseModel.predict(np.atleast_2d(yTest))
SaHat=responseModel.predict(np.atleast_2d(predictions))

#Take average of response spectra
SaTotal=np.sum(Sa,axis=0)
SaHatTotal=np.sum(SaHat,axis=0)
SaTotal/=(casesEvaluated)
SaHatTotal/=(casesEvaluated)

#Plot response spectrum
p.figure(1)
p.plot(Tr,SaTotal,label="Ground truth")
p.plot(Tr,SaHatTotal,label="Predicted")
p.legend()
p.xlabel("Fundamental period (s)")
p.ylabel("Pseudospectral acceleration (in/s^2)")

#Plot example time series
p.figure(2)
for i in range(8):
    p.subplot(4,4,2*i+1)
    index=randrange(0,casesEvaluated)
    p.plot(yTest[index,:],label="Ground truth")
    p.legend()
    p.subplot(4,4,2*i+2)
    p.plot(predictions[index,:],label="Predicted")

p.show()