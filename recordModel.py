# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a linear model to predict acceleration time histories from earthquake metadata

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from loss import *
import architectures

tf.compat.v1.enable_eager_execution()
xpath="/Applications/CS230 Data/Export/inputExpanded.csv"
ypath="/Applications/CS230 Data/Export/output.npy"

testSplit=0.05

#Load X and Y
x=np.loadtxt(xpath,delimiter=",")
y=np.load(ypath)

#Normalize except for modified one-hot rock path classification
metaStart=74
x[:,metaStart:]=x[:,metaStart:]-np.mean(x[:,metaStart:],axis=0)
x[:,metaStart:]=x[:,metaStart:]/np.std(x[:,metaStart:],axis=0)

#Shuffle and split data
m=x.shape[0]
shuffle=np.arange(m)
np.random.shuffle(shuffle)
x=x[shuffle,:]
y=y[shuffle,:]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[1:index,:]
yTrain=y[1:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

#Instantiate the model
model=architectures.linear()
#model=architectures.linearConv()

#Complile model
model.compile(optimizer="adam",loss=responseLoss,metrics=[spectrum,arias,motion,smearedMotion,recordPeak])

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/Checkpoints/modelConv-{epoch:02d}-{val_loss:.5f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=3, batch_size=32, validation_split=testSplit,callbacks=[checkpoint])
model.evaluate(xTest,yTest)

#Save model
model.save("/Applications/CS230 Data/model.keras")

p.plot(history.history['loss'])
p.show()