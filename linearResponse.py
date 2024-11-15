# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a 1D convolutional linear network to predict the response spectrum of a record

import numpy as np
import tensorflow as tf

tf.compat.v1.enable_eager_execution()
xpath="/Applications/CS230 Data/Export/spectraInput.npy"
ypath="/Applications/CS230 Data/Export/spectraOutput.npy"

#Load X and Y and expand dimensions
x=np.load(xpath)
y=np.load(ypath)
x=np.expand_dims(x,-1)
y=np.expand_dims(y,-1)

#Shuffle and split data
m=x.shape[0]
testSplit=0.05
shuffle=np.arange(m)
np.random.shuffle(shuffle)
x=x[shuffle,:,:]
y=y[shuffle,:,:]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[0:index,:,:]
yTrain=y[0:index,:,:]
xTest=x[index:,:,:]
yTest=y[index:,:,:]

#Build model
#Add convolutional layers
model=tf.keras.models.Sequential()
for i in range(6):
    model.add(tf.keras.layers.Conv1D(6,7,strides=1,padding="same",activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=2))

#Add fully-connected layers
layers=[256,128,64,32,32,32]
model.add(tf.keras.layers.Flatten())
for i in layers:
    model.add(tf.keras.layers.Dense(i,activation="relu"))
model.add(tf.keras.layers.Dense(yTrain.shape[1],activation="linear"))

#Complile model
model.compile(optimizer="adam",loss='mse',metrics=["mae"])

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/epoch-{epoch:02d}-val_loss-{val_loss:.2f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=11, batch_size=64, validation_split=0.1,callbacks=[checkpoint])
print(model.evaluate(xTest,yTest))

#Save model
model.save('/Applications/CS230 Data/responseModel.keras')
