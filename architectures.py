# (c) 2024 Patrick Nieman and Varun Sahay
# Architectures for record model


import tensorflow as tf

def linear():
    #Build sequential model
    model=tf.keras.models.Sequential()
    layers=[128,128,128,192,256,384,512,1024,2048]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(30000,activation="linear"))
    return model

def linearConv():
    model=tf.keras.models.Sequential()
    layers=[128,128,128,192,256,384,512,768,1200]
    for i in layers:
        model.add(tf.keras.layers.Dense(i,activation="relu"))

    #Add deconvolution layers
    model.add(tf.keras.layers.Reshape((-1, 1)))
    model.add(tf.keras.layers.Conv1DTranspose(3,5,strides=5,activation="relu"))
    model.add(tf.keras.layers.Conv1DTranspose(3,5,strides=5,activation="linear"))
    model.add(tf.keras.layers.Conv1DTranspose(1,1,strides=1,activation="linear"))
    model.add(tf.keras.layers.Flatten())
    return model
