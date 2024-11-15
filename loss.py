# (c) 2024 Patrick Nieman and Varun Sahay
# Custom loss functions and metrics


import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

#Model to predict response specrtra of predicted records
responsePath="/Applications/CS230 Data/responseModel.keras"
responseModel=tf.keras.models.load_model(responsePath)
responseModel.trainable=False

#Normalized MSE of predicted response spectra
def spectrum(y,yhat):
    ySpectrum=responseModel(y)
    yhatSpectrum=responseModel(yhat)
    return tf.divide(tf.divide(tf.reduce_mean(tf.square(tf.subtract(ySpectrum,yhatSpectrum))),tf.reduce_mean(tf.square(ySpectrum))),0.01)

#Relative difference in Arias intensities
def arias(y,yhat):
    aly=tf.reduce_sum(tf.square(y))
    alyhat=tf.reduce_sum(tf.square(yhat))
    return tf.divide(tf.abs(tf.subtract(aly,alyhat)),aly)

#Direct MSE comparison of records, normalized
def motion(y,yhat):
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(tf.abs(y),tf.abs(yhat)))),tf.reduce_mean(tf.square(y)))

#Direct MSE comparison of records, averaged in 16 bins, normalized, weighted to record start
def smearedMotion(y,yhat):
    ySmeared=tf.reduce_mean(tf.abs(tf.reshape(y,(16,-1))),axis=1)
    ySmeared+=tf.reduce_mean(tf.abs(tf.gather(tf.reshape(y,(16,-1)),indices=[0])),axis=1)
    ySmeared+=tf.reduce_mean(tf.abs(tf.gather(tf.reshape(y,(16,-1)),indices=[0])),axis=1)
    yhatSmeared=tf.reduce_mean(tf.abs(tf.reshape(yhat,(16,-1))),axis=1)
    yhatSmeared+=tf.reduce_mean(tf.abs(tf.gather(tf.reshape(yhat,(16,-1)),indices=[0])),axis=1)
    yhatSmeared+=tf.reduce_mean(tf.abs(tf.gather(tf.reshape(yhat,(16,-1)),indices=[0])),axis=1)
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(ySmeared,yhatSmeared))),tf.reduce_mean(tf.square(y)))

#Time of peak acceleration
def recordPeak(y,yhat):
    return tf.reduce_max(tf.divide(tf.abs(tf.subtract(tf.argmax(tf.abs(y),axis=1),tf.argmax(tf.abs(yhat),axis=1))),300))

#Custom loss model
@register_keras_serializable()
def responseLoss(y,yhat):
    spectrumLoss=spectrum(y,yhat)
    
    ariasLoss=arias(y,yhat)

    motionLoss=motion(y,yhat)

    smearedMotionLoss=smearedMotion(y,yhat)

    recordPeakLoss=recordPeak(y,yhat)
    
    #Equally weight normalized losses
    return tf.divide(tf.reduce_sum([tf.cast(spectrumLoss,tf.float64),tf.cast(ariasLoss,tf.float64),tf.cast(motionLoss,tf.float64),tf.cast(recordPeakLoss,tf.float64),tf.cast(smearedMotionLoss,tf.float64)]),1.0)