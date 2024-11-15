# (c) 2024 Patrick Nieman
# Predicts SDOF oscillator response and response spectra per the Newmark average acceleration method

import math
import numpy as np

#Response spectrum for uniform period spacing
def responseSpectrum(a,dt,dT,Tmax,xi):
    amax=np.zeros((a.shape[0],int(round(Tmax/dT)+1)))
    amax[:,0]=np.max(np.abs(a),axis=1)
    t=dT
    k=0
    while t<=Tmax+1e-7:
        accel=responseAvg(t,xi,dt,0,0,a)
        amax[:,k]=np.max(np.abs(accel),axis=1)
        t+=dT
        k+=1
    return np.arange(0,Tmax+dT,dT),amax

#Response spectrum for uniform period spacing
def responseSpectrumR(a,dt,Tr,xi):
    amax=np.zeros((a.shape[0],len(Tr)+1))
    amax[:,0]=np.max(np.abs(a),axis=1)
    k=1
    for t in Tr:
        accel=responseAvg(t,xi,dt,0,0,a)
        amax[:,k]=np.max(np.abs(accel),axis=1)
        k+=1
    return Tr,amax

#SDOF oscillator response, Newmark average acceleration method
def responseAvg(t,xi,dt,u0,v0,p):
    m=1
    wn=2*math.pi/t
    k=m*wn**2
    c=2*xi*wn*m
    u=np.zeros(p.shape)
    v=np.zeros(p.shape)
    a=np.zeros(p.shape)
    u[:,0]=u0
    v[:,0]=v0
    a[:,0]=(p[:,0]-c*v[:,0]-k*u[:,0])/m

    i=0
    while i<p.shape[1]-1:
        t1 = p[:,i+1]
        t2 = (m/dt**2) * (4*u[:,i] + 4*v[:,i]*dt + a[:,i]*dt**2)
        t3 = (c/dt) * (2*u[:,i] + v[:,i]*dt)
        t4 = 4*m/dt**2 + 2*c/dt + k

        u[:,i+1]=(t1+t2+t3)/t4
        v[:,i+1]=-v[:,i] + (2/dt) * (u[:,i+1]-u[:,i])
        a[:,i+1]=(4/dt**2) * (u[:,i+1] - u[:,i] - v[:,i]*dt - dt**2*a[:,i]/4)
        i+=1
    return u*(wn**2)