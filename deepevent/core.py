# coding: utf-8
import os
import argparse
from pyBTK import btk
from scipy import signal
from scipy.signal import argrelextrema
import numpy as np
from numpy import matlib as mb
import logging

def filter(acq,marker,fc):
	# Butterworth filter
	b, a = signal.butter(4, fc/(acq.GetPointFrequency()/2))
	Mean = np.mean(marker,axis=0)
	Minput = marker - mb.repmat(Mean,acq.GetPointFrameNumber(),1)
	Minput = signal.filtfilt(b,a,Minput,axis=0)
	Moutput = Minput + np.matlib.repmat(Mean,acq.GetPointFrameNumber(),1)

	return Moutput

def derive_centre(marker,pfn,freq):
	# Compute velocity

	marker_der = (marker[2:pfn,:] - marker[0:(pfn-2),:]) / (2 / freq)
	marker_der = np.concatenate(([[0,0,0]],marker_der,[[0,0,0]]),axis=0)
	return marker_der

def progressionframe(acq,marker="LANK"):
	__threshold = 800
	values = acq.GetPoint(marker).GetValues()

	MaxValues =[values[-1,0]-values[0,0], values[-1,1]-values[0,1]]
	absMaxValues =[np.abs(values[-1,0]-values[0,0]), np.abs(values[-1,1]-values[0,1])]

	ind = np.argmax(absMaxValues)

	if absMaxValues[ind] > __threshold:

		diff = MaxValues[ind]

		if ind ==0 :
			progressionAxis = "X"
			lateralAxis = "Y"
		else:
			progressionAxis = "Y"
			lateralAxis = "X"

		forwardProgression = True if diff>0 else False

		globalFrame = (progressionAxis+lateralAxis+"Z")
	else:
		raise Exception("[deepEvent] progression axis not detected - distance of the marker %s inferior to 800mm"%(marker))

	return globalFrame,forwardProgression


def applyRotation(acq,marker,globalFrameOrientation,forwardProgression):

	if globalFrameOrientation == "XYZ":
		rot = np.array([[1,0,0],[0,1,0],[0,0,1]])
	elif globalFrameOrientation == "YXZ":
		rot = np.array([[0,1,0],[-1,0,0],[0,0,1]])
	else:
		raise Exception("[deepEvent] code cannot work with Z as non-normal axis")

	values = acq.GetPoint(marker).GetValues()

	valuesRot = np.zeros((acq.GetPointFrameNumber(),3))
	for i in range (0, acq.GetPointFrameNumber()):
		valuesRot[i,:]= np.dot(rot,values[i,:])
		if not forwardProgression:
			valuesRot[i,:] = np.dot(np.array([[-1,0,0],[0,1,0],[0,0,1]]),valuesRot[i,:])

	acq.GetPoint(marker).SetValues(valuesRot)

	return acq

def save(acq, filename):
    #Write the FilenameOut.c3d
	writer = btk.btkAcquisitionFileWriter()
	writer.SetInput(acq)
	writer.SetFilename(filename)
	writer.Update()

def read(filename):
	reader = btk.btkAcquisitionFileReader()
	reader.SetFilename(filename)
	reader.Update()
	acq = reader.GetOutput()

	return acq

def predict(load_model,acq,markers,pfn,freq):

	nframes = 1536
	nb_data_in = 36 #6 markers x 3

	inputs = np.zeros((1,nframes,nb_data_in))
	for k in range(6):
		values = acq.GetPoint(markers[k]).GetValues()
		inputs[0,0:pfn,k*3: (k + 1)*3] = filter(acq,values,6)
		inputs[0,0:pfn,3 * len(markers) + k*3:3 * len(markers) +  (k + 1)*3] = derive_centre(inputs[0,:,k * 3:(k+1)*3],pfn,freq)

	# Prediction with the model
	predicted = load_model.predict(inputs) #shape[1,nb_frames,5] 0: no event, 1: Left Foot Strike, 2: Right Foot Strike, 3:Left Toe Off, 4: Right Toe Off

	#Threshold to set the gait events
	predicted_seuil = predicted
	for j in range(nframes):
		if predicted[0,j,1] <= 0.01:
			predicted_seuil[0,j,1] = 0
		if predicted[0,j,2] <= 0.01:
			predicted_seuil[0,j,2] = 0
		if predicted[0,j,3] <= 0.01:
			predicted_seuil[0,j,3] = 0
		if predicted[0,j,4] <= 0.01:
			predicted_seuil[0,j,4] = 0

	predicted_seuil_max = np.zeros((1,nframes,5))
	for j in range(1,5):
		predicted_seuil_max[0,argrelextrema(predicted_seuil[0,:,j],np.greater)[0],j] = 1

	for j in range(nframes):
		if np.sum(predicted_seuil_max[0,j,:]) == 0: predicted_seuil_max[0,j,0] = 1

	eventLFS = np.argwhere(predicted_seuil_max[0,:,1])
	eventRFS = np.argwhere(predicted_seuil_max[0,:,2])
	eventLFO = np.argwhere(predicted_seuil_max[0,:,3])
	eventRFO = np.argwhere(predicted_seuil_max[0,:,4])

	return eventLFS,eventRFS,eventLFO,eventRFO
