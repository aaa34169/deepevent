# coding: utf-8
import logging

import os
import argparse
from keras.models import model_from_json
from pyBTK import btk
import numpy as np

import deepevent
from deepevent import core

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',help='* input c3d file',type=str,required=True)
    parser.add_argument('-o','--output',help=' output c3d file with events',type=str)
    args = parser.parse_args()

    json_file = open(deepevent.DATA_PATH+'DeepEventModel.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(deepevent.DATA_PATH+"DeepEventWeight.h5")

    filenameIn = args.input
    if args.output is not None:
        filenameOut = args.output
    else:
        filenameOut = args.input
        logging.warning("[deepevent] input will be overwritten")


    acq0 = read(filenameIn)
    acq0.ClearEvents()

    acqF = btk.btkAcquisition.Clone(acq0)
    pfn = acqF.GetPointFrameNumber()
    freq = acqF.GetPointFrequency()
    ff = acqF.GetFirstFrame()

    md = acq0.GetMetaData()
    SubjectInfo = md.FindChild("SUBJECTS").value().FindChild("NAMES").value().GetInfo()
    SubjectValue = SubjectInfo.ToString()

    markers = ["LANK","RANK","LTOE","RTOE","LHEE","RHEE"]

    globalFrame,forwardProgression = core.progressionframe(acq0)
    for marker in markers:
        core.applyRotation(acq0,marker,globalFrame,forwardProgression)


    eventLFS,eventRFS,eventLFO,eventRFO = core.predict(model,acq0,markers,pfn,freq)

    for ind_indice in range(eventLFS.shape[0]):
        newEvent=btk.btkEvent()
        newEvent.SetLabel("Foot Strike")
        newEvent.SetContext("Left")
        newEvent.SetTime((ff-1)/freq + float(eventLFS[ind_indice]/freq))
        newEvent.SetSubject(SubjectValue[0])
        newEvent.SetId(1)
        acqF.AppendEvent(newEvent)

    for ind_indice in range(eventRFS.shape[0]):
        newEvent=btk.btkEvent()
        newEvent.SetLabel("Foot Strike")
        newEvent.SetContext("Right")
        newEvent.SetTime((ff-1)/freq + float(eventRFS[ind_indice]/freq))
        newEvent.SetSubject(SubjectValue[0])
        newEvent.SetId(1)
        acqF.AppendEvent(newEvent)

    for ind_indice in range(eventLFO.shape[0]):
        newEvent=btk.btkEvent()
        newEvent.SetLabel("Foot Off")
        newEvent.SetContext("Left") #
        newEvent.SetTime((ff-1)/freq + float(eventLFO[ind_indice]/freq))
        newEvent.SetSubject(SubjectValue[0])
        newEvent.SetId(2)
        acqF.AppendEvent(newEvent)

    for ind_indice in range(eventRFO.shape[0]):
        newEvent=btk.btkEvent()
        newEvent.SetLabel("Foot Off")
        newEvent.SetContext("Right") #
        newEvent.SetTime((ff-1)/freq + float(eventRFO[ind_indice]/freq))
        newEvent.SetSubject(SubjectValue[0])
        newEvent.SetId(2)
        acqF.AppendEvent(newEvent)

    save(acqF,filenameOut)

if __name__ == "__main__":

    main()
