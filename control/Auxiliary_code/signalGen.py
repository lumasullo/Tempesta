# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:11:27 2016

@author: Andreas
"""


from control import libnidaqmx
import numpy as np
import time
from numpy import arange
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
from lantz import Q_
import matplotlib.pyplot as plt




class Pulse(QtCore.QObject):
    """Small class for Pulse object, has a starting time and a width. With the though of
    further implementing other types of pulses or ramps, sines etc. Maybe not really necessary"""

    def __init__(self, start, width, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.start = start
        self.width = width
        
        
   
        
class Signal(QtCore.QObject):
    """Class signal has an array defining values in each sample point. When initiated array is set to []. 
    Fcn reset() resets array to [], setLength sets the array to zeros with length nSamples and fcn 
    addPulse sets values within defined area to 1.""" 
    def __init__(self, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.length = None
        self.array = []
    
    def reset(self):
        self.array = []
        
    def setLength(self, nSamples):
        self.array = np.zeros(nSamples)
    
    def addPulse(self, pulse):
        self.array[range(pulse.start, min(pulse.start+pulse.width, np.size(self.array)))] = 1  #Range excludes last argument. But will be corrent since 0 indexed
            
               
        
class SignalGenerator(QtCore.QObject):
    """Class signal generator defines an object that contains the signals that are then to be generated. 
    Also initiates the devive and creates necessary channels and tasks. Fcn Generate() concatenates the signals
    and writes to task. Then the task is started. Fcn Stop() stops the task. """

    def __init__(self, device, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)

        self.TimeBase = r'100kHzTimeBase'
        self.nSamples = None
        self.nidaq = device
        self.nidaq.reset()
        
        self.digtask = libnidaqmx.DigitalOutputTask('dtask')
            
        self.digtask.create_channel('Dev1/port0/line0', 'TiSa')
        self.digtask.create_channel('Dev1/port0/line2', '405')
        self.digtask.create_channel('Dev1/port0/line3', '488')
        self.digtask.create_channel('Dev1/port0/line4', 'dchannel4')
            
        
        self.signal355 = Signal()
        self.signal405 = Signal()
        self.signal488 = Signal()
        self.signalCam = Signal()        
        
    def Generate(self):
        signal = np.concatenate((self.signal355.array, self.signal405.array, self.signal488.array, self.signalCam.array))
#        digsignal = np.append(self.digsignal1.array, self.digsignal2.array)

        self.digtask.configure_timing_sample_clock(active_edge = 'rising',source=self.TimeBase,sample_mode = 'continuous',rate = 100000, samples_per_channel = self.nSamples)    
        self.digtask.write(signal, auto_start = False, layout='group_by_channel')
        self.digtask.start()
        
        
        
    def Stop(self):
        self.digtask.stop()
#        self.nidaq.reset()
        
        
      
        
class GraphFrame(pg.GraphicsWindow):
    """Class is child of pg.GraphicsWindow and creats the plot that plots the preview of the pulses. 
    Fcn update() updates the plot of "device" with signal "signal"  """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.plot = self.addPlot(row=1, col=0)
        self.plot.showGrid(x=False, y=False)
        self.plotsig355 = self.plot.plot(pen=pg.mkPen(100, 0, 0))
        self.plotsig405 = self.plot.plot(pen=pg.mkPen(73, 0, 188))
        self.plotsig488 = self.plot.plot(pen=pg.mkPen(0, 247, 255))
        self.plotsigCam = self.plot.plot(pen='w')
        
    def update(self, device, signal):
        
        if device == '355':
            self.plotsig355.setData(signal)
        elif device == '405':
            self.plotsig405.setData(signal)
        elif device == '488':
            self.plotsig488.setData(signal)
        elif device == 'Cam':
            self.plotsigCam.setData(signal)
           

        
class SignalFrame(QtGui.QFrame):
    """ Class SignalFrame contains the QLineEdit objects for one signal. Each pulse has a start and width QLineEdit
    with correstonding QLabels.
    Fcn's valueChanged, checkcalue and checkAllVales make sure graph is updated and that values entered do not 
    exceed the length of the signal."""
    def __init__(self, main, device, name, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.name = QtGui.QLabel(name)
        self.device = device
        self.main = main
        self.main.Cycletimeedit.editingFinished.connect(self.checkAllValues)
        
#        self.S1label = QtGui.QLabel('Start (ms)')
#        self.W1label = QtGui.QLabel('Width (ms)')  
        self.S1edit = QtGui.QLineEdit('20')
        self.S1edit.editingFinished.connect(lambda: self.S1edit.setText(str(self.valueChanged(float(self.S1edit.text())))))
        self.W1edit = QtGui.QLineEdit('10')
        self.W1edit.editingFinished.connect(lambda: self.W1edit.setText(str(self.valueChanged(float(self.S1edit.text()) + float(self.W1edit.text())) - float(self.S1edit.text()))))
#        self.S2label = QtGui.QLabel('Start (ms)')
#        self.W2label = QtGui.QLabel('Width (ms)')
        self.S2edit = QtGui.QLineEdit('40')
        self.S2edit.editingFinished.connect(lambda: self.S2edit.setText(str(self.valueChanged(float(self.S2edit.text())))))
        self.W2edit = QtGui.QLineEdit('10')
        self.W2edit.editingFinished.connect(lambda: self.W2edit.setText(str(self.valueChanged(float(self.S2edit.text()) + float(self.W2edit.text())) - float(self.S2edit.text()))))
#        self.S3label = QtGui.QLabel('Start (ms)')
#        self.W3label = QtGui.QLabel('Width (ms)')  
#        self.S3edit = QtGui.QLineEdit()
#        self.W3edit = QtGui.QLineEdit() 
        
#        self.signalplot = GraphFrame()        
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.setColumnMinimumWidth(0, 70)
        grid.addWidget(self.name, 0, 0)
#        grid.addWidget(self.S1label, 0, 1)
#        grid.addWidget(self.W1label, 0, 2)
        grid.addWidget(self.S1edit, 0, 1)
        grid.addWidget(self.W1edit, 0, 2)
#        grid.addWidget(self.S2label, 0, 3)
#        grid.addWidget(self.W2label, 0, 4)
        grid.addWidget(self.S2edit, 0, 3)
        grid.addWidget(self.W2edit, 0, 4)
#        grid.addWidget(self.S3label, 0, 5)
#        grid.addWidget(self.W3label, 0, 6)
#        grid.addWidget(self.S3edit, 1, 5)
#        grid.addWidget(self.W3edit, 1, 6)
#        grid.addWidget(self.signalplot, 1, 0, 1, 5)
        
    
    def valueChanged(self, value):
        self.main.Updatesignal(self.device)

        return self.CheckValue(value)
            
        
        
    
    def CheckValue(self, value):
        if value == '':
            value = 0
        value = float(value)
        max_value = float(self.main.Cycletimeedit.text())
        if value > max_value:
            return max_value
        else:
            return value
            
    def checkAllValues(self):
        self.S1edit.setText(str(self.valueChanged(float(self.S1edit.text()))))
        self.W1edit.setText(str(self.valueChanged(float(self.S1edit.text()) + float(self.W1edit.text())) - float(self.S1edit.text())))
        self.S2edit.setText(str(self.valueChanged(value = float(self.S2edit.text()))))
        self.W2edit.setText(str(self.valueChanged(float(self.S2edit.text()) + float(self.W2edit.text())) - float(self.S2edit.text())))
                
             


      
class SigGenWidget(QtGui.QFrame):
    """ Class defines the parent widget. Containt SignalFrames for four signals and the GraphFrame. """
    def __init__(self, device, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
#        data = np.ones(1000)
#        data[range(0, 1000)] = 1
#        #data[range(501, 1000)] = 0.2
#        task = libnidaqmx.AnalogOutputTask()
#        task.create_voltage_channel('Dev1/ao0', min_val=-10.0, max_val=10.0)
#        task.configure_timing_sample_clock(rate = 1000.0)
#        task.write(data)                
        
        self.running = False
        self.generator = SignalGenerator(device)
        self.Cycletimelabel = QtGui.QLabel('Cycletime (ms): ')
        self.Cycletimeedit = QtGui.QLineEdit('100')
        
        self.pulse1 = QtGui.QLabel('<h3>Pulse 1<h3>')
        self.pulse2 = QtGui.QLabel('<h3>Pulse 2<h3>')        
        self.pulsegraph = GraphFrame()   
#        self.sig1frame = SignalFrame('X')
#        self.sig2frame = SignalFrame('Y') 
#        self.sig3frame = SignalFrame('Z') 
        self.sig4frame = SignalFrame(self, '355', 'TTL TiSa') 
        self.sig5frame = SignalFrame(self, '405', 'TTL 405')
        self.sig6frame = SignalFrame(self, '488', 'TTL 488')
        self.sig7frame = SignalFrame(self, 'Cam', 'Camera')        
        self.genButton = QtGui.QPushButton('Generate')
        self.genButton.clicked.connect(self.StartStop)
        
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.setSpacing(0)
        grid.setColumnMinimumWidth(0, 90)
        grid.setColumnMinimumWidth(1, 70)
        grid.setColumnMinimumWidth(2, 70)
        grid.setColumnMinimumWidth(3, 70)
        grid.setColumnMinimumWidth(4, 70)
        grid.addWidget(self.Cycletimelabel, 0, 1)
        grid.addWidget(self.Cycletimeedit, 0, 2)
        
        grid.addWidget(self.pulse1, 1, 2)
        grid.addWidget(self.pulse2, 1, 3)
        grid.addWidget(QtGui.QLabel('Start 1 (ms) '), 2, 1)
        grid.addWidget(QtGui.QLabel('Width 1 (ms) '), 2, 2)
        grid.addWidget(QtGui.QLabel('Start 2 (ms) '), 2, 3)
        grid.addWidget(QtGui.QLabel('Width 2 (ms) '), 2, 4)
#        grid.addWidget(self.sig1frame, 0, 1, 1, 4)
#        grid.addWidget(self.sig2frame, 1, 1, 1, 4)
#        grid.addWidget(self.sig3frame, 2, 1, 1, 4)
        grid.addWidget(self.sig4frame, 3, 0, 1, 5)
        grid.addWidget(self.sig5frame, 4, 0, 1, 5)
        grid.addWidget(self.sig6frame, 6, 0, 1, 5)
        grid.addWidget(self.sig7frame, 7, 0, 1, 5)
        grid.addWidget(self.pulsegraph, 8, 0, 1, 5)
        grid.setRowMinimumHeight(8, 100)
        grid.addWidget(self.genButton, 9, 3)  
        
        self.Updatesignal('355')
        self.Updatesignal('405')
        self.Updatesignal('488')
        self.Updatesignal('Cam')
        
        
    def StartStop(self):

        if self.running:
            self.genButton.setText('Generate')
            self.generator.Stop()
            self.running = False
            
            
        else:
            self.genButton.setText('STOP')
            self.generator.Generate()
            self.running = True
            
        
        
    def Updatesignal(self, device):

        nSamples = int(100000 * float(self.Cycletimeedit.text()) / 1000)
        
        if device == '355':
            sig4p1 = Pulse(int(float(self.sig4frame.S1edit.text()) * 100), int(float(self.sig4frame.W1edit.text()) * 100))
            sig4p2 = Pulse(int(float(self.sig4frame.S2edit.text()) * 100), int(float(self.sig4frame.W2edit.text()) * 100))
            self.generator.signal355.reset()
            self.generator.signal355.setLength(nSamples)
            self.generator.signal355.addPulse(sig4p1)
            self.generator.signal355.addPulse(sig4p2)
            self.pulsegraph.update(device, self.generator.signal355.array)
            
        elif device == '405':
            sig5p1 = Pulse(int(float(self.sig5frame.S1edit.text()) * 100), int(float(self.sig5frame.W1edit.text()) * 100))
            sig5p2 = Pulse(int(float(self.sig5frame.S2edit.text()) * 100), int(float(self.sig5frame.W2edit.text()) * 100))
            self.generator.signal405.reset()
            self.generator.signal405.setLength(nSamples)
            self.generator.signal405.addPulse(sig5p1)        
            self.generator.signal405.addPulse(sig5p2)
            self.pulsegraph.update(device, self.generator.signal405.array)
            
        elif device == '488':
            sig6p1 = Pulse(int(float(self.sig6frame.S1edit.text()) * 100), int(float(self.sig6frame.W1edit.text()) * 100))
            sig6p2 = Pulse(int(float(self.sig6frame.S2edit.text()) * 100), int(float(self.sig6frame.W2edit.text()) * 100))
            self.generator.signal488.reset()
            self.generator.signal488.setLength(nSamples)
            self.generator.signal488.addPulse(sig6p1)  
            self.generator.signal488.addPulse(sig6p2)
            self.pulsegraph.update(device, self.generator.signal488.array)
        elif device == 'Cam':
            sig7p1 = Pulse(int(float(self.sig7frame.S1edit.text()) * 100), int(float(self.sig7frame.W1edit.text()) * 100))
            sig7p2 = Pulse(int(float(self.sig7frame.S2edit.text()) * 100), int(float(self.sig7frame.W2edit.text()) * 100))
            self.generator.signalCam.reset()
            self.generator.signalCam.setLength(nSamples) 
            self.generator.signalCam.addPulse(sig7p1)  
            self.generator.signalCam.addPulse(sig7p2)
            self.pulsegraph.update(device, self.generator.signalCam.array)

        self.generator.nSamples = nSamples
        
    

        
        
        
        

        
               
          
         

        
        
          
  


        
        
        
        
        
        
        
        
        
        
        
        