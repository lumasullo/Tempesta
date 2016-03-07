# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:11:27 2016

@author: Andreas
"""


from control import libnidaqmx
#import libnidaqmx
import numpy as np
import time
from numpy import arange
from PyQt4 import QtGui, QtCore
from lantz import Q_
import matplotlib.pyplot as plt

class Pulse(QtCore.QObject):
    
    def __init__(self, start, width, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.start = start
        self.width = width
        
        
class Signal(QtCore.QObject):
        
    def __init__(self, length, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.length = length
        self.array = np.zeros(length)
        self.pulses = None
        
    def addPulse(self, pulse):
        self.array[range(pulse.start, pulse.start+pulse.width)] = 1
            
        
class SignalGenerator(QtCore.QObject):

    def __init__(self, nSamples, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
    
        self.nSamples = nSamples
    
        self.nidaq = libnidaqmx.Device('Dev1')
        self.nidaq.reset()
        
        self.digtask = libnidaqmx.DigitalOutputTask('dtask')
        self.anatask = libnidaqmx.AnalogOutputTask('atask')
            
        self.digtask.create_channel('Dev1/port0/line0', 'dchannel1')
        self.digtask.create_channel('Dev1/port0/line1', 'dchannel2')
#        self.digtask.create_channel('Dev1/port0/line2', 'dchannel3')
#        self.digtask.create_channel('Dev1/port0/line3', 'dchannel4')
        
        self.anatask.create_voltage_channel('Dev1/ao0', 'achannel0')
        
        self.anatask.configure_timing_sample_clock(rate=100000,sample_mode='continuous', samples_per_channel=self.nSamples)
        self.digtask.configure_timing_sample_clock(active_edge = 'rising',source=r'100kHzTimeBase',sample_mode = 'continuous',rate = 100000, samples_per_channel = self.nSamples)

        
        self.anasignal1 = Signal(nSamples)
        self.anasignal2 = Signal(nSamples)   
        self.anasignal3 = Signal(nSamples)   
        self.anasignal4 = Signal(nSamples)   
        
        self.digsignal1 = Signal(nSamples)
        self.digsignal2 = Signal(nSamples)
#        self.digsignal3 = Signal(nSamples)
#        self.digsignal4 = Signal(nSamples)
        
    def Generate(self):

#        digsignal = np.append(self.digsignal1.array, np.append(self.digsignal2.array, np.append(self.digsignal3.array, self.digsignal4.array)))
        digsignal = np.append(self.digsignal1.array, self.digsignal2.array)
#        digsignal= self.digsignal1.array        
        plt.plot(digsignal)        
        self.digtask.write(digsignal, auto_start = False, layout='group_by_channel')
        self.digtask.start()
        
        
        
    def Stop(self):
        self.digtask.stop()
        self.nidaq.reset()
        
        
class SignalFrame(QtGui.QFrame):
     
    def __init__(self, name, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.name = QtGui.QLabel(name)
        
#        self.S1value = None
#        self.W1value = None        
#        self.S2value = None        
#        self.W2value = None        
#        self.S3value = None
#        self.W3value = None
        
        self.S1label = QtGui.QLabel('Start (ms)')
        self.W1label = QtGui.QLabel('Width (ms)')  
        self.S1edit = QtGui.QLineEdit()
        self.W1edit = QtGui.QLineEdit()
        self.S2label = QtGui.QLabel('Start (ms)')
        self.W2label = QtGui.QLabel('Width (ms)')
        self.S2edit = QtGui.QLineEdit()
        self.W2edit = QtGui.QLineEdit()
        self.S3label = QtGui.QLabel('Start (ms)')
        self.W3label = QtGui.QLabel('Width (ms)')  
        self.S3edit = QtGui.QLineEdit()
        self.W3edit = QtGui.QLineEdit() 
        
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 1, 0)
        grid.addWidget(self.S1label, 0, 1)
        grid.addWidget(self.W1label, 0, 2)
        grid.addWidget(self.S1edit, 1, 1)
        grid.addWidget(self.W1edit, 1, 2)
        grid.addWidget(self.S2label, 0, 3)
        grid.addWidget(self.W2label, 0, 4)
        grid.addWidget(self.S2edit, 1, 3)
        grid.addWidget(self.W2edit, 1, 4)
#        grid.addWidget(self.S3label, 0, 5)
#        grid.addWidget(self.W3label, 0, 6)
#        grid.addWidget(self.S3edit, 1, 5)
#        grid.addWidget(self.W3edit, 1, 6)
      
class SigGenWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.running = False
        
        self.Cycletimelabel = QtGui.QLabel('Cycletime (ms): ')
        self.Cycletimeedit = QtGui.QLineEdit('100')
        
        self.pulse1 = QtGui.QLabel('<h3>Pulse 1<h3>')
        self.pulse2 = QtGui.QLabel('<h3>Pulse 2<h3>')        
        
#        self.sig1frame = SignalFrame('X')
#        self.sig2frame = SignalFrame('Y') 
#        self.sig3frame = SignalFrame('Z') 
        self.sig4frame = SignalFrame('TTL 355') 
        self.sig5frame = SignalFrame('TTL 405')
        self.sig6frame = SignalFrame('TTL 488')
        self.sig7frame = SignalFrame('Camera')        
        self.genButton = QtGui.QPushButton('Generate')
        self.genButton.clicked.connect(self.StartStop)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.setSpacing(10)
    
        grid.addWidget(self.Cycletimelabel, 0, 1)
        grid.addWidget(self.Cycletimeedit, 0, 2)
        
        grid.addWidget(self.pulse1, 1, 1)
        grid.addWidget(self.pulse2, 1, 3)
#        grid.addWidget(self.sig1frame, 0, 1, 1, 4)
#        grid.addWidget(self.sig2frame, 1, 1, 1, 4)
#        grid.addWidget(self.sig3frame, 2, 1, 1, 4)
        grid.addWidget(self.sig4frame, 3, 0, 1, 4)
        grid.addWidget(self.sig5frame, 4, 0, 1, 4)
        grid.addWidget(self.sig6frame, 6, 0, 1, 4)
        grid.addWidget(self.sig7frame, 7, 0, 1, 4)
        grid.addWidget(self.genButton, 8, 3)  
        
        
    def StartStop(self):

        if self.running:
            self.genButton.setText('Generate')
            self.generator.Stop()
            del self.generator
            self.running = False
            
            
        else:
            self.genButton.setText('STOP')
            self.Gen()
            self.running = True
            
        
        
    def Gen(self):

        nSamples = int(100000 * int(self.Cycletimeedit.text()) / 1000)
        self.generator = SignalGenerator(nSamples)

        sig4p1 = Pulse(int(self.sig4frame.S1edit.text()) * 100, int(self.sig4frame.W1edit.text()) * 100)
        sig4p2 = Pulse(int(self.sig4frame.S2edit.text()) * 100, int(self.sig4frame.W2edit.text()) * 100)
        
        sig5p1 = Pulse(int(self.sig5frame.S1edit.text()) * 100, int(self.sig5frame.W1edit.text()) * 100)
        sig5p2 = Pulse(int(self.sig5frame.S2edit.text()) * 100, int(self.sig5frame.W2edit.text()) * 100)
#        
#        sig6p1 = Pulse(int(self.sig6frame.S1edit.text()) * 100, int(self.sig6frame.W1edit.text()) * 100)
#        sig6p2 = Pulse(int(self.sig6frame.S2edit.text()) * 100, int(self.sig6frame.W2edit.text()) * 100)

        self.generator.digsignal1.addPulse(sig4p1)
        self.generator.digsignal1.addPulse(sig4p2)
        
        self.generator.digsignal2.addPulse(sig5p1)
        self.generator.digsignal2.addPulse(sig5p2)
#        
#        generator.digsignal3.addPulse(sig6p1)
#        generator.digsignal3.addPulse(sig6p2)
        
        self.generator.Generate()
        
        
        
        
        
        
        
        
        
        
        
        
        