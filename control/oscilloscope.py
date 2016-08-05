# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:10:07 2016

@author: aurelien.barbotin
"""

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time
import nidaqmx
pmt_sensitivity_channel= 'Dev1/ao3'

class Oscilloscope(QtGui.QWidget):
    """Class defining an oscilloscope to monitor signal from a nidaq card"""
    def __init__(self):
        super().__init__()
        self.plotLayout = pg.GraphicsLayoutWidget()

        
        self.plot = self.plotLayout.addPlot(row=0, col=0)

        self.wave = self.plot.plot()
        
        self.nidaq = nidaqmx.Device("Dev1")
        self.channels = self.nidaq.get_analog_input_channels()
        
        self.plot.setXRange(0, 100)
        self.plot.setYRange(-0.5, 5.0)
        self.button=QtGui.QPushButton("Start")
        self.button.clicked.connect(self.start)
        self.isRunning=False
        
        self.button.clicked.connect(self.getData)        
        
        self.ai_channels = QtGui.QComboBox()
        self.ai_channels.addItems(self.channels)
        self.ai_channels.currentIndexChanged.connect(lambda: self.changeChannel(self.ai_channels.currentText()) )
        self.ailabel=QtGui.QLabel("available channels")
        
        #Communication with the ni card        
        
        #get the input 
        self.aitask = nidaqmx.AnalogInputTask()
        self.aitask.create_voltage_channel('Dev1/ai0', terminal = 'rse', min_val=-1, max_val=10.0)
        self.aitask.configure_timing_sample_clock(rate = 10000.0)
        
        layout= QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.plotLayout,0,0,7,5)
        layout.addWidget(self.button,3,5,1,1)
        layout.addWidget(self.ailabel,4,5,1,1)
        layout.addWidget(self.ai_channels,5,5,1,1)
        layout.setRowMinimumHeight(6,150)
        
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getData)

        
    def changeChannel(self,channel):
        if(self.isRunning):
            self.timer.stop()
            self.aitask.stop()
            del self.aitask
        
        
        self.aitask = nidaqmx.AnalogInputTask()
        self.aitask.create_voltage_channel(channel, terminal = 'rse', min_val=-1, max_val=10.0)
#        self.aitask.configure_timing_sample_clock(rate = 100000.0)
        if(self.isRunning):
            self.aitask.start()
            self.timer.start()
        
    def getData(self):
        if(self.isRunning):
            data = self.aitask.read()[:,0]
            self.wave.setData(data)

    def updateSensitivity(self,val):
        self.sensitivity = val / 100
        self.aotask.write(self.sensitivity,auto_start=True)
        
    def start(self):
        if(self.isRunning):
            #Stop everything
            self.isRunning = False
            self.timer.stop()
            self.aitask.stop()
            self.aitask.wait_until_done()
            self.aotask.stop()
            self.button.setText("Start")
            
        else:
            #Starts everything
            self.isRunning= True
            self.aotask.start()
            self.aotask.write(self.sensitivity,auto_start=True)   
            self.aitask.start()
            self.timer.start(10)
            self.button.setText("Stop")
    
    def closeEvent(self,*args,**kwargs):
        super().closeEvent(*args,**kwargs)
        self.timer.stop()
        self.nidaq.reset()
        
if __name__=="__main__":
    app = QtGui.QApplication([])
    menu= Oscilloscope()
    menu.show()
    app.exec_()