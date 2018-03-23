# -*- coding: utf-8 -*-


from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time

import nidaqmx

channel="Dev1/ctr0"
terminal = "PFI0"
class Oscilloscope(QtGui.QWidget):
    """Class defining an oscilloscope to monitor a signal from a nidaq card"""
    def __init__(self):
        super().__init__()
        self.plotLayout = pg.GraphicsLayoutWidget()

        
        self.plot = self.plotLayout.addPlot(row=0, col=0)

        self.wave = self.plot.plot()
        
        
        self.plot.setXRange(0, 100)
        self.plot.setYRange(-0.5, 5.0)
        self.button=QtGui.QPushButton("Start")
        self.button.clicked.connect(self.start)
        self.isRunning=False
        
        self.button.clicked.connect(self.getData)        
    
        
        #Communication with the ni card        
        
        #get the input 
        self.citask = nidaqmx.CounterInputTask()
        self.citask.create_channel_count_edges(channel, init=0 )
        self.citask.set_terminal_count_edges(channel,terminal)
#        self.citask.configure_timing_sample_clock(source=r'ai/SampleClock',samples_per_channel=samp_per_chan+500,sample_mode="finite")
        layout= QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.plotLayout,0,0,7,5)
        layout.addWidget(self.button,3,5,1,1)
        layout.setRowMinimumHeight(6,150)
        
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getData)

        
        
    def getData(self):
        """To record and display the signal. This script is called at regular time intervals."""
        if(self.isRunning):
            try:
                data = self.citask.read(700,timeout=0.1)
                dat=np.concatenate(([0],data[:-1]))
                self.wave.setData((data-dat)[1:])
            except:
                print("error")
        
    def start(self):
        """starts/stops the data acquisition"""
        if(self.isRunning):
            #Stop everything
            self.isRunning = False
            self.timer.stop()
            self.citask.stop()
            self.citask.wait_until_done()
            self.button.setText("Start")
            
        else:
            #Starts everything
            self.isRunning= True
            self.citask.start()
            self.timer.start(10)
            self.button.setText("Stop")
    
    def closeEvent(self,*args,**kwargs):
        super().closeEvent(*args,**kwargs)
        self.timer.stop()
        
if __name__=="__main__":
    app = QtGui.QApplication([])
    menu= Oscilloscope()
    menu.show()
    app.exec_()