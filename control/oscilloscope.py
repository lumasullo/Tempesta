# -*- coding: utf-8 -*-


from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import time
try:
    import nidaqmx
except:
    print("failed to import nidaqmx. But in only occures when generating documentation")
    pass

class Oscilloscope(QtGui.QWidget):
    """Class defining an oscilloscope to monitor a signal from a nidaq card"""
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
        self.citask = nidaqmx.CounterInputTask()
        self.citask.create_channel_count_edges("Dev1/ctr0", init=0 )
        self.citask.set_terminal_count_edges("Dev1/ctr0","PFI0")
#        self.citask.configure_timing_sample_clock(source=r'ai/SampleClock',samples_per_channel=samp_per_chan+500,sample_mode="finite")
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
        """Switches the recording channel."""
        if(self.isRunning):
            self.timer.stop()
            self.vitask.stop()
            del self.citask
        
        
        self.citask = nidaqmx.AnalogInputTask()
        self.citask.create_voltage_channel(channel, terminal = 'rse', min_val=-1, max_val=10.0)
        self.citask.configure_timing_sample_clock(rate = 100000.0,sample_mode="continuous")
        if(self.isRunning):
            self.citask.start()
            self.timer.start()
        
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
        self.nidaq.reset()
        
if __name__=="__main__":
    app = QtGui.QApplication([])
    menu= Oscilloscope()
    menu.show()
    app.exec_()