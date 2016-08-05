# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
modified by aurelien
"""


import nidaqmx

    
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from tkinter import Tk, filedialog, messagebox
from PIL import Image

import datetime

#from numpy import arange
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
reference_trigger='PFI12'
#Voltage Limit to apply to the devices
#xy stage:
min_ao_horizontal = -10
max_ao_horizontal = 10



#z stage:
min_ao_vertical = 0
max_ao_vertical = 10

#These factors convert an input in microns to the output in volts
x_factor = 3.75
y_factor = 3.75
z_factor = 10

pmt_sensitivity_channel = 'Dev1/ao3'

sample_rate = 10**5

# This class is intended as a widget in the bigger GUI, Thus all the commented parameters etc. It contain an instance
# of stage_scan and pixel_scan which in turn harbour the analog and digital signals respectively.
# The function run is the function that is supposed to start communication with the Nidaq
# through the Scanner object. This object was initially written as a QThread object but is not right now. 
# As seen in the commened lines of run() I also tried running in a QThread created in run().
# The rest of the functions contain mosly GUI related code.

class ScanWidget(QtGui.QFrame):
    
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.aochannels = device.get_analog_output_channels()
        self.dochannels = device.get_digital_output_lines()
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        
        self.back_factor = 10
        self.back_factor_param=QtGui.QLineEdit("100")
        
        #Creating the GUI itself
        self.widthPar = QtGui.QLineEdit('5')
        self.widthPar.editingFinished.connect(lambda: self.ScanParameterChanged('width'))
        self.heightPar = QtGui.QLineEdit('5')
        self.heightPar.editingFinished.connect(lambda: self.ScanParameterChanged('height'))
        self.sequence_timePar = QtGui.QLineEdit('0.0001') # Seconds
        self.sequence_timePar.editingFinished.connect(lambda: self.ScanParameterChanged('sequence_time'))
        self.nrFramesPar = QtGui.QLabel()
        self.step_sizePar = QtGui.QLineEdit('0.01')
        self.step_sizePar.editingFinished.connect(lambda: self.ScanParameterChanged('step_size'))
        self.sample_rate = 20000
        delay= 65*self.sample_rate / 20000      #experimental
        delay=int(delay)
        self.delay = QtGui.QLineEdit(str(delay))
        
        self.Scan_Mode = QtGui.QComboBox()
        self.scan_modes = ['xy scan','xz scan', 'Line scan']
        self.Scan_Mode.addItems(self.scan_modes)
        self.Scan_Mode.currentIndexChanged.connect(lambda: self.setScanMode(self.Scan_Mode.currentText()))
        
        self.scan_parameters = {'width': self.widthPar, 'height': self.heightPar,
                           'sequence_time': self.sequence_timePar,
                           'conversion': self.nrFramesPar,
                           'step_size': self.step_sizePar}

        self.scan_par_values = {'width': float(self.widthPar.text()),
                           'height': float(self.heightPar.text()),
                           'sequence_time': float(self.sequence_timePar.text()),
                           'frames': '-',
                           'step_size': float(self.step_sizePar.text())}
                          

                   
        self.start488Par = QtGui.QLineEdit('0')
        self.start488Par.editingFinished.connect(lambda: self.PixelParameterChanged('start488'))
        self.start405Par = QtGui.QLineEdit('0')
        self.start405Par.editingFinished.connect(lambda: self.PixelParameterChanged('start405'))
        self.start355Par = QtGui.QLineEdit('0')
        self.start355Par.editingFinished.connect(lambda: self.PixelParameterChanged('start355'))
        self.startTISPar = QtGui.QLineEdit('0')
        self.startTISPar.editingFinished.connect(lambda: self.PixelParameterChanged('startTIS'))
        self.startCAMPar = QtGui.QLineEdit('0')
        self.startCAMPar.editingFinished.connect(lambda: self.PixelParameterChanged('startCAM'))
        
        self.end488Par = QtGui.QLineEdit('0')
        self.end488Par.editingFinished.connect(lambda: self.PixelParameterChanged('end488'))
        self.end405Par = QtGui.QLineEdit('0')
        self.end405Par.editingFinished.connect(lambda: self.PixelParameterChanged('end405'))
        self.end355Par = QtGui.QLineEdit('0')
        self.end355Par.editingFinished.connect(lambda: self.PixelParameterChanged('end355'))
        self.endTISPar = QtGui.QLineEdit('0')
        self.endTISPar.editingFinished.connect(lambda: self.PixelParameterChanged('endTIS'))
        self.endCAMPar = QtGui.QLineEdit('0')
        self.endCAMPar.editingFinished.connect(lambda: self.PixelParameterChanged('endCAM'))
        
        

        self.pixel_parameters = {'startTIS': self.startTISPar,
                                 'start355': self.start355Par,
                                 'start405': self.start405Par,
                                 'start488': self.start488Par,
                                 'startCAM': self.startCAMPar,
                                 'end488': self.end488Par,
                                 'end405': self.end405Par,
                                 'end355': self.end355Par,
                                 'endTIS': self.endTISPar,
                                 'endCAM': self.endCAMPar}

        
        self.pixel_par_values = {'startTIS': float(self.startTISPar.text()),
                                 'start355': float(self.start355Par.text()),
                                 'start405': float(self.start405Par.text()),
                                 'start488': float(self.start488Par.text()),
                                 'startCAM': float(self.startCAMPar.text()),
                                 'end488': float(self.end488Par.text()),
                                 'end405': float(self.end405Par.text()),
                                 'end355': float(self.end355Par.text()),
                                 'endTIS': float(self.endTISPar.text()),
                                 'endCAM': float(self.endCAMPar.text())}
                                 
        self.current_dochannels = {'TIS': 0, '355': 1, '405': 2, '488': 3, 'CAM': 4}
        self.current_aochannels = {'x': 0, 'y': 1,'z':2}
        self.XchanPar = QtGui.QComboBox()
        self.XchanPar.addItems(self.aochannels)
        self.XchanPar.setCurrentIndex(0)
        self.XchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
        self.XchanPar.setDisabled(True)
        self.YchanPar = QtGui.QComboBox()
        self.YchanPar.addItems(self.aochannels)
        self.YchanPar.setCurrentIndex(1)
        self.YchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
        self.YchanPar.setDisabled(True)
        self.ZchanPar = QtGui.QComboBox()
        self.ZchanPar.addItems(self.aochannels)
        self.ZchanPar.setCurrentIndex(2)
        self.ZchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
        self.ZchanPar.setDisabled(True)
        
        self.chanTISPar = QtGui.QComboBox()
        self.chanTISPar.addItems(self.dochannels)
        self.chan355Par = QtGui.QComboBox()
        self.chan355Par.addItems(self.dochannels)
        self.chan405Par = QtGui.QComboBox()
        self.chan405Par.addItems(self.dochannels)
        self.chan488Par = QtGui.QComboBox()
        self.chan488Par.addItems(self.dochannels)
        self.chanCAMPar = QtGui.QComboBox()
        self.chanCAMPar.addItems(self.dochannels)
        self.DOchan_Pars_dict = {'TIS': self.chanTISPar, '355': self.chan355Par, '405': self.chan405Par, '488': self.chan488Par, 'CAM': self.chanCAMPar}
        self.chanTISPar.currentIndexChanged.connect(lambda: self.DOchannelsChanged('TIS', self.chanTISPar.currentIndex()))
        self.chan355Par.currentIndexChanged.connect(lambda: self.DOchannelsChanged('355', self.chan355Par.currentIndex()))
        self.chan405Par.currentIndexChanged.connect(lambda: self.DOchannelsChanged('405', self.chan405Par.currentIndex()))
        self.chan488Par.currentIndexChanged.connect(lambda: self.DOchannelsChanged('488', self.chan488Par.currentIndex()))
        self.chanCAMPar.currentIndexChanged.connect(lambda: self.DOchannelsChanged('CAM', self.chanCAMPar.currentIndex()))
        self.AOchannelsChanged()
        for sig in self.current_dochannels:
            self.DOchan_Pars_dict[sig].setCurrentIndex(self.current_dochannels[sig])
    
        self.stage_scan = StageScan(self,self.sample_rate)
        self.stage_scan.update_frames(self.scan_par_values)
        self.ScanParameterChanged("width")      #Used to actualise the number of frames displayed on screen
        
        self.pixel_cycle = PixelCycle(self.sample_rate)
        self.graph = GraphFrame(self.pixel_cycle)
        self.update_Scan(['TIS', '355', '405', '488', 'CAM'])
        
        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contScanRadio = QtGui.QRadioButton('Cont. Scan')
        self.contScanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(lambda: self.setScanOrNot(False))
        
        
        self.ScanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.ScanButton.clicked.connect(self.ScanOrAbort)
        self.PreviewButton = QtGui.QPushButton('Preview')
        self.PreviewButton.clicked.connect(self.PreviewScan)
        
        
        
        self.display = ImageDisplay(self,(10,10))
        self.positionner = Positionner(self)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(QtGui.QLabel('X channel'), 0, 4)
        grid.addWidget(self.XchanPar, 0, 5)
        grid.addWidget(QtGui.QLabel('Width (um):'), 0, 0, 2, 1)
        grid.addWidget(self.widthPar, 0, 1, 2, 1)
        grid.addWidget(QtGui.QLabel('Height (um):'), 0, 2, 2, 1)
        grid.addWidget(self.heightPar, 0, 3, 2, 1)
        grid.addWidget(QtGui.QLabel('Y channel'), 1, 4)
        grid.addWidget(self.YchanPar, 1, 5)
        grid.addWidget(QtGui.QLabel('Z channel'),2,4)
        grid.addWidget(self.ZchanPar,2,5)        

        grid.addWidget(QtGui.QLabel('Sequence Time (s):'),2, 0)
        grid.addWidget(self.sequence_timePar, 2, 1)
        grid.addWidget(QtGui.QLabel('Frames in scan:'), 2, 2)
        grid.addWidget(self.nrFramesPar, 2, 3)
        grid.addWidget(QtGui.QLabel('Step size (um):'), 3, 4)
        grid.addWidget(self.step_sizePar, 3, 5)
        grid.addWidget(QtGui.QLabel('correction samples:'), 4, 4)
        grid.addWidget(self.delay, 4, 5)
        grid.addWidget(QtGui.QLabel('back_factor'), 5, 4)
        grid.addWidget(self.back_factor_param, 5, 5)
        
        grid.addWidget(QtGui.QLabel('Start:'), 5, 1)
        grid.addWidget(QtGui.QLabel('End:'), 5, 2)
        grid.addWidget(QtGui.QLabel('TIS:'), 6, 0)
        grid.addWidget(self.startTISPar, 6, 1)
        grid.addWidget(self.endTISPar, 6, 2)
        grid.addWidget(self.chanTISPar, 6, 3)
        grid.addWidget(QtGui.QLabel('355:'), 7, 0)
        grid.addWidget(self.start355Par, 7, 1)
        grid.addWidget(self.end355Par, 7, 2)
        grid.addWidget(self.chan355Par, 7, 3)
        grid.addWidget(self.scanRadio, 7, 4, 2, 1)
        grid.addWidget(self.contScanRadio,8,4,2,1)
        grid.addWidget(self.Scan_Mode, 7, 5)
        grid.addWidget(QtGui.QLabel('405:'), 8, 0)
        grid.addWidget(self.start405Par, 8, 1)
        grid.addWidget(self.end405Par, 8, 2)
        grid.addWidget(self.chan405Par, 8, 3)
        grid.addWidget(self.contLaserPulsesRadio, 9, 4, 2, 1)
        grid.addWidget(QtGui.QLabel('488:'), 9, 0)
        grid.addWidget(self.start488Par, 9, 1)
        grid.addWidget(self.end488Par, 9, 2)
        grid.addWidget(self.chan488Par, 9, 3)
        grid.addWidget(QtGui.QLabel('CAM:'), 10, 0)
        grid.addWidget(self.startCAMPar, 10, 1)
        grid.addWidget(self.endCAMPar, 10, 2)
        grid.addWidget(self.chanCAMPar, 10, 3)
        grid.addWidget(self.graph, 11, 0, 1, 6)
        grid.addWidget(self.positionner,12,0,1,4)
        grid.addWidget(self.ScanButton, 13, 3)
        grid.addWidget(self.PreviewButton, 13, 4)        
        
    @property
    def scanOrNot(self):
        return self._scanOrNot

    @scanOrNot.setter
    def scanOrNot(self, value):
        self.EnableScanPars(value)
        self.ScanButton.setCheckable(not value)
        
    def EnableScanPars(self, value):
        self.widthPar.setEnabled(value)
        self.heightPar.setEnabled(value)
#        self.sequence_timePar.setEnabled(value)
        self.step_sizePar.setEnabled(value)
        if value:
            self.ScanButton.setText('Scan')
        else:
            self.ScanButton.setText('Run')
    
    def setScanOrNot(self, value):
        self.scanOrNot = value 
    
    def setScanMode(self, mode):             
            
        self.stage_scan.set_scan_mode(mode)
        self.ScanParameterChanged('scan_mode')
        
    def AOchannelsChanged (self):
        
        Xchan = self.XchanPar.currentIndex()
        Ychan = self.YchanPar.currentIndex()
        Zchan = self.ZchanPar.currentIndex()
        if Xchan == Ychan:
            Ychan = (Ychan + 1)%4
            self.YchanPar.setCurrentIndex(Ychan)
        count=len(self.aochannels)
        while( (Zchan == Xchan or Zchan == Ychan) and count>0):
            Zchan = (Zchan + 1)%len(self.aochannels)
            self.ZchanPar.setCurrentIndex(Zchan)
            count-=1
        if(count == 0):
            print("couldn't find satisfying channel for Z")
        self.current_aochannels['x'] = Xchan
        self.current_aochannels['y'] = Ychan
        self.current_aochannels['z'] = Zchan
        
    def DOchannelsChanged(self, sig, new_index):
        
        for i in self.current_dochannels:
            if i != sig and new_index == self.current_dochannels[i]:
                self.DOchan_Pars_dict[sig].setCurrentIndex(self.current_dochannels[sig])
        
        self.current_dochannels[sig] = self.DOchan_Pars_dict[sig].currentIndex()
        
        
    def ScanParameterChanged(self, parameter):
        if not parameter == 'scan_mode':
            self.scan_par_values[parameter] = float(self.scan_parameters[parameter].text())
            
        if parameter == 'sequence_time':
            self.update_Scan(['TIS', '355', '405', '488', 'CAM'])
            self.graph.update(['TIS', '355', '405', '488', 'CAM'])
        print('In ScanParameterChanged')
        self.stage_scan.update_frames(self.scan_par_values)
        self.nrFramesPar.setText(str(self.stage_scan.frames))
                
        
    def PixelParameterChanged(self, parameter):
        self.pixel_par_values[parameter] = float(self.pixel_parameters[parameter].text())
        device = parameter[-3]+parameter[-2]+parameter[-1]
        self.pixel_cycle.update([device], self.pixel_par_values, self.stage_scan.sequence_samples) 
        self.graph.update([device])
        
    def PreviewScan(self):
        
        self.stage_scan.update(self.scan_par_values)
        x_sig = self.stage_scan.sig_dict['x_sig']
        y_sig = self.stage_scan.sig_dict['y_sig']
        plt.figure()
        plt.plot(x_sig, y_sig)
        plt.axis([-0.2,self.scan_par_values['width']/x_factor+0.2, -0.2, self.scan_par_values['height']/y_factor+0.2])
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        print("preview enclenched")
        plt.show()
        
    def ScanOrAbort(self):
        if not self.scanning:
            self.PrepAndRun()
        else:
            self.scanner.abort()
    
    def PrepAndRun(self):

        if self.scanRadio.isChecked() or self.contScanRadio.isChecked():
            self.stage_scan.update(self.scan_par_values)
            self.ScanButton.setText('Abort')
            self.positionner.reset_channels()
            self.scanner = Scanner(self.nidaq, self.stage_scan, self.pixel_cycle, self.current_aochannels, self.current_dochannels, self)
            self.scanner.finalizeDone.connect(self.FinalizeDone)    
            self.scanner.scanDone.connect(self.ScanDone)            
            self.scanning = True
            self.scanner.runScan()
            
        elif self.ScanButton.isChecked():
            self.lasercycle = LaserCycle(self.nidaq, self.pixel_cycle, self.current_dochannels)
            self.ScanButton.setText('Stop')
            self.lasercycle.run()
            
        else:
            self.lasercycle.stop()
            self.ScanButton.setText('Run')
            del self.lasercycle
            
    def ScanDone(self):
        print('in ScanDone()')
        self.ScanButton.setEnabled(False)
        
    def FinalizeDone(self):
        self.ScanButton.setText('Scan')
        self.ScanButton.setEnabled(True)
        print('Scan Done')
        del self.scanner
        self.scanning = False
        
    def update_Scan(self, devices):
        self.stage_scan.update(self.scan_par_values)
        self.pixel_cycle.update(devices, self.pixel_par_values, self.stage_scan.sequence_samples)
        
    def closeEvent(self, *args, **kwargs):        
        super().closeEvent(*args, **kwargs)
        plt.close("all")
#        self.nidaq.reset()
        try:
            self.scanner.waiter.terminate()
        except:
            pass
        
        
# This class was intended as a QThread to not intefere with the rest of the widgets in the GUI. 
  # Apparently not used for the moment

class Wait_Thread(QtCore.QThread):
    waitdoneSignal = QtCore.pyqtSignal()
    def __init__(self, aotask):
        super().__init__()
        self.aotask = aotask
        self.wait = True
            
    def run(self):
        print('will wait for aotask')       
#        self.aotask.wait_until_done()
        while not self.aotask.is_done() and self.wait:
            pass
        self.wait = True
        self.waitdoneSignal.emit()
        print('aotask is done')
    
    def stop(self):
        self.wait = False

class Scanner(QtCore.QObject):
    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()
    def __init__(self, device, stage_scan, pixel_cycle, current_aochannels, current_dochannels, main, *args, **kwargs):
        """device: nidaqmx device, corresponds to the NI PCI card
        stage_scan: object containing the analog signals to drive the stage
        pixel_cycle: object containing the digital signals to drive the lasers at each pixel acquisition
        current_aochannels: available analog output channels
        current_dochannels: available digital output channels
        main: reference to an instace of ScanWidget"""
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.stage_scan = stage_scan
        self.pixel_cycle = pixel_cycle
        self.current_aochannels = current_aochannels  # Dict containing channel numbers to be written to for each signal
        self.current_dochannels = current_dochannels  # Dict containing channel numbers to be written to for each device
        self.samples_in_scan = len(self.stage_scan.sig_dict['x_sig'])
        self.main = main
        
        self.aotask = nidaqmx.AnalogOutputTask("scannerAOtask")
        self.dotask = nidaqmx.DigitalOutputTask("scannerDOtask")       
        self.waiter = Wait_Thread(self.aotask)
        self.record_thread = RecordingThreadPMT(self.main.display)
        
        self.contScan=self.main.contScanRadio.isChecked()    #Boolean specifying if we are running a continuous scanning or not
        
        self.full_ao_signal = []
        self.full_do_signal = []
        
        self.warning_time= 10
        self.scantimewar = QtGui.QMessageBox()
        self.scantimewar.setInformativeText("Are you sure you want to continue?")
        self.scantimewar.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        
        self.connect(self.record_thread, QtCore.SIGNAL("measure(float)"), self.main.display.set_pixel_value)
        self.connect(self.record_thread ,  QtCore.SIGNAL("line(PyQt_PyObject)"),self.main.display.set_line_value )
        
    def finalize(self):
        print('in finalize')
        self.contScan=self.main.contScanRadio.isChecked()    #Boolean specifying if we are running a continuous scanning or not
        if(not self.contScan):            
            self.scanDone.emit()
        self.waiter.waitdoneSignal.disconnect(self.finalize) #Important, otherwise finalize is called again when next waiting finishes.
        self.waiter.waitdoneSignal.connect(self.done)
            
        written_samps = self.aotask.get_samples_per_channel_generated()
        goals = [0,0]  #Where we write the target positions to return to
        if self.stage_scan.scan_mode == "xz scan":
            final_x = self.stage_scan.sig_dict['x_sig'][written_samps - 1]
            final_y = self.stage_scan.sig_dict['z_sig'][written_samps - 1]
            goals[0] = self.main.positionner.x
            goals[1]  =self.main.positionner.z
        else:
            final_x = self.stage_scan.sig_dict['x_sig'][written_samps - 1]
            final_y = self.stage_scan.sig_dict['y_sig'][written_samps - 1]
            goals[0]= self.main.positionner.x
            goals[1]  =self.main.positionner.y
            
        final_samps = [final_x, final_y]
        
        return_ramps = np.array([])
        for i in range(0,2):
            ramp_and_k = make_ramp(final_samps[i], goals[i], self.stage_scan.sample_rate)
            return_ramps = np.append(return_ramps, ramp_and_k[0])
            
        magic = np.ones(100)  # Seems to decrease frequency of Invalid task errors.      
        
        self.aotask.stop()    
        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                                 sample_mode = 'finite',
                                                 samples_per_channel = self.stage_scan.sample_rate)        
        
        self.aotask.write(return_ramps, layout = 'group_by_channel', auto_start = False)
        self.aotask.start()

        self.waiter.start()
        

            


    def done(self):
        print('in self.done()')
        if(self.contScan):
            #If scanning continuously, regenerate the samples and write them again
            self.aotask.stop()
            self.dotask.stop()
            self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                             sample_mode = 'finite',
                                             samples_per_channel = self.samples_in_scan)
                        
            self.dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                             rate = self.pixel_cycle.sample_rate, 
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan)
                                             
            self.waiter.waitdoneSignal.disconnect(self.done)
            self.waiter.waitdoneSignal.connect(self.finalize)
        
            self.dotask.write(self.full_do_signal, layout = 'group_by_channel', auto_start = False)
            self.aotask.write(self.full_ao_signal, layout = 'group_by_channel', auto_start = False)
            self.dotask.start()
            self.aotask.start()
            self.record_thread.start()
            self.waiter.start()
            
        else:
            self.aotask.clear()
            self.dotask.clear()
            self.record_thread.stop()
#            self.nidaq.reset()
            self.main.positionner.reset_channels()
            self.finalizeDone.emit()
            self.trigger.clear()
            del self.trigger
            print("total Done")
        
    def runScan(self):
#        self.nidaq.reset()

#        self.main.positionner.reset_channels()  #Disables the positionner to make the channels available for scan
        self.n_frames = self.stage_scan.frames
        
        image_shape = (self.stage_scan.steps_y , self.stage_scan.steps_x )
        self.main.display.update_parameters(image_shape)
        
        scan_time = self.samples_in_scan / self.main.sample_rate
        ret = QtGui.QMessageBox.Yes
        self.scantimewar.setText("Scan will last %s seconds" %scan_time)
        if scan_time > self.warning_time:
            ret = self.scantimewar.exec_()
            
        if ret == QtGui.QMessageBox.No:
            self.done()
            return
#        self.nidaq.reset() 


        self.full_ao_signal = []
        temp_aochannels = copy.copy(self.current_aochannels)
#        min_ao = -10
#        max_ao = 10
           
        # Following loop creates the voltage channels in smallest to largest order and places signals in same order.
        if(self.stage_scan.scan_mode == "xz scan"):
            chanstringx = 'Dev1/ao%s'%temp_aochannels["x"]
            self.aotask.create_voltage_channel(phys_channel = chanstringx, channel_name = 'chanx', min_val = min_ao_horizontal, max_val = max_ao_horizontal)
            signalx = self.stage_scan.sig_dict['x_sig']
            chanstringz='Dev1/ao%s'%temp_aochannels["z"]
            self.aotask.create_voltage_channel(phys_channel = chanstringz, channel_name = 'chanz', min_val = min_ao_vertical, max_val = max_ao_vertical)
            signalz = self.stage_scan.sig_dict['z_sig']
            print("length signals:",len(signalz),len(signalx))
            self.full_ao_signal = np.append(signalx,signalz)
            
        else:
            chanstringx = 'Dev1/ao%s'%temp_aochannels["x"]
            self.aotask.create_voltage_channel(phys_channel = chanstringx, channel_name = 'chanx', min_val = min_ao_horizontal, max_val = max_ao_horizontal)
            signalx = self.stage_scan.sig_dict['x_sig']
            chanstringy='Dev1/ao%s'%temp_aochannels["y"]
            self.aotask.create_voltage_channel(phys_channel = chanstringy, channel_name = 'chany', min_val = min_ao_horizontal, max_val = max_ao_horizontal)
            signaly = self.stage_scan.sig_dict['y_sig']
            self.full_ao_signal = np.append(signalx,signaly)
            
        # Same as above but for the digital signals/devices        
        
        self.full_do_signal = []
        temp_dochannels = copy.copy(self.current_dochannels)
        for i in range(0,len(temp_dochannels)):
            dev = min(temp_dochannels, key = temp_dochannels.get)
            chanstring = 'Dev1/port0/line%s'%temp_dochannels[dev]
            self.dotask.create_channel(lines = chanstring, name = 'chan%s'%dev)
            temp_dochannels.pop(dev)
            signal = self.pixel_cycle.sig_dict[dev+'sig']
            print("do signal length",len(signal))
            if len(self.full_ao_signal)%len(signal) != 0 and len(self.full_do_signal)%len(signal) != 0:
                print('Signal lengths does not match (printed from run)')
            self.full_do_signal = np.append(self.full_do_signal, signal)
        

        
        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                             sample_mode = 'finite',
                                             samples_per_channel = self.samples_in_scan)
                        
        self.dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                             rate = self.pixel_cycle.sample_rate, 
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan)
                                             
#        self.aotask.configure_analog_edge_reference_trigger()
#                                             
        self.trigger = nidaqmx.CounterOutputTask("trigger")
        self.trigger.create_channel_ticks('Dev1/ctr1',name="pasteque",  low_ticks=100000, high_ticks=1000000)
        self.trigger.set_terminal_pulse('Dev1/ctr1',"PFI12")
        self.trigger.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                             sample_mode = "hwtimed")
        self.aotask.configure_trigger_digital_edge_start(reference_trigger)
        
        self.waiter.waitdoneSignal.connect(self.finalize)
        self.dotask.write(self.full_do_signal, layout = 'group_by_channel', auto_start = False)
        self.aotask.write(self.full_ao_signal, layout = 'group_by_channel', auto_start = False)
        self.record_thread.setParameters(self.stage_scan.sequence_samples,self.samples_in_scan,self.stage_scan.sample_rate,self.stage_scan.samples_per_line)
        
        self.dotask.start()
        self.aotask.start()
        self.record_thread.start()
        self.waiter.start()
        time.sleep(0.1)
        self.trigger.start()
#        self.trigger.write(np.append(np.zeros(1000),np.ones(1000)))
      
        
    def abort(self):
        print('Aborting scan')
        self.waiter.stop()
        self.aotask.stop()
        self.dotask.stop()
        self.record_thread.exiting = True
        self.main.scanRadio.setChecked(True)    #To prevent from starting a continuous acquisition in finalize
        self.finalize()        

class LaserCycle():
    
    def __init__(self, device, pixel_cycle, curren_dochannels):
        
        self.nidaq = device
        self.pixel_cycle = pixel_cycle
        self.current_dochannels = curren_dochannels
        
    def run(self):
        self.dotask = nidaqmx.DigitalOutputTask('dotask') 
        
        full_do_signal = []
        temp_dochannels = copy.copy(self.current_dochannels)
        for i in range(0,len(temp_dochannels)):
            dev = min(temp_dochannels, key = temp_dochannels.get)
            chanstring = 'Dev1/port0/line%s'%temp_dochannels[dev]
            self.dotask.create_channel(lines = chanstring, name = 'chan%s'%dev)
            temp_dochannels.pop(dev)
            signal = self.pixel_cycle.sig_dict[dev+'sig']
            full_do_signal = np.append(full_do_signal, signal)

        self.dotask.configure_timing_sample_clock(source = r'100kHzTimeBase', 
                                             rate = self.pixel_cycle.sample_rate, 
                                             sample_mode = 'continuous')        
        self.dotask.write(full_do_signal, layout = 'group_by_channel', auto_start = False)
        
        self.dotask.start()
        
    def stop(self):
         self.dotask.stop()
         self.dotask.clear()
         del self.dotask
         self.nidaq.reset()
        
# Following class


class StageScan():
    """ contains the analog signals in sig_dict. The update function takes the parameter_values
    and updates the signals accordingly
    main: ScanWidget
    sample_rate : sample rate in number of samples per second"""
    
    def __init__(self,main,sample_rate):
        self.scan_mode = 'xy scan'
        self.sig_dict = {'x_sig': [], 'y_sig': [],'z_sig':[]}
        self.sample_rate = sample_rate
        self.sequence_samples = None    #Corresponds to the number of samples within a pixel sequence.
        self.xy_scan = xy_Scan(main,self.sample_rate)
        self.xz_scan = xz_Scan(main,self.sample_rate)
        self.line_scan = Line_scan(self.sample_rate)
        self.scans = {'xy scan': self.xy_scan,'xz scan':self.xz_scan, 'Line scan': self.line_scan}
        self.frames = 0
        self.scanWidget = main
        
        self.samples_per_line=0
        
        self.steps_x= 0
        self.steps_y = 0
        
    def set_scan_mode(self, mode):
        self.scan_mode = mode  
        
    def update_frames(self, par_values):
        self.scans[self.scan_mode].update_frames(par_values)
        self.frames = self.scans[self.scan_mode].frames
        self.steps_x = self.scans[self.scan_mode].steps_x
        self.steps_y = self.scans[self.scan_mode].steps_y
        
    def update(self, par_values):
        print('self.scan_mode = ', self.scan_mode)
        self.scans[self.scan_mode].update(par_values)
        self.sig_dict = self.scans[self.scan_mode].sig_dict
        self.sequence_samples = self.scans[self.scan_mode].sequence_samples
        self.frames = self.scans[self.scan_mode].frames
        self.samples_per_line = self.scans[self.scan_mode].samples_per_line
        
            
class Line_scan():
        
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': [],'z_sig':[]}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
        self.steps_y = 0
        self.steps_x = 0
        
    def update_frames(self, par_values):
        size_y = par_values['height'] / 2
        step_size = par_values['step_size'] / 2
        self.steps_y = int(np.ceil(size_y / step_size))
        self.frames = self.steps_y
        
    def update(self, par_values):
        
        # Create signals
        start_x = 0
        start_y = 0
        size_y = par_values['height'] / 2
        sequence_samples = np.round(self.sample_rate * par_values['sequence_time'])
        step_size = par_values['step_size'] / 2
        self.steps_y = int(np.ceil(size_y / step_size))
        self.corr_step_size = size_y/self.steps_y # Step size compatible with width
        self.sequence_samples = int(sequence_samples)
        column_samples = self.steps_y * self.sequence_samples
        ramp_and_k = make_ramp(start_y, size_y, column_samples) # ramp_and_k contains [ramp, k]
        ramp = ramp_and_k[0]
        
        self.sig_dict['y_sig'] = ramp
        self.sig_dict['x_sig'] = np.zeros(len(ramp))
        
class xy_Scan():
    """Scanning in xy"""
    def __init__(self,main, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
        self.scanWidget=main
        
        self.steps_x = 0
        self.steps_y = 0
        self.samples_per_line=0
        # Update signals according to parameters.
        # Note that rounding floats to ints may cause actual scan to differ slighly from expected scan. 
        # Maybe either limit input parameters to numbers that "fit each other" or find other solution, eg step size has to
        # be width divided by an integer.
        # Maybe not a problem ???
        
    def update_frames(self, par_values):
        step_size = par_values['step_size'] / x_factor
        size_x = par_values['width'] / x_factor
        size_y = par_values['height'] / y_factor
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        self.frames = self.steps_y*self.steps_x
                
        
    def update(self, par_values):
        
        # Create signals
        try:
            start_x = self.scanWidget.positionner.x
            start_y = self.scanWidget.positionner.y
        except:
            start_x = 0
            start_y = 0
        size_x = par_values['width'] / x_factor
        size_y = par_values['height'] / y_factor
        step_size = par_values['step_size'] / x_factor
        self.sequence_samples = int(np.round(self.sample_rate * par_values['sequence_time'])) #WARNING: Correct for units of the time, now seconds!!!!
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        self.corr_step_size = size_x/self.steps_x # Step size compatible with width
        row_samples = self.steps_x * self.sequence_samples
#        """!!!!! Caution if size doesn't work with current position"""
#        ramp_and_k = make_ramp(start_x, size_x+start_x, row_samples) # ramp_and_k contains [ramp, k]
#        k = ramp_and_k[1]
#        ltr_ramp = ramp_and_k[0]
#        rtl_ramp = ltr_ramp[::-1]  # rtl_ramp contains only ramp, no k since same k = -k
#        gradual_k = make_ramp(k, -k, self.sequence_samples)
#        turn_rtl = np.cumsum(gradual_k[0])
#        turn_ltr = -turn_rtl
#        max_turn = np.max(turn_rtl)
#        adjustor = 1 - self.sequence_samples%2
#        
#        first_part =  max_turn - turn_rtl[range(int(np.ceil(self.sequence_samples/2)), self.sequence_samples)] # Create first and last part by flipping and turnign the turn_rtl array
#        last_part = max_turn + turn_rtl[range(0, int(np.floor(self.sequence_samples/2) - adjustor + 1))]
#        y_ramp_smooth = np.append(first_part, last_part)
#        y_ramp_smooth = (self.corr_step_size/(2*max_turn)) * y_ramp_smooth        # adjust scale and offset of ramp
#        turn_rtl = ltr_ramp[-1] + turn_rtl
#        turn_ltr = rtl_ramp[-1] + turn_ltr
        x_sig = []
        y_sig = []
        new_value = start_y

        
        
        ramp_and_k = make_ramp(start_x, size_x+start_x, row_samples)
        ltr_ramp = ramp_and_k[0]
        back_factor= int(self.scanWidget.back_factor_param.text())
        back_ramp = make_ramp(size_x+start_x,start_x, back_factor*self.sequence_samples)
        back_ramp= back_ramp[0]
#        wait_signal_ltr = ltr_ramp[-1] * np.ones(self.sequence_samples)
#        wait_signal_rtl = rtl_ramp[-1] * np.ones(self.sequence_samples)

        y_ramp = make_ramp(0,self.corr_step_size, back_factor*self.sequence_samples)[0]
        
        for i in range(0,self.steps_y):
            x_sig = np.concatenate((x_sig,back_ramp,ltr_ramp))
            y_sig = np.concatenate((y_sig, new_value+y_ramp,new_value*np.ones(row_samples)))
            new_value = new_value + self.corr_step_size       
            
        self.samples_per_line = row_samples + back_factor*self.sequence_samples
            
#        i = i + 1
#        if i%2 == 0:
#            x_sig = np.concatenate((x_sig, ltr_ramp))
#        else:
#            x_sig = np.concatenate((x_sig, rtl_ramp))  
#        y_sig = np.concatenate((y_sig, new_value*np.ones(row_samples)))

        # Assign x_sig
        self.sig_dict['x_sig'] =  x_sig
        # Assign y_sig
        self.sig_dict['y_sig'] =  y_sig
        
class xz_Scan():
    """Scanning in xz"""
    def __init__(self,main,sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': [] , 'z_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
        self.samples_per_line=0
        self.scanWidget = main
        
        self.steps_x = 0
        self.steps_y = 0
        
    def update_frames(self, par_values):
        step_size = par_values['step_size'] / x_factor
        size_x = par_values['width'] / x_factor
        size_y = par_values['height'] / y_factor
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        self.frames = self.steps_y*self.steps_x
                
        
    def update(self, par_values):
        """creates the signals inside self.sig_dict"""
        try:
            start_x = self.scanWidget.positionner.x
            start_z = self.scanWidget.positionner.z
        except:
            start_x = 0
            start_z = 0
        size_x = par_values['width'] / x_factor
        size_z = par_values['height'] / z_factor
        step_size_x = par_values['step_size'] / x_factor
        step_size_z = par_values['step_size'] / z_factor
        self.sequence_samples = int(np.round(self.sample_rate * par_values['sequence_time'])) #WARNING: Correct for units of the time, now seconds!!!!
        self.steps_x = int(np.ceil(size_x / step_size_x))
        self.steps_z = int(np.ceil(size_z / step_size_z))
        self.corr_step_size_x = size_x/self.steps_x # Step size compatible with width
        self.corr_step_size_z = self.corr_step_size_x*x_factor/z_factor
        row_samples = self.steps_x * self.sequence_samples
        
        #Generation of the signals themselves
        x_sig = []
        z_sig = []
        new_value = start_z
        ramp_and_k = make_ramp(start_x, size_x+start_x, row_samples)
        ltr_ramp = ramp_and_k[0]
        back_factor= int(self.scanWidget.back_factor_param.text())
        
        back_ramp = make_ramp(size_x+start_x,start_x, back_factor*self.sequence_samples)
        back_ramp=back_ramp[0]
        
        z_ramp = make_ramp(0,self.corr_step_size_z,back_factor * self.sequence_samples)[0]
        
        for i in range(0,self.steps_z):
            x_sig = np.concatenate((x_sig, back_ramp,ltr_ramp))
            z_sig = np.concatenate((z_sig, new_value+z_ramp, new_value*np.ones(row_samples)))
            new_value = new_value + self.corr_step_size_z      
            


        self.samples_per_line = row_samples + back_factor*self.sequence_samples
        
        # Assign x_sig
        self.sig_dict['x_sig'] =  x_sig
        # Assign y_sig
        self.sig_dict['z_sig'] =  z_sig
        print("max Z:",max(z_sig),"min Z:",min(z_sig),"for x min max:",min(x_sig),max(x_sig))
        
class PixelCycle():
    
    def __init__(self, sample_rate):
        """contains the digital signals for the pixel cycle, ie the process repeated for the acquisition of each pixel. 
        The update function takes a parameter_values dict and updates the signal accordingly."""
        self.sig_dict = {'TISsig': [], '355sig': [], '405sig': [], '488sig': [], 'CAMsig': []}
        self.sample_rate = sample_rate
      
        
    def update(self, devices, par_values, cycle_samples):
        for device in devices:
            signal = np.zeros(cycle_samples)
            start_name = 'start'+device
            end_name = 'end'+device
            start_pos = par_values[start_name] * self.sample_rate
            start_pos = int(min(start_pos, cycle_samples - 1))
            end_pos = par_values[end_name] * self.sample_rate
            end_pos = int(min(end_pos, cycle_samples))
            signal[range(start_pos, end_pos)] = 1
            self.sig_dict[device+'sig'] = signal



        
class GraphFrame(pg.GraphicsWindow):
    """Class is child of pg.GraphicsWindow and creats the plot that plots the preview of the pulses. 
    Fcn update() updates the plot of "device" with signal "signal"  """

    def __init__(self, pixel_cycle, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.pixel_cycle = pixel_cycle
        self.plot = self.addPlot(row=1, col=0)
        self.plot.showGrid(x=False, y=False)
        self.plot_sig_dict = {'TIS': self.plot.plot(pen=pg.mkPen(100, 0, 0)),
                              '355': self.plot.plot(pen=pg.mkPen(97, 0, 97)), 
                              '405': self.plot.plot(pen=pg.mkPen(73, 0, 188)), 
                              '488': self.plot.plot(pen=pg.mkPen(0, 247, 255)), 
                              'CAM': self.plot.plot(pen='w')}
        
    def update(self, devices):
        
        for device in devices:
            signal = self.pixel_cycle.sig_dict[device+'sig']
            self.plot_sig_dict[device].setData(signal)      
        

class ImageDisplay(QtGui.QWidget):
    """Class creating a display for the images obtained either with an APD or PMT"""
    def __init__(self,main,shape):
        """Main: scanWidget module
        shape: tuple, corresponds to the targeted shape of the image"""
        super().__init__()
        
        self.setWindowTitle("Image from scanning") 
        
        self.array=np.zeros((shape[1],shape[0]))
        self.shape=(shape[1]-1,shape[0]-1)
        self.pos=[0,0]
        self.scanWidget=main

        self.title=QtGui.QLabel()
        self.title.setText("Image from stage scanning")
        self.title.setStyleSheet("font-size:18px")
           

        #File management
        self.initialDir = r"C:\Users\aurelien.barbotin\Documents\Data\DefaultDataFolder"
        self.saveButton = QtGui.QPushButton("Save image")
        self.saveButton.clicked.connect(self.saveImage)
        
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        self.browseButton =  QtGui.QPushButton("Choose folder")
        self.browseButton.clicked.connect(self.loadFolder)
        
            #Visualisation widget
        self.graph = pg.GraphicsLayoutWidget()
        self.vb = self.graph.addPlot()
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.img.setImage(self.array)
        
        self.isTurning = False
        
        layout=QtGui.QGridLayout()
        self.setLayout(layout)
        
        layout.addWidget(self.title,0,0)
        layout.addWidget(self.graph,1,0,5,5)
        layout.setRowMinimumHeight(1,300)
        layout.addWidget(self.saveButton,6,1)
        layout.addWidget(self.browseButton,6,2)
        layout.addWidget(self.folderEdit,6,0)
        
        
    def update_parameters(self,shape):
        """reshapes the array with the proper dimensions before acquisition"""
        self.array=np.zeros((shape[1],shape[0]))
        self.shape=(shape[0]-1,shape[1]-1)
        self.pos=[0,0]
        self.img.setImage(self.array)
    def changePMTsensitivity(self,value):
        self.pmt_sensitivity = value/100
        self.pmt_value_line.setText(str(self.pmt_sensitivity))
        
    def loadFolder(self):
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root, initialdir=self.initialDir)
            root.destroy()
            if(folder != ''):
                self.folderEdit.setText(folder)
        except:
            pass
        
    def saveImage(self):
        im = Image.fromarray(self.array)
        
#        scan_par_values = {'width': float(self.widthPar.text()),
#                           'height': float(self.heightPar.text()),
#                           'sequence_time': float(self.sequence_timePar.text()),
#                           'frames': '-',
#                           'step_size': float(self.step_sizePar.text())}
        
        width = self.scanWidget.scan_par_values["width"]
        height = self.scanWidget.scan_par_values["width"]
        sequence_time = self.scanWidget.scan_par_values["sequence_time"]
        step_size = self.scanWidget.scan_par_values["step_size"]
        
        now = datetime.datetime.now()
        instant_string= str(now.day) +"_" + str(now.month) +"_" +str(now.hour) +"h"+ str(now.minute)+"_"+str(now.second)+"s_"
        name = instant_string + "fov"+str(width)+"x"+str(height)+"um_seqtime_"+str(sequence_time)+"s_step_size"+str(step_size)+".tif"
        im.save(self.folderEdit.text()+"\\"+name)
            
    def set_pixel_value(self,val):
        """sets the value of one pixel from an input array"""
        if not self.isTurning:
            if(self.pos[1]%2==0):
                self.array[self.pos[0],self.pos[1]]=val
            else:
                self.array[self.shape[0]-self.pos[0],self.pos[1]]=val
            self.pos[0]+=1
            if(self.pos[0]>self.shape[0]):
                self.pos[0]=0
                self.pos[1]+=1
                self.pos[1]=self.pos[1]%(self.shape[1]+1)
                self.isTurning = True
            self.img.setImage(self.array)
        else:
            self.isTurning = False
            
    def set_line_value(self,line):
        line = np.asarray(line)
        if(self.pos[1]%2==0):
            self.array[:,self.pos[1]]=line                
        else:
#            self.array[self.pos[0],:]=line[::-1]
            self.array[:,self.pos[1]]=line               
            
        self.pos[1]+=1
        self.pos[1]=self.pos[1]%(self.shape[0]+1)
            
        self.img.setImage(self.array)
           
class RecordingThread(QtCore.QThread):
    """Thread recording an image with an APD (Counter input) while the stage is scanning"""
    def __init__(self,display):
        """display: displaying window in scanwidget (self.display)"""
        super().__init__()
        self.imageDisplay = display

        
    def setParameters(self,n_frames,samples_per_channel,sample_rate):
        """prepares the thread for data acquisition with the different parameters values"""
        self.samples_per_frames = round(samples_per_channel/n_frames)
        self.n_frames = n_frames
        self.samples_in_scan = samples_per_channel
        self.rate = sample_rate    
        if(self.rate != sample_rate * self.samples_in_scan / samples_per_channel):
            print("error arrondi")
            
        print("parameters for acquisition of data : sample rate",self.rate,"samples_per_channel:",self.samples_in_scan)
        self.citask = nidaqmx.CounterInputTask()
        self.citask.create_channel_count_edges("Dev1/ctr0", init=0 )
        self.citask.set_terminal_count_edges("Dev1/ctr0","PFI12")
        self.citask.configure_timing_sample_clock(source = r'ao/SampleClock',
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan)
    def run(self):
        """runs this thread to acquire the data in task"""
        self.citask.start()
        for u in range(self.n_frames):
            data=self.citask.read(samples_per_channel = self.samples_per_frames )
            val = (data[-1]-data[0])/(self.samples_per_frames)
            self.imageDisplay.set_pixel_value(val)
            
#        self.stop()
        
    def stop(self):
        self.citask.stop()
        self.citask.clear()
        del self.citask
        
    
class RecordingThreadPMT(QtCore.QThread):
    """Thread to record an image with the PMT in conjonction with the stage scanning"""
    def __init__(self,main):
        """
        main is the corresponding imageDisplay
        """
        super().__init__()
        self.imageDisplay=main
        self.exiting = False
        self.delay=int(self.imageDisplay.scanWidget.delay.text())
        print("delay:",self.delay,type(self.delay))
        
    def setParameters(self,sequence_samples,samples_per_channel,sample_rate,samples_per_line):
        """prepares the thread for data acquisition with the different parameters values"""
        self.samples_per_line = samples_per_line
        self.sequence_samples = sequence_samples
        
        self.steps_per_line = samples_per_line / sequence_samples

        self.n_frames=round(samples_per_channel / samples_per_line)      
        
        self.samples_in_scan = samples_per_channel
        self.rate = sample_rate    
        if(self.rate != sample_rate * self.samples_in_scan / samples_per_channel):
            print("error arrondi")
            
        print("parameters for acquisition of data : sample rate",self.rate,"samples_per_channel:",self.samples_in_scan)
        self.aitask = nidaqmx.AnalogInputTask()
        self.aitask.create_voltage_channel('Dev1/ai0', terminal = 'rse', min_val=-10, max_val=10.0)
        self.aitask.configure_timing_sample_clock(source = r'ao/SampleClock',
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan+self.delay)
        self.aitask.configure_trigger_digital_edge_start(reference_trigger)
                                             
                      
    
    def run(self):
        """runs this thread to acquire the data in task"""
        self.aitask.start()
        self.aitask.read(self.delay,fill_mode='group_by_channel',timeout=30)[0,:]   #To synchronize analog input and output
        counter = self.n_frames
        result = np.zeros((self.steps_per_line-1,self.n_frames))
        print("n samples:",self.sequence_samples)
        while(counter>0 and not self.exiting):
            data=self.aitask.read(self.samples_per_line , fill_mode='group_by_channel',timeout=10)[0,:]

            data = np.split(data,self.steps_per_line)   #Transforms into a list of arrays corresponding to 1 pixel each
#            if self.sequence_samples>10:
#                if counter%2==0:
#                    data = [np.sum((x[0:10])) for x in data]
#                else:
#                    data = [np.sum(x[-10:]) for x in data]
            
            data = [np.sum(x) for x in data]
            for u in range(int(self.imageDisplay.scanWidget.back_factor_param.text())):
                data=data[1:]  #Remove first elements corresponding to repositionning
            self.emit(QtCore.SIGNAL("line(PyQt_PyObject)"),data)
            counter-=1
#            if counter%2==0:
#                result[:,counter]=data
#            else:
#                result[:,counter]=data[::-1]
#        print("start run")
#        data=self.aitask.read(self.samples_per_line*self.n_frames , fill_mode='group_by_channel',timeout=100)[0,:]
#        print("end read")
#        lines = np.split(data,self.n_frames)
#        for u in range(self.n_frames):
#            temp = np.split(lines[u],self.steps_per_line)
#            temp = [np.sum(x) for x in temp]
#            temp=temp[:-1]
#            if u%2==0:
#                result[:,u]=np.asarray(temp)
#            else:
#                result[:,u]=np.asarray(temp)[::-1]
#
#        self.imageDisplay.img.setImage(result)
        print(self.aitask.get_read_current_position(),"current pos")
        self.aitask.stop()
        
    def stop(self):
        try:
            self.aitask.stop()
            self.aitask.clear()
            del self.aitask
        except:
            pass
            
class Positionner(QtGui.QWidget):
    """This class communicates with the different analog outputs of the nidaq card. When not scanning, it drives the 3 axis 
    x, y and z as well as the PMT sensitivity"""
    
    def __init__(self,main):
        """main:scanWidget"""
        super().__init__()
        self.scanWidget = main
        
        #Position of the different devices in V
        self.x=0
        self.y=0
        self.z=0
        self.pmt_sensitivity = 0.2
        
        #Parameters for the ramp (driving signal for the different channels)
        self. ramp_time = 800    #Time for each ramp in ms
        self.sample_rate = 10**5
        self.n_samples = int(self.ramp_time * 10**-3 * self.sample_rate)
        
        #This boolean is set to False when tempesta is scanning to prevent this positionner to
        #access the analog output channels
        self.isActive = True    
        
        #PMT control        
        self.pmt_value_line = QtGui.QLineEdit()      
        self.pmt_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.pmt_slider.valueChanged.connect(self.changePMTsensitivity)
        self.pmt_slider.setRange(0,125)
        self.pmt_slider.setTickInterval(10)
        self.pmt_slider.setTickPosition(QtGui.QSlider.TicksBothSides)
        self.pmt_slider.setValue(self.pmt_sensitivity * 100 )
        
        self.pmt_value_line.setText(str(self.pmt_sensitivity))
        
        self.pmt_minVal = QtGui.QLabel("0")
        self.pmt_maxVal= QtGui.QLabel("1.25")
        self.pmt_sliderLabel = QtGui.QLabel("PMT sensitivity")      
        
        #Creating the analog output tasks
        self.sensitivityTask = nidaqmx.AnalogOutputTask()
        self.sensitivityTask.create_voltage_channel( pmt_sensitivity_channel , min_val=0, max_val=1.25)
        self.sensitivityTask.start()   
        self.sensitivityTask.write(self.pmt_sensitivity,auto_start=True) 
        
        self.aoTask = nidaqmx.AnalogOutputTask("positionnerTask")
        
        xchan="Dev1/ao"+str(self.scanWidget.current_aochannels["x"])
        self.aoTask.create_voltage_channel( xchan, channel_name="x" , min_val=min_ao_horizontal, max_val= max_ao_horizontal)
        

        ychan="Dev1/ao"+str(self.scanWidget.current_aochannels["y"])
        self.aoTask.create_voltage_channel( ychan , channel_name="y",min_val=min_ao_horizontal, max_val= max_ao_horizontal)
        
        zchan="Dev1/ao"+str(self.scanWidget.current_aochannels["z"])
        self.aoTask.create_voltage_channel( zchan, channel_name="z" , min_val=min_ao_vertical, max_val= max_ao_vertical)
        
        self.aoTask.configure_timing_sample_clock(rate = self.sample_rate,sample_mode="finite",samples_per_channel=self.n_samples)
        self.aoTask.start()
        
        #Axes control
        self.x_value_line = QtGui.QLineEdit()   
        self.x_value_line.setText(str(self.x))
        self.x_value_line.editingFinished.connect(self.editx)
        self.x_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.x_slider.sliderReleased.connect(self.move_x)
        self.x_slider.setRange(100*min_ao_horizontal , 100 * max_ao_horizontal)
        self.x_slider.setValue(self.x)
        self.x_minVal = QtGui.QLabel("-37.5")
        self.x_maxVal= QtGui.QLabel("37.5")
        self.x_sliderLabel = QtGui.QLabel("x position(µm)")      
        
        self.y_value_line = QtGui.QLineEdit()      
        self.y_value_line.setText(str(self.y))
        self.y_value_line.editingFinished.connect(self.edity)
        self.y_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.y_slider.sliderReleased.connect(self.move_y)
        self.y_slider.setRange(100*min_ao_horizontal , 100 * max_ao_horizontal)
        self.y_slider.setValue(self.y)
        self.y_minVal = QtGui.QLabel("-37.5")
        self.y_maxVal= QtGui.QLabel("37.5")
        self.y_sliderLabel = QtGui.QLabel("y position(µm)")   
        
        self.z_value_line = QtGui.QLineEdit()      
        self.z_value_line.setText(str(self.z))
        self.z_value_line.editingFinished.connect(self.editz)
        self.z_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.sliderReleased.connect(self.move_z)
        self.z_slider.setRange(100*min_ao_vertical , 100 * max_ao_vertical)
        self.z_slider.setValue(self.z)
        self.z_minVal = QtGui.QLabel("0")
        self.z_maxVal= QtGui.QLabel("100")
        self.z_sliderLabel = QtGui.QLabel("z position(µm)")   
        
        
#        QtCore.QObject.connect(self,self.z_slider.mouseReleaseEvent, self.move_z)
#        self.z_slider.sliderReleased.connect(self.test)
#        self.z_slider.mouseReleaseEvent().connect(self.move_y)
        
        self.title=QtGui.QLabel()
        self.title.setText("Stage Positionner")
        self.title.setStyleSheet("font-size:18px")
        
        layout = QtGui.QGridLayout()        
        self.setLayout(layout)
        layout.addWidget(self.title,0,9)
        layout.addWidget(self.pmt_sliderLabel,0,0,1,1)
        layout.addWidget(self.pmt_minVal,1,0,1,1)
        layout.addWidget(self.pmt_maxVal,1,2,1,1)
        layout.addWidget(self.pmt_slider,2,0,1,3)
        layout.addWidget(self.pmt_value_line,2,3,1,1)
                
        layout.addWidget(self.z_sliderLabel,0,4,1,1)
        layout.addWidget(self.z_minVal,1,4,1,1)
        layout.addWidget(self.z_maxVal,1,6,1,1)
        layout.addWidget(self.z_slider,2,4,1,3)
        layout.addWidget(self.z_value_line,2,7,1,1)
        
        layout.addWidget(self.x_sliderLabel,3,0,1,1)
        layout.addWidget(self.x_minVal,4,0,1,1)
        layout.addWidget(self.x_maxVal,4,2,1,1)
        layout.addWidget(self.x_slider,5,0,1,3)
        layout.addWidget(self.x_value_line,5,3,1,1)
        
        layout.addWidget(self.y_sliderLabel,3,4,1,1)
        layout.addWidget(self.y_minVal,4,4,1,1)
        layout.addWidget(self.y_maxVal,4,6,1,1)
        layout.addWidget(self.y_slider,5,4,1,3)
        layout.addWidget(self.y_value_line,5,7,1,1)
        

            
    def move(self):
        """moves the 3 axis and the PMT to the positions specified by the sliders"""
        if self.isActive:
            val_x = self.x_slider.value()/100
            if self.x != val_x:
                signalx = make_ramp(self.x,val_x,self.n_samples)[0]
                self.x = val_x
            else:
                signalx = np.ones(self.n_samples) * self.x
    
            val_y = self.y_slider.value()/100
            if self.y != val_y:
                signaly = make_ramp(self.y,val_y,self.n_samples)[0]
                self.y= val_y           
            else:
                signaly = np.ones(self.n_samples) * self.y
                
            val_z = self.z_slider.value()/100
            if self.z != val_z:
                signalz = make_ramp(self.z,val_z,self.n_samples)[0]

                self.z = val_z
            else:
                signalz = np.ones(self.n_samples) * self.z   
                

            full_signal = np.append(signalx,signaly)
            full_signal=np.append(full_signal,signalz)
  
            self.aoTask.write(full_signal,layout = 'group_by_channel',auto_start=True)
        else:
            pass
            
    def changePMTsensitivity(self):
        value = self.pmt_slider.value()/100
        self.pmt_value_line.setText(str(value))
                    
        if self.pmt_sensitivity != value:
            signalpmt = make_ramp(self.pmt_sensitivity,value,self.n_samples)[0]
            self.sensitivityTask.write(signalpmt)
            self.pmt_sensitivity = value
        
    def move_x(self):
        value = self.x_slider.value()/100
        self.x_value_line.setText(str(round(value*x_factor,2)))
        print("move x")
        self.move()
    def move_y(self):
        value = self.y_slider.value()/100
        self.y_value_line.setText(str(round(value*y_factor,2)))
        self.move()
    def move_z(self):
        value = self.z_slider.value()/100
        self.z_value_line.setText(str(round(value*z_factor,2)))
        self.move()
    
    def editx(self):
        self.x_slider.setValue(100*float(self.x_value_line.text()) / x_factor)
        self.move()
    def edity(self):
        self.y_slider.setValue(100*float(self.y_value_line.text()) / y_factor)
        self.move()
    def editz(self):
        self.z_slider.setValue(100*float(self.z_value_line.text()) / z_factor)    
        self.move()
        
    def go_to_zero(self):
        self.x=0
        self.y=0
        self.z=0
        self.pmt_sensitivity=0
        self.move()
        return

    def reset_channels(self):
        """Method called when the analog output channels need to be used by another resource."""
        if(self.isActive):
            print("disabling channels")
            self.aoTask.stop()
            self.aoTask.clear()
            del self.aoTask
            self.isActive = False
            
        else:
                        #Restarting the analog channels
            print("restarting channels")
            self.sensitivityTask = nidaqmx.AnalogOutputTask()
            self.sensitivityTask.create_voltage_channel( pmt_sensitivity_channel , min_val=0, max_val=1.25)
            self.sensitivityTask.start()   
            self.sensitivityTask.write(self.pmt_sensitivity,auto_start=True) 
            
            self.aoTask = nidaqmx.AnalogOutputTask("positionnerTask")
            
            xchan="Dev1/ao"+str(self.scanWidget.current_aochannels["x"])
            self.aoTask.create_voltage_channel( xchan, channel_name="x" , min_val=min_ao_horizontal, max_val= max_ao_horizontal)
            
    
            ychan="Dev1/ao"+str(self.scanWidget.current_aochannels["y"])
            self.aoTask.create_voltage_channel( ychan , channel_name="y",min_val=min_ao_horizontal, max_val= max_ao_horizontal)
            
            zchan="Dev1/ao"+str(self.scanWidget.current_aochannels["z"])
            self.aoTask.create_voltage_channel( zchan, channel_name="z" , min_val=min_ao_vertical, max_val= max_ao_vertical)
            
            self.aoTask.configure_timing_sample_clock(rate = self.sample_rate,sample_mode="finite",samples_per_channel=self.n_samples)
            self.aoTask.start()
            self.isActive = True
            print("in reset:",self.aoTask)
            
    def closeEvent(self,*args,**kwargs):
        if(self.isActive):
            #Resets the sliders, which will reset each channel to 0
            print("closeEvent positionner")
            self.pmt_slider.setValue(0)
            self.x_slider.setValue(0)
            self.y_slider.setValue(0)
            self.z_slider.setValue(0)
            self.move()
            self.aoTask.wait_until_done(timeout=2)
            self.aoTask.stop()
            self.aoTask.clear()
            self.sensitivityTask.stop()
            self.sensitivityTask.clear()
        
            

        
def make_ramp(start, end, samples):

    ramp = []
    k  = (end - start) / (samples - 1)
    for i in range(0, samples):
        ramp.append(start + k * i)
        
    return np.array([np.asarray(ramp), k])
    
    
def distance_to_voltage_Y(D_signal):
    a1 = 0.6524
    a2 = -0.0175
    a3 = 0.0004
    samples = len(D_signal)
    V_signal = np.zeros(samples)
    now = time.time()
    D_value = 15
    for i in range(0, samples):
        D_value = D_signal[i]
        V_value = a1*D_value + a2*D_value**2 + a3*np.power(D_value, 3)
        V_signal[i] = V_value
        
    elapsed = time.time() - now
    print('Elapsed time: ', elapsed)
    return V_signal
        
def distance_to_voltage_X(D_signal):
    a1 = 0.6149
    a2 = -0.0146
    a3 = 0.0003
    samples = len(D_signal)
    V_signal = np.zeros(samples)
    now = time.time()
    D_value = 15
    for i in range(0, samples):
        D_value = D_signal[i]
        V_value = a1*D_value + a2*D_value**2 + a3*np.power(D_value, 3)
        V_signal[i] = V_value
        
    elapsed = time.time() - now
    print('Elapsed time: ', elapsed)
    return V_signal
    
    
#    
#if __name__ == '__main__':
#    ScanWid = ScanWidget(nidaqmx.Device('Dev1'))
##    ScanWid.update_Scan()
