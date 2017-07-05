# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
"""

try:
    import libnidaqmx
except:
    from control import libnidaqmx
    
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
import time
from numpy import arange
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

import control.guitools as guitools




# This class is intended as a widget in the bigger GUI, Thus all the commented parameters etc. It contain an instance
# of stage_scan and pixel_scan which in turn harbour the analog and digital signals respectively.
# The function run is the function that is supposed to start communication with the Nidaq
# through the Scanner object. This object was initially written as a QThread object but is not right now. 
# As seen in the commened lines of run() I also tried running in a QThread created in run().
# The rest of the functions contain mosly GUI related code.

class ScanWidget(QtGui.QMainWindow):
    
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.aochannels = device.get_analog_output_channels()
        self.dochannels = device.get_digital_output_lines()
        self.all_devices = ['TIS', '355', '405', '488', 'CAM']
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        self.saveScanBtn = QtGui.QPushButton('Save Scan')
        saveScanFcn = lambda: guitools.saveScan(self)
        self.saveScanBtn.clicked.connect(saveScanFcn) 
        self.loadScanBtn = QtGui.QPushButton('Load Scan')
        loadScanFcn = lambda: guitools.loadScan(self)
        self.loadScanBtn.clicked.connect(loadScanFcn)      
        
        self.sampleRateEdit = QtGui.QLineEdit()
        
        self.size_xPar = QtGui.QLineEdit('10')
        self.size_xPar.editingFinished.connect(lambda: self.ScanParameterChanged('size_x'))
        self.size_yPar = QtGui.QLineEdit('10')
        self.size_yPar.editingFinished.connect(lambda: self.ScanParameterChanged('size_y'))
        self.size_zPar = QtGui.QLineEdit('10')
        self.size_zPar.editingFinished.connect(lambda: self.ScanParameterChanged('size_z'))
        self.sequence_timePar = QtGui.QLineEdit('100') # Milliseconds
        self.sequence_timePar.editingFinished.connect(lambda: self.ScanParameterChanged('sequence_time'))
        self.nrFramesPar = QtGui.QLabel()
        self.scanDuration = QtGui.QLabel()
        self.step_sizeXYPar = QtGui.QLineEdit('0.5')
        self.step_sizeXYPar.editingFinished.connect(lambda: self.ScanParameterChanged('step_sizeXY'))
        self.step_sizeZPar = QtGui.QLineEdit('0.5')
        self.step_sizeZPar.editingFinished.connect(lambda: self.ScanParameterChanged('step_sizeZ'))
        self.sample_rate = 100000
#        self.sample_rate = np.float(self.sampleRateEdit.text())
        
        self.Scan_Mode_label= QtGui.QLabel('Scan mode:')        
        self.Scan_Mode = QtGui.QComboBox()
        self.scan_modes = ['FOV scan', 'VOL scan', 'Line scan']
        self.Scan_Mode.addItems(self.scan_modes)
        self.Scan_Mode.currentIndexChanged.connect(lambda: self.setScanMode(self.Scan_Mode.currentText()))
        
        self.prim_scan_dim_label= QtGui.QLabel('Primary scan dim:')          
        self.prim_scan_dim = QtGui.QComboBox()
        self.scan_dims = ['x', 'y']
        self.prim_scan_dim.addItems(self.scan_dims)
        self.prim_scan_dim.currentIndexChanged.connect(lambda: self.setPrimScanDim(self.prim_scan_dim.currentText()))
        
        self.scan_parameters = {'size_x': self.size_xPar, 
                                'size_y': self.size_yPar,
                                'size_z': self.size_zPar,
                                'sequence_time': self.sequence_timePar,
                                'step_sizeXY': self.step_sizeXYPar,
                                'step_sizeZ': self.step_sizeZPar}

        self.scan_par_values = {'size_x': float(self.size_xPar.text()),
                           'size_y': float(self.size_yPar.text()),
                           'size_z': float(self.size_zPar.text()),
                           'sequence_time': float(self.sequence_timePar.text())/1000,
                           'step_sizeXY': float(self.step_sizeXYPar.text()), 
                           'step_sizeZ': float(self.step_sizeZPar.text())}
                           
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

        
        self.pixel_par_values = {'startTIS': float(self.startTISPar.text())/1000,
                                 'start355': float(self.start355Par.text())/1000,
                                 'start405': float(self.start405Par.text())/1000,
                                 'start488': float(self.start488Par.text())/1000,
                                 'startCAM': float(self.startCAMPar.text())/1000,
                                 'end488': float(self.end488Par.text())/1000,
                                 'end405': float(self.end405Par.text())/1000,
                                 'end355': float(self.end355Par.text())/1000,
                                 'endTIS': float(self.endTISPar.text())/1000,
                                 'endCAM': float(self.endCAMPar.text())/1000}
        
        self.current_dochannels = {'TIS': 0, '355': 1, '405': 2, '488': 3, 'CAM': 4}
        self.current_aochannels = {'x': 0, 'y': 1, 'z': 2}
        self.XchanPar = QtGui.QComboBox()
        self.XchanPar.addItems(self.aochannels)
        self.XchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
        self.YchanPar = QtGui.QComboBox()
        self.YchanPar.addItems(self.aochannels)
        self.YchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
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
    
        self.stage_scan = StageScan(self.sample_rate)
        self.pixel_cycle = PixelCycle(self.sample_rate)
        self.graph = GraphFrame(self.pixel_cycle)
        self.update_Scan(self.all_devices)
        
        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(lambda: self.setScanOrNot(False))
        
        
        self.ScanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.ScanButton.clicked.connect(self.ScanOrAbort)
        self.PreviewButton = QtGui.QPushButton('Preview')
        self.PreviewButton.clicked.connect(self.PreviewScan)
        
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)     
        
        
        
#        grid.setRowMaximumHeight(0, 20)
#        grid.setRowMamimumHeight(1, 20)
        grid.addWidget(self.loadScanBtn, 0 , 0)
        grid.addWidget(self.saveScanBtn, 0 , 1)
#        grid.addWidget(QtGui.QLabel('X channel'), 1, 6)
#        grid.addWidget(self.XchanPar, 1, 5)
        grid.addWidget(QtGui.QLabel('Size X (um):'), 2, 0, 1, 1)
        grid.addWidget(self.size_xPar, 2, 1, 2, 1)
        grid.addWidget(QtGui.QLabel('Size Y (um):'), 2, 2, 1, 1)
        grid.addWidget(self.size_yPar, 2, 3, 2, 1)
        grid.addWidget(QtGui.QLabel('Size Z (um):'), 2, 4, 1, 1)
        grid.addWidget(self.size_zPar, 2, 5, 2, 1)
#        grid.addWidget(QtGui.QLabel('Y channel'), 2, 6)
#        grid.addWidget(self.YchanPar, 2, 5)
        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 4, 0)
        grid.addWidget(self.sequence_timePar, 4, 1)
        grid.addWidget(QtGui.QLabel('Frames in scan:'), 4, 2)
        grid.addWidget(self.nrFramesPar, 4, 3)
        grid.addWidget(QtGui.QLabel('Step size XY (um):'), 4, 4)
        grid.addWidget(self.step_sizeXYPar, 4, 5)
        grid.addWidget(QtGui.QLabel('Step size Z (um):'), 5, 4)
        grid.addWidget(self.step_sizeZPar, 5, 5)
        grid.addWidget(QtGui.QLabel('Scan duration (s):'), 6, 4)
        grid.addWidget(self.scanDuration, 6, 5)
        grid.addWidget(QtGui.QLabel('Start (ms):'), 7, 1)
        grid.addWidget(QtGui.QLabel('End (ms):'), 7, 2)
        grid.addWidget(QtGui.QLabel('TIS:'), 8, 0)
        grid.addWidget(self.startTISPar, 8, 1)
        grid.addWidget(self.endTISPar, 8, 2)
        grid.addWidget(self.chanTISPar, 8, 3)
        grid.addWidget(QtGui.QLabel('488 OFF::'), 9, 0)
        grid.addWidget(self.start355Par, 9, 1)
        grid.addWidget(self.end355Par, 9, 2)
        grid.addWidget(self.chan355Par, 9, 3)
        grid.addWidget(self.scanRadio, 9, 4, 2, 1)
        grid.addWidget(self.Scan_Mode_label, 8, 5)
        grid.addWidget(self.Scan_Mode, 9, 5)
        grid.addWidget(self.prim_scan_dim_label, 10, 5)
        grid.addWidget(self.prim_scan_dim, 11, 5)
        grid.addWidget(QtGui.QLabel('405:'), 10, 0)
        grid.addWidget(self.start405Par, 10, 1)
        grid.addWidget(self.end405Par, 10, 2)
        grid.addWidget(self.chan405Par, 10, 3)
        grid.addWidget(self.contLaserPulsesRadio,11, 4, 2, 1)
        grid.addWidget(QtGui.QLabel('488 EX:'), 11, 0)
        grid.addWidget(self.start488Par, 11, 1)
        grid.addWidget(self.end488Par, 11, 2)
        grid.addWidget(self.chan488Par, 11, 3)
        grid.addWidget(QtGui.QLabel('CAM:'), 12, 0)
        grid.addWidget(self.startCAMPar, 12, 1)
        grid.addWidget(self.endCAMPar, 12, 2)
        grid.addWidget(self.chanCAMPar, 12, 3)
        grid.addWidget(self.graph, 13, 0, 1, 6)
        grid.addWidget(self.ScanButton, 14, 3)
        grid.addWidget(self.PreviewButton, 14, 4)
        
    @property
    def scanOrNot(self):
        return self._scanOrNot

    @scanOrNot.setter
    def scanOrNot(self, value):
        self.EnableScanPars(value)
        self.ScanButton.setCheckable(not value)
        
    def EnableScanPars(self, value):
        self.size_xPar.setEnabled(value)
        self.size_yPar.setEnabled(value)
        self.step_sizeXYPar.setEnabled(value)
        self.Scan_Mode.setEnabled(value)
        self.prim_scan_dim.setEnabled(value)
        if value:
            self.ScanButton.setText('Scan')
        else:
            self.ScanButton.setText('Run')
    
    def setScanOrNot(self, value):
        self.scanOrNot = value 
    
    def setScanMode(self, mode):             
            
        self.stage_scan.set_scan_mode(mode)
        self.ScanParameterChanged('scan_mode')
        
    def setPrimScanDim(self, dim):             
            
        self.stage_scan.set_prim_scan_dim(dim)
        self.ScanParameterChanged('prim_scan_dim')
        
        
    def AOchannelsChanged (self):
        """Function is obsolete since we never change channels this way, Z-channel not implemented"""
#        Xchan = self.XchanPar.currentIndex()
#        Ychan = self.YchanPar.currentIndex()
#        if Xchan == Ychan:
#            Ychan = (Ychan + 1)%4
#            self.YchanPar.setCurrentIndex(Ychan)
#        self.current_aochannels['x'] = Xchan
#        self.current_aochannels['y'] = Ychan
        
    def DOchannelsChanged(self, sig, new_index):
        
        for i in self.current_dochannels:
            if i != sig and new_index == self.current_dochannels[i]:
                self.DOchan_Pars_dict[sig].setCurrentIndex(self.current_dochannels[sig])
        
        self.current_dochannels[sig] = self.DOchan_Pars_dict[sig].currentIndex()
        
        
    def ScanParameterChanged(self, parameter):
        if not parameter in ('scan_mode', 'prim_scan_dim'):
            if parameter == 'sequence_time':
                self.scan_par_values[parameter] = float(self.scan_parameters[parameter].text())/1000 #To get in seconds
            else:    
                self.scan_par_values[parameter] = float(self.scan_parameters[parameter].text())
            
        if parameter == 'sequence_time':
            self.update_Scan(self.all_devices)
            self.graph.update(self.all_devices)
        self.stage_scan.update_frames(self.scan_par_values)
        self.nrFramesPar.setText(str(self.stage_scan.frames))
        self.scanDuration.setText(str((1/1000)*self.stage_scan.frames*float(self.sequence_timePar.text())))
                
    def PixelParameterChanged(self, parameter):
        self.pixel_par_values[parameter] = float(self.pixel_parameters[parameter].text()) / 1000
        device = parameter[-3]+parameter[-2]+parameter[-1]
        self.pixel_cycle.update([device], self.pixel_par_values, self.stage_scan.sequence_samples) 
        self.graph.update([device])
        
    def PreviewScan(self):
        
        self.stage_scan.update(self.scan_par_values)
        x_sig = self.stage_scan.sig_dict['x_sig']
        y_sig = self.stage_scan.sig_dict['y_sig']
        plt.plot(x_sig, y_sig)
        plt.axis([-0.2,self.scan_par_values['size_x']+0.2, -0.2, self.scan_par_values['size_y']+0.2])
        
    def ScanOrAbort(self):
        if not self.scanning:
            self.PrepAndRun()
        else:
            self.scanner.abort()
    
    #PrepAndRun is only called if scanner is not running (See ScanOrAbort funciton)
    def PrepAndRun(self):

        if self.scanRadio.isChecked():
            self.stage_scan.update(self.scan_par_values)
            self.ScanButton.setText('Abort')
            self.scanner = Scanner(self.nidaq, self.stage_scan, self.pixel_cycle, self.current_aochannels, self.current_dochannels, self)
            self.scanner.finalizeDone.connect(self.FinalizeDone)    
            self.scanner.scanDone.connect(self.ScanDone)            
            self.scanning = True
            self.scanner.runScan()
            
        elif self.ScanButton.isChecked():
#            self.lasercycle = LaserCycle(self.pixel_cycle, self.current_dochannels)
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
        try:
            self.scanner.waiter.terminate()
        except:
            pass
        

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
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.stage_scan = stage_scan
        self.pixel_cycle = pixel_cycle
        self.current_aochannels = current_aochannels  # Dict containing channel numbers to be written to for each signal
        self.current_dochannels = current_dochannels  # Dict containing channel numbers to be written to for each device
        self.samples_in_scan = len(self.stage_scan.sig_dict['x_sig'])
        self.main = main
        
        self.aotask = libnidaqmx.AnalogOutputTask('aotask')
        self.dotask = libnidaqmx.DigitalOutputTask('dotask')         
        
        self.waiter = Wait_Thread(self.aotask)
        
        self.warning_time= 10
        self.scantimewar = QtGui.QMessageBox()
        self.scantimewar.setInformativeText("Are you sure you want to continue?")
        self.scantimewar.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
    
        
    def finalize(self):
        print('in finalize')
        self.scanDone.emit()
        self.waiter.waitdoneSignal.disconnect(self.finalize) #Apparently important, otherwise finalize is called again when next waiting finishes.
        self.waiter.waitdoneSignal.connect(self.done)
        written_samps = self.aotask.get_samples_per_channel_generated()
        
#        Following code should correct channels mentioned in Buglist. Not correct though, assumes channels are 0, 1 and 2.
        final_samps = [0, 0, 0]
        temp_aochannels = copy.copy(self.current_aochannels)
        for i in range(0,3):
            dim = min(temp_aochannels, key = temp_aochannels.get)   # dim = dimension ('x', 'y' or 'z') containing smallest channel nr. 
            final_samps[i] = self.stage_scan.sig_dict[dim+'_sig'][written_samps - 1]
            temp_aochannels.pop(dim)
                 
        
        return_ramps = np.array([])
        for i in range(0,3):
            ramp_and_k = make_ramp(final_samps[i], 0, self.stage_scan.sample_rate)
            return_ramps = np.append(return_ramps, ramp_and_k[0])
        
                
        
#        magic = np.ones(100)  # Seems to decrease frequency of Invalid task errors.          
        print('aotaskis: ', self.aotask)
        self.aotask.stop()    
        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                                 sample_mode = 'finite',
                                                 samples_per_channel = self.stage_scan.sample_rate)        
        
        self.aotask.write(return_ramps, layout = 'group_by_channel', auto_start = False)
        self.aotask.start()

        self.waiter.start()


    def done(self):
        print('in self.done()')
        self.aotask.clear()
        self.dotask.clear()
        self.nidaq.reset()
        self.finalizeDone.emit()
        
    def runScan(self):
        scan_time = self.samples_in_scan / self.main.sample_rate
        ret = QtGui.QMessageBox.Yes
        self.scantimewar.setText("Scan will take %s seconds" %scan_time)
        if scan_time > self.warning_time:
            ret = self.scantimewar.exec_()
            
        if ret == QtGui.QMessageBox.No:
            self.done()
            return


        full_ao_signal = []
        temp_aochannels = copy.copy(self.current_aochannels)
        min_ao = -10
        max_ao = 10
           
        # Following loop creates the voltage channels in smallest to largest order and places signals in same order.
        
        for i in range(0,3):
            dim = min(temp_aochannels, key = temp_aochannels.get)   # dim = dimension ('x' or 'y') containing smallest channel nr. 
            chanstring = 'Dev1/ao%s'%temp_aochannels[dim]
            self.aotask.create_voltage_channel(phys_channel = chanstring, channel_name = 'chan%s'%dim, min_val = min_ao, max_val = max_ao)
            temp_aochannels.pop(dim)
            signal = self.stage_scan.sig_dict[dim+'_sig']
            full_ao_signal = np.append(full_ao_signal, signal)
#            final_samps = np.append(final_samps, signal[-1])

        
        # Same as above but for the digital signals/devices        
        
        full_do_signal = []
        temp_dochannels = copy.copy(self.current_dochannels)
        for i in range(0,len(temp_dochannels)):
            dev = min(temp_dochannels, key = temp_dochannels.get)
            chanstring = 'Dev1/port0/line%s'%temp_dochannels[dev]
            self.dotask.create_channel(lines = chanstring, name = 'chan%s'%dev)
            temp_dochannels.pop(dev)
            
            if self.stage_scan.scan_mode == 'VOL_scan':
                
                
                
            else:
                signal = self.pixel_cycle.sig_dict[dev+'sig']
            
            if len(full_ao_signal)%len(signal) != 0 and len(full_do_signal)%len(signal) != 0:
                print('Signal lengths does not match (printed from run)')
            full_do_signal = np.append(full_do_signal, signal)
        
        
        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                             sample_mode = 'finite',
                                             samples_per_channel = self.samples_in_scan)
                        
        self.dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                             rate = self.pixel_cycle.sample_rate, 
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan)
                        

     
        # Following is to create ramps back to zero for the analog channels during one second.     
     
#        return_ramps = np.array([])
#        for i in range(0,2):
#            ramp_and_k = make_ramp(final_samps[i], 0, self.stage_scan.sample_rate)
#            return_ramps = np.append(return_ramps, ramp_and_k[0]) 
#            
#        print(np.ones(1)) # This line alone fixes the problem...
     
        self.waiter.waitdoneSignal.connect(self.finalize)
        
        self.dotask.write(full_do_signal, layout = 'group_by_channel', auto_start = False)
        self.aotask.write(full_ao_signal, layout = 'group_by_channel', auto_start = False)

        self.dotask.start()
        self.aotask.start()

        self.waiter.start()
#        self.aotask.wait_until_done() ##Need to wait for task to finish, otherwise aotask will be deleted 
#        self.aotask.stop()
#        
#        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
#                                                 sample_mode = 'finite',
#                                                 samples_per_channel = self.stage_scan.sample_rate)
#                                                 
#                              
#                                             
#        self.aotask.write(return_ramps, layout = 'group_by_channel', auto_start = False)
#        self.aotask.start()
#        self.aotask.wait_until_done()
#
#
#        self.aotask.clear()          ## when function is finished and task aborted
#        self.dotask.clear()        

#        self.doneSignal.emit()
#
        
    def abort(self):
        print('Aborting scan')
        self.waiter.stop()
        self.aotask.stop()
        self.dotask.stop()
        self.finalize()        

class LaserCycle():
    
#    def __init__(self, pixel_cycle, curren_dochannels):
    def __init__(self, device, pixel_cycle, curren_dochannels):
        
        self.nidaq = device
        self.pixel_cycle = pixel_cycle
        self.current_dochannels = curren_dochannels
        
    def run(self):
        self.dotask = libnidaqmx.DigitalOutputTask('dotask') 
        
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
        
# Following class containg the analog signals in sig_dict. The update function takes the parameter_values
# and updates the signals accordingly


class StageScan():
    
    def __init__(self, sample_rate):
        self.scan_mode = 'FOV scan'
        self.prim_scan_dim = 'x'
        self.sig_dict = {'x_sig': [], 'y_sig': [], 'z_sig': []}
        self.sample_rate = sample_rate
        self.sequence_samples = None
        self.FOV_scan = FOV_Scan(self.sample_rate)
        self.VOL_scan = VOL_Scan(self.sample_rate)
        self.line_scan = Line_scan(self.sample_rate)
        self.scans = {'FOV scan': self.FOV_scan, 'VOL scan': self.VOL_scan, 'Line scan': self.line_scan}
        self.frames = 0
        
    def set_scan_mode(self, mode):
        self.scan_mode = mode
        
    def set_prim_scan_dim(self, dim):
        self.prim_scan_dim = dim  
        
    def update_frames(self, par_values):
        self.scans[self.scan_mode].update_frames(par_values)
        self.frames = self.scans[self.scan_mode].frames
        
    def update(self, par_values):
        print('in update stage_scan')
        print('self.scan_mode = ', self.scan_mode)
        self.scans[self.scan_mode].update(par_values, self.prim_scan_dim)
        self.sig_dict = self.scans[self.scan_mode].sig_dict
        self.sequence_samples = self.scans[self.scan_mode].sequence_samples
        self.frames = self.scans[self.scan_mode].frames
        
            
class Line_scan():
        
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': [], 'z_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
    def update_frames(self, par_values):
        size_y = par_values['size_y'] / 2
        step_size = par_values['step_sizeXY'] / 2
        steps_y = int(np.ceil(size_y / step_size))
        self.frames = steps_y + 1 # +1 because nr of frames per line is one more than nr of steps
        
    def update(self, par_values, prim_scan_dim):
        
        # Create signals
        start_x = 0
        start_y = 0
        size_y = par_values['size_y'] / 2
        sequence_samples = np.round(self.sample_rate * par_values['sequence_time'])
        step_size = par_values['step_sizeXY'] / 2
        self.steps_y = int(np.ceil(size_y / step_size))
        self.corr_step_size = size_y/self.steps_y # Step size compatible with width
        self.sequence_samples = int(sequence_samples)
        column_samples = self.steps_y * self.sequence_samples
        ramp_and_k = make_ramp(start_y, size_y, column_samples) # ramp_and_k contains [ramp, k]
        ramp = ramp_and_k[0]
        
        self.sig_dict[prim_scan_dim+'_sig'] = 1.14 * ramp
        print(ramp)
        for key in self.sig_dict:
            if not key[0] == prim_scan_dim:
                self.sig_dict[key] = np.zeros(len(ramp))
                
        
class FOV_Scan():
    
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': [], 'z_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
        # Update signals according to parameters.
        # Note that rounding floats to ints may cause actual scan to differ slighly from expected scan. 
        # Maybe either limit input parameters to numbers that "fit each other" or find other solution, eg step size has to
        # be width divided by an integer.
        # Maybe not a problem ???
        
    def update_frames(self, par_values):
        step_size = par_values['step_sizeXY'] / 2
        size_x = par_values['size_x'] / 2
        size_y = par_values['size_y'] / 2
        steps_x = int(np.ceil(size_x / step_size))
        steps_y = int(np.ceil(size_y / step_size))
        self.frames = (steps_y+1)*(steps_x+1) # +1 because nr of frames per line is one more than nr of steps
                
        
    def update(self, par_values, prim_scan_dim):
        """Changed recently to remove the "smooth" curves. I don't think they're necessary"""
        # Create signals
        start_x = 0
        start_y = 0
        size_x = par_values['size_x'] / 2
        size_y = par_values['size_y'] / 2
        step_size = par_values['step_sizeXY'] / 2
        self.sequence_samples = int(np.round(self.sample_rate * par_values['sequence_time'])) #WARNING: Correct for units of the time, now seconds!!!!
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        self.corr_step_size = size_x/self.steps_x # Step size compatible with width
        row_samples = self.steps_x * self.sequence_samples
        
        ramp_and_k = make_ramp(start_x, size_x, row_samples) # ramp_and_k contains [ramp, k]
        k = ramp_and_k[1]
        ltr_ramp = ramp_and_k[0]
        rtl_ramp = ltr_ramp[::-1]  # rtl_ramp contains only ramp, no k since same k = -k
        gradual_k = make_ramp(k, -k, self.sequence_samples)
        turn_rtl = np.cumsum(gradual_k[0])
        turn_ltr = -turn_rtl
        max_turn = np.max(turn_rtl)
        adjustor = 1 - self.sequence_samples%2
        
        # Make smooth y-shift
        first_part =  max_turn - turn_rtl[range(int(np.ceil(self.sequence_samples/2)), self.sequence_samples)] # Create first and last part by flipping and turnign the turn_rtl array
        last_part = max_turn + turn_rtl[range(0, int(np.floor(self.sequence_samples/2) - adjustor + 1))]
        y_ramp_smooth = np.append(first_part, last_part)
        y_ramp_smooth = (self.corr_step_size/(2*max_turn)) * y_ramp_smooth        # adjust scale and offset of ramp

        turn_rtl = ltr_ramp[-1] + turn_rtl;

        prim_dim_sig = []
        sec_dim_sig = []
        new_value = start_y
        for i in range(0,self.steps_y):
            if i%2==0:               
                prim_dim_sig = np.concatenate((prim_dim_sig, ltr_ramp))#, turn_rtl))
            else:
                prim_dim_sig = np.concatenate((prim_dim_sig, rtl_ramp))#, turn_ltr))
            sec_dim_sig = np.concatenate((sec_dim_sig, new_value*np.ones(row_samples)))#, new_value+y_ramp_smooth))
            new_value = new_value + self.corr_step_size
            
#        i = i + 1
#        if i%2 == 0:
#            prim_dim_sig = np.concatenate((prim_dim_sig, ltr_ramp))
#        else:
#            prim_dim_sig = np.concatenate((prim_dim_sig, rtl_ramp))  
#        sec_dim_sig = np.concatenate((sec_dim_sig, new_value*np.ones(row_samples)))

        # Assign primary scan dir
        self.sig_dict[prim_scan_dim+'_sig'] = 1.14 * prim_dim_sig #1.14 is emperically measured correction factor
        # Assign second and third dim
        for key in self.sig_dict:
            if not key[0] == prim_scan_dim and not key[0] == 'z':
                self.sig_dict[key] = 1.14 * sec_dim_sig
            elif not key[0] == prim_scan_dim:
                self.sig_dict[key] = np.zeros(len(sec_dim_sig))
        
        
class VOL_Scan():
    """VOL_scan is the class representing the scanning movement for a volumetric 
    scan i.e. multiple conscutive XY-planes with a certain delta z distance."""
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': [], 'z_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_sizeXY = None
        self.corr_step_sizeZ = None        
        self.sequence_samples = None
        self.frames = 0        
        
    def update_frames(self, par_values):
        pass
    
    def update(self, par_values, prim_scan_dim):
        print('Updating VOL scan')
        # Create signals
        start_x = 0
        start_y = 0
        start_z = 0
        size_x = par_values['size_x'] / 2 #Division by 2 to convert from distance to voltage
        size_y = par_values['size_y'] / 2
        size_z = par_values['size_z'] / 2
        step_sizeXY = par_values['step_sizeXY'] / 2
        step_sizeZ = par_values['step_sizeZ'] / 2        
        self.sequence_samples = int(np.round(self.sample_rate * par_values['sequence_time'])) #WARNING: Correct for units of the time, now seconds!!!!
        self.steps_x = int(np.ceil(size_x / step_sizeXY))
        self.steps_y = int(np.ceil(size_y / step_sizeXY))
        self.steps_z = int(np.ceil(size_z / step_sizeZ))
        self.corr_step_sizeXY = size_x/self.steps_x # Step size compatible with width
        self.corr_step_sizeZ = size_z/self.steps_z # Step size compatible with range
        row_samples = self.steps_x * self.sequence_samples
        
        ramp_and_k = make_ramp(start_x, size_x, row_samples) # ramp_and_k contains [ramp, k]
        k = ramp_and_k[1]
        ltr_ramp = ramp_and_k[0]
        rtl_ramp = ltr_ramp[::-1]  # rtl_ramp contains only ramp, no k since same k = -k        
        
        prim_dim_sig = []
        sec_dim_sig = []
        new_value = start_y
        for i in range(0,self.steps_y):
            if i%2==0:               
                prim_dim_sig = np.concatenate((prim_dim_sig, ltr_ramp))
            else:
                prim_dim_sig = np.concatenate((prim_dim_sig, rtl_ramp))
            sec_dim_sig = np.concatenate((sec_dim_sig, new_value*np.ones(row_samples)))
            new_value = new_value + self.corr_step_sizeXY
            
        samples_p_slice = len(prim_dim_sig) #Used in Scanner->runScan
        self.cycles_p_slice = samples_p_slice / self.sequence_samples
        """Below we make the concatenation along the third dimension, between the "slices"
        we add a smooth transition to avoid too rapid motion that seems to cause strange movent.
        This needs to be syncronized with the pixel cycle signal"""
        fullZ_prim_dim_sig = prim_dim_sig
        fullZ_sec_dim_sig = sec_dim_sig
        fullZ_third_dim_sig = start_z*np.ones(len(prim_dim_sig))
        new_value = start_z + 1
        prim_dim_transition = make_smooth_step(prim_dim_sig[-1], prim_dim_sig[0], self.sequence_samples)
        sec_dim_transition = make_smooth_step(sec_dim_sig[-1], sec_dim_sig[0], self.sequence_samples)
        third_dim_transition = make_smooth_step(0, self.corr_step_sizeZ, self.sequence_samples)
        
        for i in range(1,self.steps_z-1):
            fullZ_prim_dim_sig = np.concatenate((fullZ_prim_dim_sig, prim_dim_transition))
            fullZ_sec_dim_sig = np.concatenate((fullZ_sec_dim_sig, sec_dim_transition))
            fullZ_third_dim_sig = np.concatenate((fullZ_third_dim_sig, new_value + third_dim_transition))
            
            fullZ_prim_dim_sig = np.concatenate((fullZ_prim_dim_sig, prim_dim_sig))
            fullZ_sec_dim_sig = np.concatenate((fullZ_sec_dim_sig, sec_dim_sig))
            fullZ_third_dim_sig = np.concatenate((fullZ_third_dim_sig, new_value*np.ones(len(prim_dim_sig))))
            new_value = new_value + self.corr_step_sizeZ
            
        fullZ_prim_dim_sig = np.concatenate((fullZ_prim_dim_sig, prim_dim_sig))
        fullZ_sec_dim_sig = np.concatenate((fullZ_sec_dim_sig, sec_dim_sig))
        fullZ_third_dim_sig = np.concatenate((fullZ_third_dim_sig, new_value*np.ones(len(prim_dim_sig))))    
        # Assign primary scan dir
        self.sig_dict[prim_scan_dim+'_sig'] = 1.14 * fullZ_prim_dim_sig #1.14 is an emperically measured correction factor
        # Assign second and third dim
        for key in self.sig_dict:
            if not key[0] == prim_scan_dim and not key[0] == 'z':
                self.sig_dict[key] = 1.14 * fullZ_sec_dim_sig
            elif not key[0] == prim_scan_dim:
                self.sig_dict[key] = 1.14 * fullZ_third_dim_sig
                
        print('Final X: '+str(fullZ_prim_dim_sig[-1]))
        print('Final Y: '+str(fullZ_sec_dim_sig[-1]))
        print('Final Z: '+str(fullZ_third_dim_sig[-1]))
        
        
        
# The follwing class contains the digital signals for the pixel cycle. The update function takes a parameter_values
# dict and updates the signal accordingly.         
        
class PixelCycle():
    
    def __init__(self, sample_rate):
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
        
    def update(self, devices = None):
                 
        if devices == None:
            devices = self.plot_sig_dict
            
        for device in devices:
            signal = self.pixel_cycle.sig_dict[device+'sig']
            self.plot_sig_dict[device].setData(signal)      
        
        
        
def make_ramp(start, end, samples):

    return np.linspace(start, end, num=samples)
    
    
def make_smooth_step(start, end, samples):

    x = np.linspace(start,end, num = samples, endpoint=True)
    x = 0.5 - 0.5*np.cos(x*np.pi)
    
    signal = start + (end-start)*x
    
    return signal

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
    
    
    
if __name__ == '__main__':
    ScanWid = ScanWidget(libnidaqmx.Device('Dev1'))
#    ScanWid.update_Scan()
