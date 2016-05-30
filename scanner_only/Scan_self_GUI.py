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
import matplotlib.pyplot as plt
import copy
import time
from numpy import arange
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore




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
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        self.widthPar = QtGui.QLineEdit('10')
        self.widthPar.editingFinished.connect(lambda: self.ScanParameterChanged('width'))
        self.heightPar = QtGui.QLineEdit('10')
        self.heightPar.editingFinished.connect(lambda: self.ScanParameterChanged('height'))
        self.sequence_timePar = QtGui.QLineEdit('0.1') # Seconds
        self.sequence_timePar.editingFinished.connect(lambda: self.ScanParameterChanged('sequence_time'))
        self.nrFramesPar = QtGui.QLabel()
        self.step_sizePar = QtGui.QLineEdit('0.5')
        self.step_sizePar.editingFinished.connect(lambda: self.ScanParameterChanged('step_size'))
        self.sample_rate = 100000
        
        self.Scan_Mode = QtGui.QComboBox()
        self.scan_modes = ['FOV scan', 'Line scan']
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
        self.current_aochannels = {'x': 0, 'y': 1}
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
        self.update_Scan(['TIS', '355', '405', '488', 'CAM'])
        
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
#        self.setLayout(grid)
#        grid.setRowMaximumHeight(0, 20)
#        grid.setRowMamimumHeight(1, 20)
        grid.addWidget(QtGui.QLabel('X channel'), 0, 4)
        grid.addWidget(self.XchanPar, 0, 5)
        grid.addWidget(QtGui.QLabel('Width (um):'), 0, 0, 2, 1)
        grid.addWidget(self.widthPar, 0, 1, 2, 1)
        grid.addWidget(QtGui.QLabel('Height (um):'), 0, 2, 2, 1)
        grid.addWidget(self.heightPar, 0, 3, 2, 1)
        grid.addWidget(QtGui.QLabel('Y channel'), 1, 4)
        grid.addWidget(self.YchanPar, 1, 5)
        grid.addWidget(QtGui.QLabel('Sequence Time (s):'),3, 0)
        grid.addWidget(self.sequence_timePar, 3, 1)
        grid.addWidget(QtGui.QLabel('Frames in scan:'), 3, 2)
        grid.addWidget(self.nrFramesPar, 3, 3)
        grid.addWidget(QtGui.QLabel('Step size (um):'), 3, 4)
        grid.addWidget(self.step_sizePar, 3, 5)
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
        grid.addWidget(self.ScanButton, 12, 3)
        grid.addWidget(self.PreviewButton, 12, 4)        
        
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
        if Xchan == Ychan:
            Ychan = (Ychan + 1)%4
            self.YchanPar.setCurrentIndex(Ychan)
        self.current_aochannels['x'] = Xchan
        self.current_aochannels['y'] = Ychan
        
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
        plt.plot(x_sig, y_sig)
        plt.axis([-0.2,self.scan_par_values['width']+0.2, -0.2, self.scan_par_values['height']+0.2])
        
    def ScanOrAbort(self):
        if not self.scanning:
            self.PrepAndRun()
        else:
            self.scanner.abort()
    
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
        
        
# This class was intended as a QThread to not intefere with the rest of the widgets in the GUI. 

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
        final_x = self.stage_scan.sig_dict['x_sig'][written_samps - 1]
        final_y = self.stage_scan.sig_dict['y_sig'][written_samps - 1]
        final_samps = [final_x, final_y]
        
        return_ramps = np.array([])
        for i in range(0,2):
            ramp_and_k = make_ramp(final_samps[i], 0, self.stage_scan.sample_rate)
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
        self.aotask.clear()
        self.dotask.clear()
        self.nidaq.reset()
        self.finalizeDone.emit()
        
    def runScan(self):
        self.nidaq.reset()
        scan_time = self.samples_in_scan / self.main.sample_rate
        ret = QtGui.QMessageBox.Yes
        self.scantimewar.setText("Scan will %s seconds" %scan_time)
        if scan_time > self.warning_time:
            ret = self.scantimewar.exec_()
            
        if ret == QtGui.QMessageBox.No:
            self.done()
            return
#        self.nidaq.reset() 


        full_ao_signal = []
        temp_aochannels = copy.copy(self.current_aochannels)
        min_ao = -10
        max_ao = 10
           
        # Following loop creates the voltage channels in smallest to largest order and places signals in same order.
        
        for i in range(0,2):
            dim = min(temp_aochannels, key = temp_aochannels.get)   # dim = dimension ('x' or 'y') containing smallest channel nr. 
            chanstring = 'Dev1/ao%s'%temp_aochannels[dim]
            self.aotask.create_voltage_channel(phys_channel = chanstring, channel_name = 'chan%s'%dim, min_val = min_ao, max_val = max_ao)
            temp_aochannels.pop(dim)
            signal = self.stage_scan.sig_dict[dim+'_sig']
            if i == 1 and len(full_ao_signal) != len(signal):
                print('Length of signals are not equal (printed from RunScan()')
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
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sample_rate = sample_rate
        self.sequence_samples = None
        self.FOV_scan = FOV_Scan(self.sample_rate)
        self.line_scan = Line_scan(self.sample_rate)
        self.scans = {'FOV scan': self.FOV_scan, 'Line scan': self.line_scan}
        self.frames = 0
        
    def set_scan_mode(self, mode):
        self.scan_mode = mode  
        
    def update_frames(self, par_values):
        self.scans[self.scan_mode].update_frames(par_values)
        self.frames = self.scans[self.scan_mode].frames
        
    def update(self, par_values):
        print('in update stage_scan')
        print('self.scan_mode = ', self.scan_mode)
        self.scans[self.scan_mode].update(par_values)
        self.sig_dict = self.scans[self.scan_mode].sig_dict
        self.sequence_samples = self.scans[self.scan_mode].sequence_samples
        self.frames = self.scans[self.scan_mode].frames
        
#        if self.scan_mode == 'FOV scan':
#            print('in update FOV_scan')
#            self.FOV_scan.update(par_values)
#            self.sig_dict = self.FOV_scan.sig_dict
#            self.sequence_samples = self.FOV_scan.sequence_samples
#            self.frames = self.FOV_scan.steps_x*self.FOV_scan.steps_y
#            
#        elif self.scan_mode == 'Line scan':
#            print('in update Line_scan')
#            self.line_scan.update(par_values)  
#            self.sig_dict = self.line_scan.sig_dict
#            self.sequence_samples = self.line_scan.sequence_samples
#            self.frames = self.line_scan.steps_y
            
class Line_scan():
        
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
    def update_frames(self, par_values):
        size_y = par_values['height'] / 2
        step_size = par_values['step_size'] / 2
        steps_y = int(np.ceil(size_y / step_size))
        self.frames = steps_y
        
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
        
        self.sig_dict['y_sig'] = 1.14 * ramp
        self.sig_dict['x_sig'] = np.zeros(len(ramp))
        
class FOV_Scan():
    
    def __init__(self, sample_rate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
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
        step_size = par_values['step_size'] / 2
        size_x = par_values['width'] / 2
        size_y = par_values['height'] / 2
        steps_x = int(np.ceil(size_x / step_size))
        steps_y = int(np.ceil(size_y / step_size))
        self.frames = steps_y*steps_x
                
        
    def update(self, par_values):
        
        # Create signals
        start_x = 0
        start_y = 0
        size_x = par_values['width'] / 2
        size_y = par_values['height'] / 2
        step_size = par_values['step_size'] / 2
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
        
        first_part =  max_turn - turn_rtl[range(int(np.ceil(self.sequence_samples/2)), self.sequence_samples)] # Create first and last part by flipping and turnign the turn_rtl array
        last_part = max_turn + turn_rtl[range(0, int(np.floor(self.sequence_samples/2) - adjustor + 1))]
        y_ramp_smooth = np.append(first_part, last_part)
        y_ramp_smooth = (self.corr_step_size/(2*max_turn)) * y_ramp_smooth        # adjust scale and offset of ramp

        turn_rtl = ltr_ramp[-1] + turn_rtl;

        x_sig = []
        y_sig = []
        new_value = start_y
        for i in range(0,self.steps_y):
            if i%2==0:               
                x_sig = np.concatenate((x_sig, ltr_ramp, turn_rtl))
            else:
                x_sig = np.concatenate((x_sig, rtl_ramp, turn_ltr))
            y_sig = np.concatenate((y_sig, new_value*np.ones(row_samples), new_value+y_ramp_smooth))
            new_value = new_value + self.corr_step_size
            
        i = i + 1
        if i%2 == 0:
            x_sig = np.concatenate((x_sig, ltr_ramp))
        else:
            x_sig = np.concatenate((x_sig, rtl_ramp))  
        y_sig = np.concatenate((y_sig, new_value*np.ones(row_samples)))

        # Assign x_sig
        self.sig_dict['x_sig'] = 1.14 * x_sig
        # Assign y_sig
        self.sig_dict['y_sig'] = 1.14 * y_sig
        
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
        
    def update(self, devices):
        
        for device in devices:
            signal = self.pixel_cycle.sig_dict[device+'sig']
            self.plot_sig_dict[device].setData(signal)      
        
        
        
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
    
    
    
if __name__ == '__main__':
    ScanWid = ScanWidget(libnidaqmx.Device('Dev1'))
#    ScanWid.update_Scan()