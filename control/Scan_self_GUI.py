# -*- coding: utf-8 -*-


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

initialDir = r"C:\Users\aurelien.barbotin\Documents\Data\DefaultDataFolder"

fraction_removed=0.1
#z stage:
min_ao_vertical = 0
max_ao_vertical = 10

#These factors convert an input in microns to the output in volts, and are hardware dependant
x_factor = 3.75
y_factor = 3.75
z_factor = 10

#Resonnance frequency of the stage for amplitude
f_0 = 53
#To save the sensor data
record_sensor_output = False
save_folder=r"C:\Users\aurelien.barbotin\Documents\Data\signals_15_8"
correction_factors = {'x':3.75,'y':3.75,'z':10}
minimum_voltages = {'x':-10,'y':-10,'z':0}
maximum_voltages = {'x':10,'y':10,'z':10}

pmt_sensitivity_channel = 'Dev1/ao3'

sample_rate = 10**5

# This class is intended as a widget in the bigger GUI, Thus all the commented parameters etc. It contain an instance
# of stage_scan and pixel_scan which in turn harbour the analog and digital signals respectively.
# The function run is the function that is supposed to start communication with the Nidaq
# through the Scanner object. This object was initially written as a QThread object but is not right now. 
# As seen in the commened lines of run() I also tried running in a QThread created in run().
# The rest of the functions contain mosly GUI related code.

class ScanWidget(QtGui.QFrame):
    """Class generating the GUI for stage scanning. This GUI is intended to specify the different parameters such as
    pixel dwell time, step size, scanning width and height etc. This class is intended as a widget in the bigger GUI.
    
    :param nidaqmx.Device device: NiDaq card.
    """
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.aochannels = device.get_analog_output_channels()
        self.dochannels = device.get_digital_output_lines()
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        
        self.back_factor = 10
        self.back_factor_param=QtGui.QLineEdit("2")
        self.back_factor_param.editingFinished.connect(lambda: self.ScanParameterChanged("back_factor"))
        
        #Creating the GUI itself
        self.widthPar = QtGui.QLineEdit('10')
        self.widthPar.editingFinished.connect(lambda: self.ScanParameterChanged('width'))
        self.heightPar = QtGui.QLineEdit('10')
        self.heightPar.editingFinished.connect(lambda: self.ScanParameterChanged('height'))
        self.sequence_timePar = QtGui.QLineEdit('0.00007') # Seconds
        self.sequence_timePar.editingFinished.connect(lambda: self.ScanParameterChanged('sequence_time'))
        self.nrFramesPar = QtGui.QLabel()
        self.frequencyLabel = QtGui.QLabel()
        self.step_sizePar = QtGui.QLineEdit('0.05')
        self.step_sizePar.editingFinished.connect(lambda: self.ScanParameterChanged('step_size'))
        self.sample_rate = 70000  #works until 70000
        self.delay = QtGui.QLineEdit("0")
        
        self.scan_modes = ['xy scan','xz scan','yz scan']
        self.Scan_Mode = QtGui.QComboBox()
        self.Scan_Mode.addItems(self.scan_modes)
        self.Scan_Mode.currentIndexChanged.connect(lambda: self.setScanMode(self.Scan_Mode.currentText()))
        
        self.recording_devices = ['APD','PMT']
        self.recording_device = QtGui.QComboBox()
        self.recording_device.addItems(self.recording_devices)
        self.recording_device.currentIndexChanged.connect(lambda: self.setRecordingDevice( self.recording_device.currentText() ) )
        self.current_recording_device="APD"
        
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
        
        
        
        self.display = ImageDisplay(self,(200,200))
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
        grid.addWidget(QtGui.QLabel('Line scanning frequency(Hz):'), 3, 2)        
        grid.addWidget(self.frequencyLabel,3,3)
        grid.addWidget(QtGui.QLabel('Step size (um):'), 3, 4)
        grid.addWidget(self.step_sizePar, 3, 5)
        grid.addWidget(QtGui.QLabel('correction samples:'), 4, 4)
        grid.addWidget(self.delay, 4, 5)
        grid.addWidget(QtGui.QLabel('return time (ms)'), 5, 4)
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
        grid.addWidget(self.recording_device,8,5)
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
        grid.addWidget(self.positionner,13,0,1,4)    
        
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
        """Sets the scanning strategy to be implemented by the StageScan class
        
        :param string mode: xy scan, xz scan or yz scan"""
        
        self.stage_scan.set_scan_mode(mode)
        self.ScanParameterChanged('scan_mode')
    def setRecordingDevice(self,device):
        """sets the name of the device which will be used for scanning.
        
        :param string device: APD or PMT"""
        self.current_recording_device = device
        
    def AOchannelsChanged (self):
        """If one analog output channel is changed by the user, makes sure that 2 tasks are not sent to the
        same channel."""
        Xchan = self.XchanPar.currentIndex()
        Ychan = self.YchanPar.currentIndex()
        Zchan = self.ZchanPar.currentIndex()
        if Xchan == 0:
            Ychan = 1
        elif Xchan ==1:
                Ychan =0
        else:
            Xchan=0
            Ychan=1
            self.XchanPar.setCurrentIndex(Xchan)
            self.YchanPar.setCurrentIndex(Ychan)
#        count=len(self.aochannels)
#        while( (Zchan == Xchan or Zchan == Ychan) and count>0):
#            Zchan = (Zchan + 1)%len(self.aochannels)
#            self.ZchanPar.setCurrentIndex(Zchan)
#            count-=1
#        if(count == 0):
#            print("couldn't find satisfying channel for Z")
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
        self.frequencyLabel.setText(str(self.stage_scan.frequency) )
        
    def PixelParameterChanged(self, parameter):
        self.pixel_par_values[parameter] = float(self.pixel_parameters[parameter].text())
        device = parameter[-3]+parameter[-2]+parameter[-1]
        self.pixel_cycle.update([device], self.pixel_par_values, self.stage_scan.sequence_samples) 
        self.graph.update([device])
        
    def PreviewScan(self):
        """Displays a matplotlib graph representing the scanning's trajectory."""
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
        """Prepares Tempesta for scanning then starts the scan."""
        if self.scanRadio.isChecked() or self.contScanRadio.isChecked():
            self.stage_scan.update(self.scan_par_values)
            self.ScanButton.setText('Abort')
            channels_used=[self.stage_scan.axis_1,self.stage_scan.axis_2]
            self.positionner.reset_channels(channels_used)
            self.scanner = Scanner(self.nidaq, self.stage_scan, self.pixel_cycle, self.current_aochannels, self.current_dochannels,self.current_recording_device, self)
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
        """Called when *self.scanner* is done"""
        print('in ScanDone()')
        self.ScanButton.setEnabled(False)
        
    def FinalizeDone(self):
        self.ScanButton.setText('Scan')
        self.ScanButton.setEnabled(True)
        print('Scan Done')
        channels_to_reset = [self.stage_scan.axis_1,self.stage_scan.axis_2]
        del self.scanner
        self.scanning = False
        self.positionner.reset_channels(channels_to_reset)
        
    def update_Scan(self, devices):
        """Creates a scan with the new parameters"""
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
    """Thread regulating the scanning timing. Is used to pass from one step to another after it is finished."""
    waitdoneSignal = QtCore.pyqtSignal()
    def __init__(self, aotask):
        """:param nidaqmx.AnalogOutputTask aotask: task this thread is waiting for."""
        super().__init__()
        self.aotask = aotask
        self.wait = True
        
        self.isRunning=False
            
    def run(self):
        """runs until *self.aotask* is finished, and then emits the waitdoneSignal"""
        print('will wait for aotask')       
        self.isRunning=True
#        self.aotask.wait_until_done()
        while not self.aotask.is_done() and self.wait:
            pass
        self.wait = True
        self.waitdoneSignal.emit()
        self.isRunning=False
        print('aotask is done')
    
    def stop(self):
        """stops the thread, called in case of manual interruption."""
        self.wait = False

class Scanner(QtCore.QObject):        
    """This class plays the role of interface between the software and the hardware. It writes the different signals to the
    electronic cards and manages the timing of a scan.
    
    :param nidaqmx.Device device: NiDaq card
    :param StageScan stage_scan: object containing the analog signals to drive the stage
    :param PixelCycle pixel_cycle: object containing the digital signals to drive the lasers at each pixel acquisition
    :param dict current_aochannels: available analog output channels
    :param dict current_dochannels: available digital output channels
    :param string recording_device: the name of the device which will get the photons from the scan (APD or PMT)
    :param ScanWidget main: main scan GUI"""
        
    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()
    def __init__(self, device, stage_scan, pixel_cycle, current_aochannels, current_dochannels,recording_device, main, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.stage_scan = stage_scan
        self.pixel_cycle = pixel_cycle
        self.current_aochannels = current_aochannels  # Dict containing channel numbers to be written to for each signal
        self.current_dochannels = current_dochannels  # Dict containing channel numbers to be written to for each device
        self.samples_in_scan = len(self.stage_scan.sig_dict['sig_'+self.stage_scan.axis_1])
        self.recording_device = recording_device
        self.main = main
        
        self.aotask = nidaqmx.AnalogOutputTask("scannerAOtask")
        self.trigger = nidaqmx.CounterOutputTask("trigger")
#        self.dotask = nidaqmx.DigitalOutputTask("scannerDOtask")       
        self.waiter = Wait_Thread(self.aotask)
        if self.recording_device =="PMT":
            self.record_thread = RecordingThreadPMT(self.main.display)
        else:
            self.record_thread = RecordingThreadAPD(self.main.display)
            print("recording device:",self.recording_device)
        
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
        """Once the scanning is finished, sends the different devices to the position specified by the Positionner."""
        print('in finalize')
        self.contScan=self.main.contScanRadio.isChecked()    #Boolean specifying if we are running a continuous scanning or not
        if(not self.contScan):            
            self.scanDone.emit()
        self.waiter.waitdoneSignal.disconnect(self.finalize) #Important, otherwise finalize is called again when next waiting finishes.
        self.waiter.waitdoneSignal.connect(self.done)
            
        written_samps = self.aotask.get_samples_per_channel_generated()
        print("written samples",written_samps)
        goals = [0,0]  #Where we write the target positions to return to
        final_1 = self.stage_scan.sig_dict['sig_'+self.stage_scan.axis_1][written_samps - 1]
        final_2 = self.stage_scan.sig_dict['sig_'+self.stage_scan.axis_2][written_samps - 1]
        goals[0] = getattr(self.main.positionner, self.stage_scan.axis_1)
        goals[1] = getattr(self.main.positionner, self.stage_scan.axis_2)
            
        final_samps = [final_1, final_2]
        return_time= 0.05    #Return time of 50ms, it is enough
        return_ramps = np.array([])
        for i in range(0,2):
            ramp_and_k = make_ramp(final_samps[i], goals[i], int(self.stage_scan.sample_rate*return_time) )
#            flat_signal = np.ones( ( int(return_time * self.stage_scan.sample_rate)-self.stage_scan.sample_rate//2) )*goals[i]
            
#            return_ramps = np.concatenate((return_ramps, ramp_and_k[0],flat_signal))
            return_ramps = np.concatenate((return_ramps, ramp_and_k[0]))
        self.aotask.stop()    
#        time.sleep(0.01)

        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                                 sample_mode = 'finite',
                                                 samples_per_channel = int(return_time*self.stage_scan.sample_rate) )        
        
        self.aotask.write(return_ramps, layout = 'group_by_channel', auto_start = False)
        self.aotask.start()

        self.waiter.start()
        

            


    def done(self):
        """Once the different scans are in their initial position, starts again a new scanning session if in continuous scan mode.
        If not releases the channels for the positionner."""
        print('in self.done()')
        if(self.contScan):
            print("in contScan")
            #If scanning continuously, regenerate the samples and write them again
            self.aotask.stop()
#            self.dotask.stop()
            self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                             sample_mode = 'finite',
                                             samples_per_channel = self.samples_in_scan)
            self.aotask.configure_trigger_digital_edge_start(reference_trigger)             
#            self.dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
#                                             rate = self.pixel_cycle.sample_rate, 
#                                             sample_mode = 'finite', 
#                                             samples_per_channel = self.samples_in_scan)
            self.trigger.stop()                                 
            self.waiter.waitdoneSignal.disconnect(self.done)
            self.waiter.waitdoneSignal.connect(self.finalize)
        
#            self.dotask.write(self.full_do_signal, layout = 'group_by_channel', auto_start = False)
            self.aotask.write(self.full_ao_signal, layout = 'group_by_channel', auto_start = False)
#            self.dotask.start()

            self.record_thread.start()
            self.aotask.start()
            self.waiter.start()
            time.sleep(0.01)
            self.trigger.start()
        else:
            self.aotask.clear()
#            self.dotask.clear()
            self.trigger.stop()
            self.trigger.clear()
            del self.trigger
            self.record_thread.stop()
            self.finalizeDone.emit()
            print("total Done")
        
    def runScan(self):
        """Called when the run button is pressed. Prepares the display, the recording thread for acquisition, and writes the 
        values of the scan in the corresponding tasks"""
        self.n_frames = self.stage_scan.frames
        
        image_shape = (self.stage_scan.steps_2 , self.stage_scan.steps_1 )
        self.main.display.update_parameters(image_shape)
        
        scan_time = self.samples_in_scan / self.main.sample_rate
        ret = QtGui.QMessageBox.Yes
        self.scantimewar.setText("Scan will last %s seconds" %scan_time)
        if scan_time > self.warning_time:
            ret = self.scantimewar.exec_()
            
        if ret == QtGui.QMessageBox.No:
            self.contScan=False
            self.done()
            return


        self.full_ao_signal = []
        temp_aochannels = copy.copy(self.current_aochannels)
           
        # Following loop creates the voltage channels in smallest to largest order and places signals in same order.
        axis_1 = self.stage_scan.axis_1
        axis_2 = self.stage_scan.axis_2
        
        chanstring_1 = 'Dev1/ao%s'%temp_aochannels[axis_1] #typically, the first axis is x and 
#        self.stage_scan.axis_1 = "x", then the corresponding ao channel is number 0 and chanstring_1 = 'Dev1/ao1'
        self.aotask.create_voltage_channel(phys_channel = chanstring_1, channel_name = 'chan1', 
                                           min_val = minimum_voltages[axis_1], max_val = maximum_voltages[axis_1])
        signal_1 = self.stage_scan.sig_dict['sig_'+axis_1]
        print("frequency stage scan:",self.stage_scan.frequency,"correc factor",amplitude_correction(fraction_removed,self.stage_scan.frequency))
        
        chanstring_2='Dev1/ao%s'%temp_aochannels[axis_2]
        self.aotask.create_voltage_channel(phys_channel = chanstring_2, channel_name = 'chan2',
                                           min_val = minimum_voltages[axis_2], max_val = maximum_voltages[axis_2])
        signal_2 = self.stage_scan.sig_dict['sig_'+axis_2]
        print("length signals:",len(signal_1),len(signal_2))
        self.full_ao_signal = np.append(signal_1,signal_2)
        
        if len(signal_1)!=len(signal_2):
            print("error: wrong signal size")


        if record_sensor_output:
            self.stage_scan.samples_per_line
            name=str(round(self.main.sample_rate /self.stage_scan.samples_per_line))+"Hz"
            np.save(save_folder+"\\"+"driving_signal_"+name,signal_1)
        # Same as above but for the digital signals/devices        
        
#        self.full_do_signal = []
#        temp_dochannels = copy.copy(self.current_dochannels)
#        for i in range(0,len(temp_dochannels)):
#            dev = min(temp_dochannels, key = temp_dochannels.get)
#            chanstring = 'Dev1/port0/line%s'%temp_dochannels[dev]
#            self.dotask.create_channel(lines = chanstring, name = 'chan%s'%dev)
#            temp_dochannels.pop(dev)
#            signal = self.pixel_cycle.sig_dict[dev+'sig']
#            print("do signal length",len(signal))
#            if len(self.full_ao_signal)%len(signal) != 0 and len(self.full_do_signal)%len(signal) != 0:
#                print('Signal lengths does not match (printed from run)')
#            self.full_do_signal = np.append(self.full_do_signal, signal)
        

        
        self.aotask.configure_timing_sample_clock(rate = self.stage_scan.sample_rate,
                                             sample_mode = 'finite',
                                             samples_per_channel = self.samples_in_scan)
                        
#        self.dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
#                                             rate = self.pixel_cycle.sample_rate, 
#                                             sample_mode = 'finite', 
#                                             samples_per_channel = self.samples_in_scan)
#        self.dotask.write(self.full_do_signal, layout = 'group_by_channel', auto_start = False)
#        self.dotask.start()
                     
        self.trigger.create_channel_ticks('Dev1/ctr1',name="pasteque",  low_ticks=100000, high_ticks=1000000)
        self.trigger.set_terminal_pulse('Dev1/ctr1',"PFI12")
        self.trigger.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                             sample_mode = "hwtimed")
        self.aotask.configure_trigger_digital_edge_start(reference_trigger)                        
        
        self.waiter.waitdoneSignal.connect(self.finalize)
        self.aotask.write(self.full_ao_signal, layout = 'group_by_channel', auto_start = False)
        self.record_thread.setParameters(self.stage_scan.sequence_samples,self.samples_in_scan,self.stage_scan.sample_rate,self.stage_scan.samples_per_line,axis_1)
        self.record_thread.start()
        time.sleep(0.01)    #Necessary for good synchronization
        self.aotask.start()
        self.waiter.start()
        self.trigger.start()
#        self.trigger.write(np.append(np.zeros(1000),np.ones(1000)))     
        
    def abort(self):
        """Stops the current scan. Stops the recording thread and calls the method finalize"""
        print('Aborting scan')
        self.waiter.stop()
        self.aotask.stop()
        print("in abort, aotask samples generated",self.aotask.get_samples_per_channel_generated())
#        self.dotask.stop()
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
        
class StageScan():
    """Scanning in xy
    
    :param ScanWidget main: main scan GUI
    :param float sample_rate: sample rate in samples per second"""
    def __init__(self,main, sample_rate):
        self.scan_mode = 'xy scan'
        self.sig_dict = {'sig_x': [], 'sig_y': [],'sig_z':[]}
        self.sample_rate = sample_rate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0
        
        self.axis_1 = self.scan_mode[0]     #x if in '[x]y scan'
        self.axis_2 = self.scan_mode[1]     #y if in 'x[y] scan'
        
        self.scanWidget=main
        
        self.steps_1 = 0
        self.steps_2 = 0
        self.samples_per_line=0
        
        self.frequency=0
        
    def set_scan_mode(self,mode):
        self.scan_mode = mode
        self.axis_1 = self.scan_mode[0]     #x for instance
        self.axis_2 = self.scan_mode[1]     #y for instance
        
    def update_frames(self, par_values):
        self.steps_1 = int(np.ceil(par_values['width'] / par_values['step_size']))
        self.steps_2 = int(np.ceil(par_values['height'] / par_values['step_size']))
        self.frames = self.steps_1*self.steps_2
        
        step_size_1 = par_values['step_size'] / correction_factors[self.axis_1]
        self.sequence_samples = int(np.ceil(self.sample_rate * par_values['sequence_time'])) #WARNING: Correct for units of the time, now seconds!!!!
        if self.sequence_samples==1:
            self.sequence_samples+=1
        row_samples=self.steps_1*self.sequence_samples
        self.frequency = round(self.sample_rate/ (row_samples*2),1)
                
        
    def update(self, par_values):
        """creates the signals inside the dictionnary self.sig_dict
        
        :param dict par_values:contains the name and value of the scanning parameters"""
        # Create signals
        try:
            start_1 = getattr(self.scanWidget.positionner, self.axis_1)
            start_2 = getattr(self.scanWidget.positionner, self.axis_2)
            print("start_1:",start_1)
        except:
            start_1 = 0
            start_2 = 0
            print("couldn't access to the positionner")
        print("width:",par_values['width'])
        size_1 = par_values['width'] / correction_factors[self.axis_1]
        size_2 = par_values['height'] / correction_factors[self.axis_2]
        step_size_1 = par_values['step_size'] / correction_factors[self.axis_1]
        step_size_2 = par_values['step_size'] / correction_factors[self.axis_2]
        
        self.sequence_samples = int(np.ceil(self.sample_rate * par_values['sequence_time'])) #We want at least two sample per point
        if self.sequence_samples==1:
            self.sequence_samples+=1
            print("not enough samples")
        self.steps_1 = int(np.ceil(size_1 / step_size_1))
        self.steps_2 = int(np.ceil(size_2 / step_size_2))
        self.frames = self.steps_1*self.steps_2
        
        self.corr_step_size = size_2/self.steps_2 # Step size compatible with width
        row_samples = self.steps_1 * self.sequence_samples
        sig_1 = []
        sig_2 = []
        
        new_value = start_2
        n_samples_ramp = int(2* fraction_removed * row_samples)
        n_samples_flat= int((row_samples-n_samples_ramp)/2)
        n_samples_flat_2 = row_samples-n_samples_flat-n_samples_ramp
        ramp_axis_2 = make_ramp(0,self.corr_step_size, n_samples_ramp)[0]
        print("row samples",row_samples)
        #sine scanning
        sine = np.arange(0,row_samples*self.steps_2)/(row_samples*2) * 2 * np.pi
        sine=np.sin(sine) * size_1/2    #Sine varies from -1 to 1 so need to divide by 2
        self.frequency = self.sample_rate/ (row_samples*2)
        print("scan, self.freq",self.frequency)
        for i in range(0,self.steps_2):            
            sig_2 = np.concatenate( (sig_2, new_value * np.ones(n_samples_flat),new_value+ramp_axis_2, self.corr_step_size+new_value * np.ones(n_samples_flat_2) ) )
            new_value = new_value + self.corr_step_size       
            
#        self.samples_per_line = row_samples + n_samples_return
        self.samples_per_line = row_samples
        print("sequence samples:",self.sequence_samples)
        #Correction for amplitude:
        sig_1=sine
        sig_1*=amplitude_correction(fraction_removed,self.frequency)
        sig_1+=start_1
        print("shape signals",sig_1.shape,sig_2.shape) 
        # Assign signal to axis 1
        self.sig_dict['sig_'+self.axis_1] =  sig_1
        # Assign signal to axis 2
        self.sig_dict['sig_'+self.axis_2] =  sig_2
        np.save(r"C:\Users\aurelien.barbotin\Documents\Data\signal.npy",sig_2)
        
class PixelCycle():
    """Contains the digital signals for the pixel cycle, ie the process repeated for the acquisition of each pixel. 
    The update function takes a parameter_values dict and updates the signal accordingly.
    
    :param float sample_rate: sample rate in samples per seconds"""
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
        

class ImageDisplay(QtGui.QWidget):
    """Class creating a display for the images obtained either with an APD or PMT
    
    :param ScanWidget main: main scan GUI
    :param tuple shape: width and height of the image."""
    def __init__(self,main,shape):
        super().__init__()
        
        self.setWindowTitle("Image from scanning") 
        
        self.array=np.zeros((shape[1],shape[0]))
        self.shape=(shape[1],shape[0])
        self.pos=[0,0]
        self.scanWidget=main
           

        #File management
        self.initialDir = initialDir
        self.saveButton = QtGui.QPushButton("Save image")
        self.saveButton.clicked.connect(self.saveImage)
        
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        self.browseButton =  QtGui.QPushButton("Choose folder")
        self.browseButton.clicked.connect(self.loadFolder)
        
            #Visualisation widget
        self.graph = pg.GraphicsLayoutWidget()
        self.vb = self.graph.addPlot(row=1,col=1)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.img.translate(-0.5, -0.5)
        self.vb.setAspectLocked(True)
        self.img.setImage(self.array)
        self.vb.setMinimumHeight(200)
        
        #To get intensity profile along a line
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.graph.addItem(self.hist, row=1, col=2)
        
        self.profile_plot = self.graph.addPlot(row=2,col=1)
        self.profile_plot.setMaximumHeight(150)
        self.line = pg.LineSegmentROI([[0, 0], [10,0]], pen='r')
        self.vb.addItem(self.line)        
        self.line.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updatePlot)
        self.viewtimer.start(50)

        self.isTurning = False
        
        layout=QtGui.QGridLayout()
        self.setLayout(layout)
        
        layout.addWidget(self.graph,0,0,5,5)
        layout.setRowMinimumHeight(0,300)
        layout.setColumnMinimumWidth(0,300)
        layout.addWidget(self.saveButton,5,1)
        layout.addWidget(self.browseButton,5,2)
        layout.addWidget(self.folderEdit,5,0)
        
        
    def update_parameters(self,shape):
        """reshapes the array with the proper dimensions before acquisition
        
        :param tuple shape: width and height of the image."""
        self.array=np.zeros((shape[1],shape[0]))
        self.shape=(shape[0],shape[1])
        self.pos=[0,0]
        self.img.setImage(self.array)
        
    def updatePlot(self):
        selected = self.line.getArrayRegion(self.array, self.img)
        self.profile_plot.plot(selected, clear=True,pen=(100,100,100), symbolBrush=(255,0,0), symbolPen='w')
        
    def loadFolder(self):
        """Open a window to browse folders, and to select the folder in which images are to be saved."""
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
        """Saves the current image conainted in *self.array* under a predefined name containing 
        the main scanning parameters along with the date, under the tiff format."""
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
        """sets the value of one pixel from an input array. Not used anymore: point by point filling is too slow,
        we use set_line_value instead."""
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
        """Inserts *line* at the proper position in *self.array*
        
        :param numpy.ndarray line: line of data to insert in the image."""
        line = np.asarray(line)
        self.array[:,self.pos[1]]=line             
            
        self.pos[1]+=1
        self.pos[1]=self.pos[1]%(self.shape[0])
            
        self.img.setImage(self.array)
           
class RecordingThreadAPD(QtCore.QThread):
    """Thread recording an image with an APD (Counter input) while the stage is scanning
    
    :param ImageDisplay display: ImageDisplay in scanwidget"""
    def __init__(self,main):
        super().__init__()
        self.imageDisplay=main
        self.exiting = True
        self.delay=0
        
        
    def setParameters(self,sequence_samples,samples_per_channel,sample_rate,samples_per_line,main_axis):
        """prepares the thread for data acquisition with the different parameters values
        
        :param int sequence_samples: number of samples for acquisition of 1 point
        :param int samples_per_channel: Total number of samples generated per channel in the scan
        :param int sample_rate: sample rate in number of samples per second
        :param int samples_per_line: Total number of samples acquired or generate per line, to go back and forth.
        :param string main_axis: The axis driven with a sine
        """
        self.samples_per_line = samples_per_line
        self.sequence_samples = sequence_samples
        self.rate = sample_rate 
        self.main_axis=main_axis #Usually it is x
        self.frequency= self.rate / self.samples_per_line/2
        
        self.steps_per_line = self.imageDisplay.shape[1]
 
        self.n_frames=self.imageDisplay.shape[0]
        
        self.samples_in_scan = samples_per_channel
            



        #To record the sensor output
        record_channel='Dev1/ai5'
        if self.main_axis=="y":
            record_channel='Dev1/ai6'
            
        self.delay = self.rate/self.frequency/4     #elimination of 1/4 period at the beginning
        self.delay+= phase_correction(self.frequency)/self.frequency * self.rate/2/np.pi
        self.delay = int(self.delay)
        
        self.aitask = nidaqmx.AnalogInputTask()
        self.aitask.create_voltage_channel(record_channel,  min_val=-0.5, max_val=10.0)        
        self.aitask.configure_timing_sample_clock(source = r'ao/SampleClock',
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan+self.delay)
        self.aitask.configure_trigger_digital_edge_start(reference_trigger) 
        print("init citask")
        self.citask = nidaqmx.CounterInputTask()
        self.citask.create_channel_count_edges("Dev1/ctr0", init=0 )
        self.citask.set_terminal_count_edges("Dev1/ctr0","PFI0")
        self.citask.configure_timing_sample_clock(source = r'ao/SampleClock',
                                             sample_mode = 'finite', 
                                            samples_per_channel = self.samples_in_scan+self.delay)
        self.citask.set_arm_start_trigger_source(reference_trigger)
        self.citask.set_arm_start_trigger(trigger_type='digital_edge')
    def run(self):
        """runs this thread to acquire the data in task"""
        self.exiting=False
        self.aitask.start()
        self.citask.start()
#        while not self.imageDisplay.scanWidget.scanner.waiter.isRunning:
#            print("we wait for beginning of aotask")
        

        
#        while self.citask.get_samples_per_channel_acquired()<2:
#            print("waiting for 1st samples to be acquired")
        print("samples apd acquired beofre:",self.citask.get_samples_per_channel_acquired())
        throw_apd_data = self.citask.read(self.delay)
        print("samples apd acquired after:",self.citask.get_samples_per_channel_acquired())
        throw_sensor_data=self.aitask.read(self.delay,timeout=10)  #To synchronize analog input and output

        print("self.delay",self.delay)
        counter = self.n_frames
        print("samples per line",self.samples_per_line)
        last_value = throw_apd_data[-1]        
        
        amplitude=float(self.imageDisplay.scanWidget.widthPar.text())/correction_factors[self.main_axis]
        initial_position = getattr(self.imageDisplay.scanWidget.positionner,self.main_axis)
        
        while(counter>0 and not self.exiting):
            apd_data=self.citask.read(self.samples_per_line)
            sensor_data = self.aitask.read(self.samples_per_line,timeout=10)
            if counter==5:
                print("saving sensor data...")
                np.save(r"C:\Users\aurelien.barbotin\Documents\Data\signal_ref_2.npy", sensor_data)
            sensor_data=sensor_data[:,0]
            substraction_array = np.concatenate(([last_value],apd_data[:-1]))
            last_value = apd_data[-1]
            apd_data = apd_data-substraction_array #Now apd_data is an array contains at each position the number of counts at this position
            line = line_from_sine(apd_data,sensor_data,self.steps_per_line,amplitude,initial_position)
            self.emit(QtCore.SIGNAL("line(PyQt_PyObject)"),line)
            if counter<6:
                print("counter",counter,"exiting",self.exiting)
            counter-=1
        print("samples acquired in citask",self.citask.get_samples_per_channel_acquired(),"samples available",self.citask.get_samples_per_channel_available())
        self.exiting=True
        self.aitask.stop()
        self.citask.stop()
        
    def stop(self):
        self.exiting=True
        try:
            self.aitask.stop()
            self.citask.stop()
            del self.aitask
            del self.citask
            print("stopped regularly")
        except:
            del self.aitask
            del self.citask
            print("deeestroyed")
        
    
class RecordingThreadPMT(QtCore.QThread):
    """Thread to record an image with the PMT while the stage is scanning
    
    :param ImageDisplay main: ImageDisplay in scanwidget"""
    def __init__(self,main):
        super().__init__()
        self.imageDisplay=main
        self.exiting = True
        self.delay=0
        
    def setParameters(self,sequence_samples,samples_per_channel,sample_rate,samples_per_line,main_axis):
        """prepares the thread for data acquisition with the different parameters values
        
        :param int sequence_samples: number of samples for acquisition of 1 point
        :param int samples_per_channel: Total number of samples generated per channel in the scan
        :param int sample_rate: sample rate in number of samples per second
        :param int samples_per_line: Total number of samples acquired or generate per line, to go back and forth.
        :param string main_axis: The axis driven with a sine
        """
        self.samples_per_line = samples_per_line
        self.sequence_samples = sequence_samples
        self.rate = sample_rate 
        self.main_axis=main_axis #Usually it is x
        self.frequency= self.rate / self.samples_per_line/2
        
        self.steps_per_line = self.imageDisplay.shape[1]
 
        self.n_frames=self.imageDisplay.shape[0]
        
        self.samples_in_scan = samples_per_channel   
        if(self.rate != sample_rate * self.samples_in_scan / samples_per_channel):
            print("error arrondi")
            
        print("parameters for acquisition of data : sample rate",self.rate,"samples_per_channel:",self.samples_in_scan)
        self.aitask = nidaqmx.AnalogInputTask()
        self.aitask.create_voltage_channel('Dev1/ai0', terminal = 'rse', min_val=-1, max_val=10.0)

    #To record the sensor output
        record_channel='Dev1/ai5'
        if self.main_axis=="y":
            record_channel='Dev1/ai6'
            
        self.aitask.create_voltage_channel(record_channel, terminal = 'rse', min_val=-0.5, max_val=10.0)
        self.delay = self.rate/self.frequency/4     #elimination of 1/4 period at the beginning
        self.delay+= phase_correction(self.frequency)/self.frequency * self.rate/2/np.pi
        print("delay",self.delay,"value 1",phase_correction(self.frequency)/self.frequency * self.rate/2/np.pi,"value 2",self.rate/self.frequency/4)
        self.delay = int(self.delay)
        
        self.aitask.configure_timing_sample_clock(source = r'ao/SampleClock',
                                             sample_mode = 'finite', 
                                             samples_per_channel = self.samples_in_scan+self.delay)

                                             
                      
    
    def run(self):
        """runs this thread to acquire the data in task"""
        self.exiting=False
        self.aitask.start()
        dat=self.aitask.read(self.delay,timeout=30)  #To synchronize analog input and output
        #To change!!
        counter = self.n_frames
        if record_sensor_output:
            sensor_vals = np.zeros(self.samples_in_scan+self.delay)
            sensor_vals[0:self.delay]=dat[:,1]
            
        amplitude=float(self.imageDisplay.scanWidget.widthPar.text())/correction_factors[self.main_axis]
        initial_position = getattr(self.imageDisplay.scanWidget.positionner,self.main_axis)

        
        while(counter>0 and not self.exiting):
            data=self.aitask.read(self.samples_per_line,timeout=10)
            if record_sensor_output:
                sensor_vals[self.delay+(self.n_frames-counter)*self.samples_per_line:self.delay+(self.n_frames-counter+1)*self.samples_per_line] = data[:,1]

            line = line_from_sine(data[:,0],data[:,1],self.steps_per_line,amplitude,initial_position)
            self.emit(QtCore.SIGNAL("line(PyQt_PyObject)"),line)
            if counter<6:
                print("counter:",counter)
            counter-=1
            
        if record_sensor_output:
            name=str(round(self.rate /self.samples_per_line))+"Hz"
            np.save(save_folder+"\\" +"sensor_output_x"+name,sensor_vals)
        self.aitask.stop()
        self.exiting=True
    def stop(self):
        """Stops the worker"""
        try:
            self.aitask.stop()
            self.aitask.clear()
            del self.aitask
        except:
            pass
            
class Positionner(QtGui.QWidget):
    """This class communicates with the different analog outputs of the nidaq card. When not scanning, it drives the 3 axis 
    x, y and z as well as the PMT sensitivity.
    
    :param ScanWidget main: main scan GUI"""
    
    def __init__(self,main):
        super().__init__()
        self.scanWidget = main
        
        #Position of the different devices in V
        self.x = 0
        self.y = 0
        self.z = 0
        self.pmt_sensitivity = 0.3
        
        #Parameters for the ramp (driving signal for the different channels)
        self. ramp_time = 800    #Time for each ramp in ms
        self.sample_rate = 10**5
        self.n_samples = int(self.ramp_time * 10**-3 * self.sample_rate)
        
        #This boolean is set to False when tempesta is scanning to prevent this positionner to
        #access the analog output channels
        self.isActive = True    
        self.active_channels = ["x","y","z"]
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
        
        self.title=QtGui.QLabel()
        self.title.setText("Stage Positionner")
        self.title.setStyleSheet("font-size:18px")
        
        layout = QtGui.QGridLayout()        
        self.setLayout(layout)
        layout.addWidget(self.title,0,0)
        layout.addWidget(self.pmt_sliderLabel,1,0,1,1)
        layout.addWidget(self.pmt_minVal,2,0,1,1)
        layout.addWidget(self.pmt_maxVal,2,2,1,1)
        layout.addWidget(self.pmt_slider,3,0,1,3)
        layout.addWidget(self.pmt_value_line,3,3,1,1)
                
        layout.addWidget(self.z_sliderLabel,1,4,1,1)
        layout.addWidget(self.z_minVal,2,4,1,1)
        layout.addWidget(self.z_maxVal,2,6,1,1)
        layout.addWidget(self.z_slider,3,4,1,3)
        layout.addWidget(self.z_value_line,3,7,1,1)
        
        layout.addWidget(self.x_sliderLabel,4,0,1,1)
        layout.addWidget(self.x_minVal,5,0,1,1)
        layout.addWidget(self.x_maxVal,5,2,1,1)
        layout.addWidget(self.x_slider,6,0,1,3)
        layout.addWidget(self.x_value_line,6,3,1,1)
        
        layout.addWidget(self.y_sliderLabel,4,4,1,1)
        layout.addWidget(self.y_minVal,5,4,1,1)
        layout.addWidget(self.y_maxVal,5,6,1,1)
        layout.addWidget(self.y_slider,6,4,1,3)
        layout.addWidget(self.y_value_line,6,7,1,1)
        

            
    def move(self):
        """moves the 3 axis to the positions specified by the sliders"""
        full_signal=[]
        for chan in self.active_channels:
            slider=getattr(self,chan+"_slider")

            new_pos = slider.value()
            new_pos/=100
            current_pos=getattr(self,chan)
            if current_pos!=new_pos:
                signal=make_ramp(current_pos,new_pos,self.n_samples)[0]
                print("in if, signal shape",signal.shape)
            else:
                signal = current_pos * np.ones(self.n_samples)
                print("in else, shape",signal.shape)
            setattr(self,chan,new_pos)
            full_signal=np.concatenate((full_signal,signal))
            
        self.aoTask.write(full_signal,layout = 'group_by_channel',auto_start=True)
            
    def changePMTsensitivity(self):
        """Sets the sensitivity of the PMT to the value specified by the corresponding slider"""
        value = self.pmt_slider.value()/100
        self.pmt_value_line.setText(str(value))
                    
        if self.pmt_sensitivity != value:
            signalpmt = make_ramp(self.pmt_sensitivity,value,self.n_samples)[0]
            self.sensitivityTask.write(signalpmt)
            self.pmt_sensitivity = value
        
    def move_x(self):
        """Specifies the movement of the x axis."""
        value = self.x_slider.value()/100
        self.x_value_line.setText(str(round(value*x_factor,2)))
        print("move x")
        self.move()
    def move_y(self):
        """Specifies the movement of the y axis."""
        value = self.y_slider.value()/100
        self.y_value_line.setText(str(round(value*y_factor,2)))
        self.move()
    def move_z(self):
        """Specifies the movement of the z axis."""
        value = self.z_slider.value()/100
        self.z_value_line.setText(str(round(value*z_factor,2)))
        self.move()
    
    def editx(self):
        """Method called when a position for x is entered manually. Repositions the slider
        and initiates the movement of the stage"""
        self.x_slider.setValue(100*float(self.x_value_line.text()) / x_factor)
        self.move()
    def edity(self):
        """Method called when a position for y is entered manually. Repositions the slider
        and initiates the movement of the stage"""
        self.y_slider.setValue(100*float(self.y_value_line.text()) / y_factor)
        self.move()
    def editz(self):
        """Method called when a position for z is entered manually. Repositions the slider
        and initiates the movement of the stage"""
        self.z_slider.setValue(100*float(self.z_value_line.text()) / z_factor)    
        self.move()
        
    def go_to_zero(self):
        self.x=0
        self.y=0
        self.z=0
        self.pmt_sensitivity=0
        self.move()
        return

    def reset_channels(self,channels):
        """Method called when the analog output channels need to be used by another resource, typically for scanning. 
        Deactivates the Positionner when it is active and reactives it when it is not, typically after a scan.
        
        :param dict channels: the channels which are used or released by another object. The positionner does not touch the other channels"""
        if(self.isActive):
            print("disabling channels")
            self.aoTask.stop()
            self.aoTask.clear()
            del self.aoTask
            total_channels=["x","y","z"]
            self.active_channels = [x for x in total_channels if not x in channels]     #returns a list containing the axis not in use
            self.aoTask = nidaqmx.AnalogOutputTask("positionnerTask")
            axis=self.active_channels[0]
            channel="Dev1/ao"+str(self.scanWidget.current_aochannels[ axis ])
            self.aoTask.create_voltage_channel(channel,channel_name=axis,min_val= minimum_voltages[axis],max_val=maximum_voltages[axis])
            self.isActive = False
            
        else:
                    #Restarting the analog channels
            print("restarting channels")
            self.aoTask.stop()
            self.aoTask.clear()
            del self.aoTask
            self.aoTask = nidaqmx.AnalogOutputTask("positionnerTask")
            
            total_channels=["x","y","z"]
            self.active_channels=total_channels
            for elt in total_channels :
                channel="Dev1/ao"+str(self.scanWidget.current_aochannels[ elt ])
                self.aoTask.create_voltage_channel( channel, channel_name=elt , min_val=minimum_voltages[elt], max_val= maximum_voltages[elt])
                
#                xchan="Dev1/ao"+str(self.scanWidget.current_aochannels["x"])
#                self.aoTask.create_voltage_channel( xchan, channel_name="x" , min_val=min_ao_horizontal, max_val= max_ao_horizontal)
#                
#        
#                ychan="Dev1/ao"+str(self.scanWidget.current_aochannels["y"])
#                self.aoTask.create_voltage_channel( ychan , channel_name="y",min_val=min_ao_horizontal, max_val= max_ao_horizontal)
#                
#                zchan="Dev1/ao"+str(self.scanWidget.current_aochannels["z"])
#                self.aoTask.create_voltage_channel( zchan, channel_name="z" , min_val=min_ao_vertical, max_val= max_ao_vertical)
            
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
   
def record_from_sine(acquisition_signal, ref_signal, number_of_points):
    """Distributes the values acquired in acquisition_signal according to the corresponding
    position measured in ref_signal"""
    volt_range=max(ref_signal)-min(ref_signal)
    results = np.zeros(number_of_points)
    print("real amplitude:",volt_range * correction_factors["x"]*2)
    if volt_range!=max(ref_signal)-min(ref_signal):
        print("bad synchronization in voltage range")
        print("max",max(ref_signal),ref_signal[-1],"min",min(ref_signal),ref_signal[0])
    voltage_increment = -volt_range / number_of_points
    print("voltage incr",voltage_increment)
    counter=0
    pixel=0
    number_samples_in_pix=0
    current_pos=ref_signal[0]
    """pixel corresponds to the pixel the loop fills.
    counter is the numero of the iteration, to know at which instance of the array we are at each moment
    number_samples_in_pix is the number of samples which contributed to the current pixel, used for normalization
    current_pos is in which voltage interval we are. the loop always works between current_pos and current_pos+voltage_increment"""
    lut=np.zeros(number_of_points)
    print("acquisition signal shae",acquisition_signal.shape)
    print("number of points:",number_of_points,"ref shape",ref_signal.shape)
    for measure in acquisition_signal:
        results[pixel]+= measure
        number_samples_in_pix+=1
        if ref_signal[counter]<current_pos+voltage_increment and pixel<number_of_points:
            current_pos+=voltage_increment
            results[pixel]/=number_samples_in_pix
            pixel+=1
            number_samples_in_pix=0
        if pixel> lut.shape[0]-1:
            print("pixel out of range in record from sine")
            pixel= number_of_points-1
            
        
        
        lut[pixel]=counter            
        counter+=1
    lut=lut.astype(np.int)
    np.save(r"C:\Users\aurelien.barbotin\Documents\Data\signal_ref.npy",ref_signal)
    return lut,results
        
def line_with_lut(data,lut):
    """creates line from an LUT"""
    spli=np.split(data,lut)
    result=[np.mean(x) for x in spli]
    result=np.asarray(result)
    return result
    
def phase_correction(frequency):
    """models the frequency-dependant phase shift of a Piezoconcept LFSH2 stage, x axis(fast)
    
    :param float frequency: frequency of the driving signal"""
    coeffs=[  1.06208989e-09,   4.13293782e-07,  -1.65441906e-04,
         2.35337482e-02,  -3.73010981e-02]
    polynom=np.poly1d(coeffs)
    phase = polynom(frequency)
    return phase

def line_from_sine(detector_signal,stage_sensor_signal,number_of_pixels,scan_amplitude,initial_position):
    """This function takes mainly as input 2 arrays of same length, one corresponding to the position measured by a sensor at 
    time t and the other giving the intensity measured by the detector at this same time. From this, we reconstruct the image 
    measured while scanning oer a line, no matter which driving signal we use.
    
    :param numpy.ndarray detector_signal: array containing the signal measured
    :param numpy.ndarray stage_sensor_signal: array cotianing the positions of the stage measured by its sensor
    :param int number_of_pixels: the number of pixels into which the data will be split
    :param float scan_amplitude: the target scan amplitude converted in volts
    :param float initial_position: the initial position of the stage in volts"""
    min_val=initial_position-scan_amplitude/2
    voltage_increment= scan_amplitude/(number_of_pixels)
    number_of_samples_per_pixel=np.zeros(number_of_pixels)    
    image_line = np.zeros(number_of_pixels)    
    stage_sensor_signal = (stage_sensor_signal-5)*2     #Converts the stage sensor signal into a signal similar to the driving one
    stage_sensor_signal= stage_sensor_signal-min_val   #Takes only positive values, normally from zero to max_val
    for index,value in enumerate(detector_signal):
        pixel_pos = np.trunc(stage_sensor_signal[index]/voltage_increment)
        if pixel_pos< number_of_pixels and pixel_pos>=0:
            image_line[pixel_pos]+=value
            number_of_samples_per_pixel[pixel_pos]+=1
    image_line/=number_of_samples_per_pixel
    return np.nan_to_num(image_line)
    
def amplitude_correction(fraction_removed,frequency):
    """corrects the amplitude to take into account the sample we throw away and the response of the stage
    
    :param float fraction_removed: fraction of a cosine removed at the beginning and end of acquisition of a line
    :param float frequency: scanning frequency in Hz"""
    cosine_factor = 1/np.cos(np.pi*fraction_removed)
    coeffs = [  3.72521796e-12,  -1.27313251e-09,   1.57438425e-07,
        -7.70042004e-06,   5.38779963e-05,  -8.34837794e-04,
         1.00054532e+00]
    polynom = np.poly1d(coeffs)
    frequency_factor= 1 /polynom(frequency)
    return frequency_factor * cosine_factor
    
    


if __name__=="main":
    app = QtGui.QApplication([])
    nidaq = nidaqmx.Device("Dev1")
    win = ScanWidget(nidaq)
    win.show()
    app.exec_()