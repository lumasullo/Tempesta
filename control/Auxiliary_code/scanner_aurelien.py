# -*- coding: utf-8 -*-


import os
import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from tkinter import Tk, filedialog
from PIL import Image
import control.guitools as guitools
import datetime

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
refTrigger = 'PFI12'

fracRemoved = 0.1

# Resonance frequency of the stage for amplitude
f_0 = 53
# To save the sensor data
recSensorOutput = False
saveFolder = r"C:\Users\aurelien.barbotin\Documents\Data\signals_15_8"

# These dictionnaries contain values specific to the different axis of our
# piezo motors.
# corrFactors: for each direction, corresponds to the movement in
# µm induced by a command of 1V
corrFactors = {'x': 4.06, 'y': 3.9, 'z': 10}

# minimum and maximum voltages which can drive the different axis
minVolt = {'x': -10, 'y': -10, 'z': 0}
maxVolt = {'x': 10, 'y': 10, 'z': 10}

PMTsensitivityChan = 'Dev1/ao3'

# sampleRate = 10**5
# TODO: Time in ms, distance in µm


class ScanWidget(QtGui.QFrame):
    """Class generating the GUI for stage scanning. This GUI is intended to
    specify the different parameters such as pixel dwell time, step size,
    scanning width and height etc. This class is intended as a widget in the
    bigger GUI.

    :param nidaqmx.Device device: NiDaq card.
    :param QtGui.QMainWindow main: main GUI
    """

    def __init__(self, device, main, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nidaq = device
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        self.main = main  # The main GUI

        # Creating the GUI itself
        initWidthHeight = 1
        initStep = 0.05
        self.widthPar = QtGui.QLineEdit(str(initWidthHeight))
        self.widthPar.editingFinished.connect(
            lambda: self.scanParameterChanged('width'))
        self.heightPar = QtGui.QLineEdit(str(initWidthHeight))
        self.heightPar.editingFinished.connect(
            lambda: self.scanParameterChanged('height'))
        self.seqTimePar = QtGui.QLineEdit('100')  # mseconds
        self.seqTimePar.editingFinished.connect(
            lambda: self.scanParameterChanged('seqTime'))
        self.nPlanesPar = QtGui.QLabel(self)
        self.freqLabel = QtGui.QLabel(self)
        self.stepSizePar = QtGui.QLineEdit(str(initStep))
        self.stepSizePar.editingFinished.connect(
            lambda: self.scanParameterChanged('stepSize'))

        steps = int(np.ceil(initWidthHeight**2/initStep**2))
        self.nScanStepsLabel = QtGui.QLabel(str(steps))

        self.sampleRate = 70000  # works until 70000
        self.delay = QtGui.QLineEdit("0")

        self.scanModes = ['xy scan', 'xz scan', 'yz scan']
        self.scanMode = QtGui.QComboBox()
        self.scanMode.addItems(self.scanModes)
        self.scanMode.currentIndexChanged.connect(
            lambda: self.setScanMode(self.scanMode.currentText()))

        self.recDevices = [None, 'APD', 'PMT']
        self.recDevice = QtGui.QComboBox()
        self.recDevice.addItems(self.recDevices)
        self.recDevice.currentIndexChanged.connect(
            lambda: self.setRecDevice(self.recDevice.currentText()))
        self.currRecDevice = None

        # Number of image planes to record when doing a 3D scan
        self.nPlanesPar = QtGui.QLineEdit('1')
        self.nPlanesPar.editingFinished.connect(
            lambda: self.scanParameterChanged('nPlanes'))
        # Counts the iterations while performing a 3D sted scan
        self.sted_scan_counter = 0

        self.scanParameters = {'width': self.widthPar,
                               'height': self.heightPar,
                               'seqTime': self.seqTimePar,
                               'nPlanes': self.nPlanesPar,
                               'stepSize': self.stepSizePar}

        self.scanParValues = {'width': float(self.widthPar.text()),
                              'height': float(self.heightPar.text()),
                              'seqTime': float(self.seqTimePar.text()) / 1000,
                              'nPlanes': int(self.nPlanesPar.text()),
                              'stepSize': float(self.stepSizePar.text())}

        self.start488Par = QtGui.QLineEdit('0')
        self.start488Par.editingFinished.connect(
            lambda: self.pxParameterChanged('start488'))
        self.start473Par = QtGui.QLineEdit('0')
        self.start473Par.editingFinished.connect(
            lambda: self.pxParameterChanged('start473'))
        self.start405Par = QtGui.QLineEdit('0')
        self.start405Par.editingFinished.connect(
            lambda: self.pxParameterChanged('start405'))
        self.startCAMPar = QtGui.QLineEdit('0')
        self.startCAMPar.editingFinished.connect(
            lambda: self.pxParameterChanged('startCAM'))

        self.end488Par = QtGui.QLineEdit('0')
        self.end488Par.editingFinished.connect(
            lambda: self.pxParameterChanged('end488'))
        self.end473Par = QtGui.QLineEdit('0')
        self.end473Par.editingFinished.connect(
            lambda: self.pxParameterChanged('end473'))
        self.end405Par = QtGui.QLineEdit('0')
        self.end405Par.editingFinished.connect(
            lambda: self.pxParameterChanged('end405'))
        self.endCAMPar = QtGui.QLineEdit('0')
        self.endCAMPar.editingFinished.connect(
            lambda: self.pxParameterChanged('endCAM'))

        self.pxParameters = {'start405': self.start405Par,
                             'start473': self.start473Par,
                             'start488': self.start488Par,
                             'startCAM': self.startCAMPar,
                             'end488': self.end488Par,
                             'end473': self.end473Par,
                             'end405': self.end405Par,
                             'endCAM': self.endCAMPar}

        self.pxParValues = {'start405': float(self.start488Par.text()) / 1000,
                            'start473': float(self.start405Par.text()) / 1000,
                            'start488': float(self.start473Par.text()) / 1000,
                            'startCAM': float(self.startCAMPar.text()) / 1000,
                            'end405': float(self.end473Par.text()) / 1000,
                            'end473': float(self.end405Par.text()) / 1000,
                            'end488': float(self.end488Par.text()) / 1000,
                            'endCAM': float(self.endCAMPar.text()) / 1000}

        self.currDOchan = {'405': 0, '473': 2, '488': 3, 'CAM': 4}
        self.currAOchan = {'x': 0, 'y': 1, 'z': 2}
#        self.XchanPar = QtGui.QComboBox()
#        self.XchanPar.addItems(self.aochannels)
#        self.XchanPar.setCurrentIndex(0)
#        self.XchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
#        self.XchanPar.setDisabled(True)
#        self.YchanPar = QtGui.QComboBox()
#        self.YchanPar.addItems(self.aochannels)
#        self.YchanPar.setCurrentIndex(1)
#        self.YchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
#        self.YchanPar.setDisabled(True)
#        self.ZchanPar = QtGui.QComboBox()
#        self.ZchanPar.addItems(self.aochannels)
#        self.ZchanPar.setCurrentIndex(2)
#        self.ZchanPar.currentIndexChanged.connect(self.AOchannelsChanged)
#        self.ZchanPar.setDisabled(True)
#
#        self.chan405Par = QtGui.QComboBox()
#        self.chan405Par.addItems(self.dochannels)
#        self.chan473Par = QtGui.QComboBox()
#        self.chan473Par.addItems(self.dochannels)
#        self.chan488Par = QtGui.QComboBox()
#        self.chan488Par.addItems(self.dochannels)
#        self.chanCAMPar = QtGui.QComboBox()
#        self.chanCAMPar.addItems(self.dochannels)
#        self.DOchanParsDict = {'405': self.chan405Par, '473': self.chan473Par,
#                               '488': self.chan488Par, 'CAM': self.chanCAMPar}
#        for sig in self.currDOchan:
#            self.DOchanParsDict[sig].setCurrentIndex(self.currDOchan[sig])

        self.stageScan = StageScan(self, self.sampleRate)
        self.stageScan.updateFrames(self.scanParValues)
        # Used to update the number of frames displayed on screen
        self.scanParameterChanged("width")

        self.pxCycle = PixelCycle(self.sampleRate)
        self.graph = GraphFrame(self.pxCycle)
        self.positionner = Positionner(self)
        self.updateScan(['405', '473', '488', 'CAM'])

        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contScanRadio = QtGui.QRadioButton('Cont. Scan')
        self.contScanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(
           lambda: self.setScanOrNot(False))

        self.ScanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.ScanButton.clicked.connect(self.scanOrAbort)
        self.PreviewButton = QtGui.QPushButton('Preview')
        self.PreviewButton.clicked.connect(self.previewScan)

        self.display = ImageDisplay(self, (200, 200))

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
#        grid.addWidget(QtGui.QLabel('X channel'), 0, 4)
#        grid.addWidget(self.XchanPar, 0, 5)
        grid.addWidget(QtGui.QLabel('Width (µm):'), 0, 0)
        grid.addWidget(self.widthPar, 0, 1)
        grid.addWidget(QtGui.QLabel('Height (µm):'), 0, 2)
        grid.addWidget(self.heightPar, 0, 3)
#        grid.addWidget(QtGui.QLabel('Y channel'), 1, 4)
#        grid.addWidget(self.YchanPar, 1, 5)
#        grid.addWidget(QtGui.QLabel('Z channel'), 2, 4)
#        grid.addWidget(self.ZchanPar, 2, 5)

        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 1, 0)
        grid.addWidget(self.seqTimePar, 1, 1)
        grid.addWidget(QtGui.QLabel('Scanning steps:'), 1, 2)
        grid.addWidget(self.nScanStepsLabel, 1, 3)
        grid.addWidget(QtGui.QLabel('Line scanning frequency (Hz):'), 2, 2)
        grid.addWidget(self.freqLabel, 2, 3)
        grid.addWidget(QtGui.QLabel('Step size (µm):'), 0, 4)
        grid.addWidget(self.stepSizePar, 0, 5)
#        grid.addWidget(QtGui.QLabel('correction samples:'), 4, 4)
#        grid.addWidget(self.delay, 4, 5)
        grid.addWidget(QtGui.QLabel('Number of planes for 3D scan'), 1, 4)
        grid.addWidget(self.nPlanesPar, 1, 5)

        grid.addWidget(self.scanRadio, 3, 4)
        grid.addWidget(self.contScanRadio, 3, 5)
        grid.addWidget(QtGui.QLabel("Scan mode:"), 2, 4)
        grid.addWidget(self.scanMode, 2, 5)

#        grid.addWidget(QtGui.QLabel("Detector:"), 8, 4)
#        grid.addWidget(self.recDevice, 8, 5)

        grid.addWidget(QtGui.QLabel('Start  (ms):'), 5, 1)
        grid.addWidget(QtGui.QLabel('End  (ms):'), 5, 2)
        grid.addWidget(QtGui.QLabel('405:'), 7, 0)
        grid.addWidget(self.start405Par, 7, 1)
        grid.addWidget(self.end405Par, 7, 2)
        grid.addWidget(QtGui.QLabel('473:'), 8, 0)
        grid.addWidget(self.start473Par, 8, 1)
        grid.addWidget(self.end473Par, 8, 2)
        grid.addWidget(self.contLaserPulsesRadio, 7, 3, 4, 1)
        grid.addWidget(QtGui.QLabel('488:'), 9, 0)
        grid.addWidget(self.start488Par, 9, 1)
        grid.addWidget(self.end488Par, 9, 2)
        grid.addWidget(QtGui.QLabel('CAM:'), 10, 0)
        grid.addWidget(self.startCAMPar, 10, 1)
        grid.addWidget(self.endCAMPar, 10, 2)
        grid.addWidget(self.graph, 11, 0, 1, 6)

        grid.addWidget(self.ScanButton, 12, 0, 1, 3)
        grid.addWidget(self.PreviewButton, 12, 3, 1, 3)
        grid.addWidget(self.positionner, 13, 0, 1, 6)

    @property
    def scanOrNot(self):
        return self._scanOrNot

    @scanOrNot.setter
    def scanOrNot(self, value):
        self.enableScanParameters(value)
        self.ScanButton.setCheckable(not value)

    def enableScanParameters(self, value):
        self.widthPar.setEnabled(value)
        self.heightPar.setEnabled(value)
#        self.seqTimePar.setEnabled(value)
        self.stepSizePar.setEnabled(value)
        if value:
            self.ScanButton.setText('Scan')
        else:
            self.ScanButton.setText('Run')

    def setScanOrNot(self, value):
        self.scanOrNot = value

    def setScanMode(self, mode):
        """Sets the scanning strategy to be implemented by the StageScan class

        :param string mode: xy scan, xz scan or yz scan"""

        self.stageScan.setScanMode(mode)
        self.scanParameterChanged('scanMode')

    def setRecDevice(self, device):
        """sets the name of the device which will be used for scanning.

        :param string device: APD or PMT"""
        self.currRecDevice = device

    def AOchannelsChanged(self):
        """If one analog output channel is changed by the user, makes sure that
        2 tasks are not sent to the same channel."""
        Xchan = self.XchanPar.currentIndex()
        Ychan = self.YchanPar.currentIndex()
        Zchan = self.ZchanPar.currentIndex()
        if Xchan == 0:
            Ychan = 1
        elif Xchan == 1:
            Ychan = 0
        else:
            Xchan = 0
            Ychan = 1
            self.XchanPar.setCurrentIndex(Xchan)
            self.YchanPar.setCurrentIndex(Ychan)
#        count=len(self.aochannels)
#        while( (Zchan == Xchan or Zchan == Ychan) and count>0):
#            Zchan = (Zchan + 1)%len(self.aochannels)
#            self.ZchanPar.setCurrentIndex(Zchan)
#            count-=1
#        if(count == 0):
#            print("couldn't find satisfying channel for Z")
        self.currAOchan['x'] = Xchan
        self.currAOchan['y'] = Ychan
        self.currAOchan['z'] = Zchan

    def DOchannelsChanged(self, sig, new_index):

        for i in self.currDOchan:
            if i != sig and new_index == self.currDOchan[i]:
                self.DOchanParsDict[sig].setCurrentIndex(self.currDOchan[sig])

        self.currDOchan[sig] = self.DOchanParsDict[sig].currentIndex()

    def scanParameterChanged(self, par):
        if par != 'scanMode':
            self.scanParValues[par] = float(self.scanParameters[par].text())
            if par == 'seqTime':
                self.scanParValues[par] *= 0.001
                self.updateScan(['405', '473', '488', 'CAM'])

        self.stageScan.updateFrames(self.scanParValues)
        self.nScanStepsLabel.setText(str(self.stageScan.nScanSteps))
        self.freqLabel.setText(str(self.stageScan.freq))

    def pxParameterChanged(self, par):
        self.pxParValues[par] = float(self.pxParameters[par].text()) / 1000
        device = [par[-3] + par[-2] + par[-1]]
        self.pxCycle.update(device, self.pxParValues, self.stageScan.seqSamps)
        self.graph.update(device)

    def previewScan(self):
        """Displays a matplotlib graph representing the scanning's
        trajectory."""
        print(self.scanParValues)
        self.stageScan.update(self.scanParValues)
        plt.figure()
        plt.plot(self.stageScan.sigDict['x'], self.stageScan.sigDict['y'])
        plt.axis([-0.2, self.scanParValues['width']/corrFactors['x'] + 0.2,
                  -0.2, self.scanParValues['height']/corrFactors['y'] + 0.2])
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.show()

    def scanOrAbort(self):
        if not self.scanning:
            self.prepAndRun()
        else:
            self.scanner.abort()

    def prepAndRun(self):
        """Prepares Tempesta for scanning then starts the scan.
        Only called if scanner is not running (See scanOrAbort function)"""
        if self.scanRadio.isChecked() or self.contScanRadio.isChecked():
            self.stageScan.update(self.scanParValues)
            self.ScanButton.setText('Abort')
            channels_used = [self.stageScan.ax1, self.stageScan.ax2]
            self.positionner.resetChannels(channels_used)
            self.scanner = Scanner(
                self.nidaq, self.stageScan, self.pxCycle, self.currAOchan,
                self.currDOchan, self.currRecDevice, self)
            self.scanner.finalizeDone.connect(self.finalizeDone)
            self.scanner.scanDone.connect(self.scanDone)
            self.scanning = True
            self.scanner.runScan()

        elif self.ScanButton.isChecked():
            self.lasercycle = LaserCycle(self.nidaq, self.pxCycle,
                                         self.currDOchan)
            self.ScanButton.setText('Stop')
            self.lasercycle.run()

        else:
            self.lasercycle.stop()
            self.ScanButton.setText('Run')
            del self.lasercycle

    def scanDone(self):
        """Called when *self.scanner* is done"""
        print('in scanDone()')
        self.ScanButton.setEnabled(False)

    def finalizeDone(self):
        self.ScanButton.setText('Scan')
        self.ScanButton.setEnabled(True)
        print('Scan Done')
        channels_to_reset = [self.stageScan.ax1, self.stageScan.ax2]
        del self.scanner
        self.scanning = False
        self.positionner.resetChannels(channels_to_reset)

    def updateScan(self, devices):
        """Creates a scan with the new parameters"""
        self.stageScan.update(self.scanParValues)
        self.pxCycle.update(devices, self.pxParValues, self.stageScan.seqSamps)

    def stedScan(self):
        """does a stage scan and records data with an APD at the same time to
        record STED images."""
        self.scanRadio.setChecked(True)
        self.ScanButton.setEnabled(False)
        self.scanMode.setCurrentIndex(0)  # Set to xy scan

        if self.sted_scan_counter == 0:
            self.stageScan.update(self.scanParValues)
            channels_used = [self.stageScan.ax1, self.stageScan.ax2]
            self.positionner.resetChannels(channels_used)
#        else:
#            self.display.saveImage("sted_plane_"+str(self.sted_scan_counter))

        if self.sted_scan_counter == self.scanParameters['nPlanes']:
            print("end 3D scan")
            self.sted_scan_counter = 0
            return
        self.scanner = Scanner(self.nidaq, self.stageScan, self.pxCycle,
                               self.currAOchan, self.currDOchan,
                               self.currRecDevice, self)
        self.scanner.scanDone.connect(self.sted_scan)
        self.scanning = True
        self.sted_scan_counter += 1
        self.scanner.runScan()

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        plt.close("all")
#        self.nidaq.reset()
        try:
            self.scanner.waiter.terminate()
        except BaseException:
            pass


class WaitThread(QtCore.QThread):
    """Thread regulating the scanning timing. Is used to pass from one step to
    another after it is finished."""
    waitdoneSignal = QtCore.pyqtSignal()

    def __init__(self, task):
        """:param nidaqmx.AnalogOutputTask aotask: task this thread is waiting
        for."""
        super().__init__()
        self.task = task
        self.wait = True

        self.isRunning = False

    def run(self):
        """runs until *self.aotask* is finished, and then emits the
        waitdoneSignal"""
        print('will wait for aotask')
        self.isRunning = True
        if self.wait:
            self.task.wait_until_done()
        self.wait = True
        self.waitdoneSignal.emit()
        self.isRunning = False

    def stop(self):
        """stops the thread, called in case of manual interruption."""
        self.wait = False


class Scanner(QtCore.QObject):
    """This class plays the role of interface between the software and the
    hardware. It writes the different signals to the electronic cards and
    manages the timing of a scan.

    :param nidaqmx.Device device: NiDaq card
    :param StageScan stageScan: object containing the analog signals to drive
    the stage
    :param PixelCycle pxCycle: object containing the digital signals to
    drive the lasers at each pixel acquisition
    :param dict currAOchan: available analog output channels
    :param dict currDOchan: available digital output channels
    :param string recDevice: the name of the device which will get the
    photons from the scan (APD or PMT)
    :param ScanWidget main: main scan GUI."""

    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()

    def __init__(self, device, stageScan, pxCycle, currAOchan, currDOchan,
                 recDevice, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pxCycle = pxCycle
        # Dict containing channel numbers to be written to for each signal
        self.currAOchan = currAOchan
        # Dict containing channel numbers to be written to for each device
        self.currDOchan = currDOchan
        self.sampsInScan = len(self.stageScan.sigDict[self.stageScan.ax1])
        self.recDevice = recDevice
        self.main = main

        self.aotask = nidaqmx.Task('aotask')
        self.dotask = nidaqmx.Task('dotask')
        self.trigger = nidaqmx.Task('trigger')
        self.waiter = WaitThread(self.aotask)
        if self.recDevice == "PMT":
            self.recThread = RecThreadPMT(self.main.display)
        elif self.recDevice == "APD":
            self.recThread = RecThreadAPD(self.main.display)

        # Boolean specifying if we are running a continuous scanning or not
        self.contScan = self.main.contScanRadio.isChecked()

        self.warning_time = 10
        self.scantimewar = QtGui.QMessageBox()
        text = "Are you sure you want to continue?"
        self.scantimewar.setInformativeText(text)
        self.scantimewar.setStandardButtons(QtGui.QMessageBox.Yes |
                                            QtGui.QMessageBox.No)

#        self.connect(self.recThread, QtCore.SIGNAL("measure(float)"),
#                     self.main.display.setPxValue)
#        self.connect(self.recThread, QtCore.SIGNAL("line(PyQt_PyObject)"),
#                     self.main.display.setLineValue)

    def runScan(self):
        """Called when the run button is pressed. Prepares the display, the
        recording thread for acquisition, and writes the values of the scan in
        the corresponding tasks"""
        self.nScanSteps = self.stageScan.nScanSteps

#        image_shape = (self.stageScan.steps2, self.stageScan.steps1)
#        self.main.display.updateParameters(image_shape)

        scan_time = self.sampsInScan / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scantimewar.setText("Scan will last %s seconds" % scan_time)
        if scan_time > self.warning_time:
            ret = self.scantimewar.exec_()

        if ret == QtGui.QMessageBox.No:
            self.contScan = False
            self.done()
            return

        tempAOchan = copy.copy(self.currAOchan)

        # Creates the voltage channels in smallest to largest order and places
        # signals in same order.
        ax1 = self.stageScan.ax1
        ax2 = self.stageScan.ax2

        # typically, the first axis is x and
        chanString1 = 'Dev1/ao%s' % tempAOchan[ax1]
        # self.stageScan.ax1 = "x", then the corresponding ao channel is
        # number 0 and chanString1 = 'Dev1/ao1'
        self.aotask.ao_channels.add_ao_voltage_chan(
            physical_channel=chanString1,
            name_to_assign_to_channel='chan1',
            min_val=minVolt[ax1], max_val=maxVolt[ax1])
        signal1 = self.stageScan.sigDict[ax1]
        print("frequency stage scan:", self.stageScan.freq,
              "correc factor", ampCorrection(fracRemoved, self.stageScan.freq))

        chanString2 = 'Dev1/ao%s' % tempAOchan[ax2]
        self.aotask.ao_channels.add_ao_voltage_chan(
            physical_channel=chanString2,
            name_to_assign_to_channel='chan2',
            min_val=minVolt[ax2], max_val=maxVolt[ax2])
        signal2 = self.stageScan.sigDict[ax2]
        print("length signals:", len(signal1), len(signal2))

        # Generate the delay samples in the end of the scan
        freq = self.stageScan.freq
        # elimination of 1/4 period at the beginning
        self.delay = self.stageScan.sampleRate / freq / 2
        self.delay += phaseCorr(freq)/freq*self.stageScan.sampleRate/2/np.pi
        self.delay = int(self.delay)

        sine = np.arange(self.delay)/self.delay*2*np.pi*freq \
            / self.stageScan.sampleRate
        sine = np.sin(sine) * self.stageScan.size1 / 2
        signal1 = np.concatenate((signal1, sine))
        signal2 = np.concatenate((signal2, signal2[-1]*np.ones(self.delay)))

        self.stageScan.sigDict[ax1] = signal1
        self.stageScan.sigDict[ax2] = signal2

        print("delay in AO signal:", self.delay)
        if len(signal1) != len(signal2):
            print("error: wrong signal size")

        if recSensorOutput:
            self.stageScan.sampsPerLine
            name = str(round(self.main.sampleRate /
                             self.stageScan.sampsPerLine)) + "Hz"
            np.save(saveFolder + "\\" + "driving_signal_" + name, signal1)

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan + self.delay)
        self.aotask.triggers.start_trigger.cfg_dig_edge_start_trig(refTrigger)
        self.fullAOsignal = np.vstack((signal1, signal2))
        self.aotask.write(self.fullAOsignal, auto_start=False)

        # Digital signals/devices
        tmpDOchan = copy.copy(self.currDOchan)
        fullDOsignal = np.zeros((len(tmpDOchan), self.pxCycle.cycleSamps),
                                dtype=bool)
        for i in range(0, len(tmpDOchan)):
            dev = min(tmpDOchan, key=tmpDOchan.get)
            chanstring = 'Dev1/port0/line%s' % tmpDOchan[dev]
            self.dotask.do_channels.add_do_chan(
                lines=chanstring, name_to_assign_to_lines='chan%s' % dev)
            tmpDOchan.pop(dev)
            fullDOsignal[i] = self.pxCycle.sigDict[dev]

        self.dotask.timing.cfg_samp_clk_timing(
            rate=self.pxCycle.sampleRate,
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.stageScan.seqSamps)

        self.dotask.write(fullDOsignal, auto_start=False)

        # Trigger
        trigCounter = self.trigger.co_channels.add_co_pulse_chan_ticks(
            'Dev1/ctr1', '', name_to_assign_to_channel="pasteque",
            low_ticks=100000, high_ticks=1000000)
        trigCounter.co_pulse_term = refTrigger

        hwMode = nidaqmx.constants.AcquisitionType.HW_TIMED_SINGLE_POINT
        self.trigger.timing.cfg_samp_clk_timing(rate=1,
                                                source=r'ao/SampleClock',
                                                sample_mode=hwMode)

        self.waiter.waitdoneSignal.connect(self.finalize)

        if self.recDevice is not None:
            self.recThread.setParameters(self.stageScan.seqSamps,
                                         self.sampsInScan,
                                         self.stageScan.sampleRate,
                                         self.stageScan.sampsPerLine,
                                         ax1)
            self.recThread.start()
        time.sleep(0.05)  # Necessary for good synchronization
        self.aotask.start()
        self.dotask.start()
        self.waiter.start()
        self.trigger.start()
#        self.trigger.write(np.append(np.zeros(1000), np.ones(1000)))

    def abort(self):
        """Stops the current scan. Stops the recording thread and calls the
        method finalize"""
        print('Aborting scan')
        self.waiter.stop()
        self.aotask.stop()
        self.dotask.stop()
        if self.recDevice is not None:
            self.recThread.exiting = True
        # To prevent from starting a continuous acquisition in finalize
        self.main.scanRadio.setChecked(True)
        self.finalize()

    def finalize(self):
        """Once the scanning is finished, sends the different devices to the
        position specified by the Positionner."""
        print('in finalize')
        # Boolean specifying if we are running a continuous scanning or not
        self.contScan = self.main.contScanRadio.isChecked()
        if(not self.contScan):
            self.scanDone.emit()
        # Important, otherwise finalize is called again when next waiting
        # finishes.
        self.waiter.waitdoneSignal.disconnect(self.finalize)
        self.waiter.waitdoneSignal.connect(self.done)

        # TODO: Test abort
        writtenSamps = int(np.round(self.aotask.out_stream.curr_write_pos))
        print("written samples", writtenSamps)
        goals = [0, 0]  # Where we write the target positions to return to
        final_1 = self.stageScan.sigDict[self.stageScan.ax1][writtenSamps - 1]
        final_2 = self.stageScan.sigDict[self.stageScan.ax2][writtenSamps - 1]
        goals[0] = getattr(self.main.positionner, self.stageScan.ax1)
        goals[1] = getattr(self.main.positionner, self.stageScan.ax2)

        finalSamps = [final_1, final_2]
        returnTime = 0.05  # Return time of 50ms, it is enough
        samps = int(self.stageScan.sampleRate * returnTime)
        returnRamps = np.zeros((2, samps))
        returnRamps[0] = makeRamp(finalSamps[0], goals[0], samps)[0]
        returnRamps[1] = makeRamp(finalSamps[1], goals[1], samps)[0]

        self.aotask.stop()

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=samps)

        self.aotask.triggers.start_trigger.cfg_dig_edge_start_trig(refTrigger)
        self.aotask.write(returnRamps, auto_start=False)

        self.aotask.start()
        self.waiter.start()

    def done(self):
        """Once the different scans are in their initial position, starts again
        a new scanning session if in continuous scan mode. If not, it releases
        the channels for the positionner."""
        print('in self.done()')
        if(self.contScan):
            print("in contScan")
            # If scanning continuously, regenerate the samples and write them
            # again
            self.aotask.stop()
            self.dotask.stop()

            self.dotask.timing.cfg_samp_clk_timing(
                rate=self.pxCycle.sampleRate,
                source=r'ao/SampleClock',
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=self.stageScan.seqSamps)
            self.trigger.stop()
            self.waiter.waitdoneSignal.disconnect(self.done)
            self.waiter.waitdoneSignal.connect(self.finalize)

            self.aotask.write(self.fullAOsignal, auto_start=False)
            self.dotask.start()

            self.recThread.start()
            self.aotask.start()
            self.waiter.start()
            time.sleep(0.01)
            self.trigger.start()
        else:
            self.aotask.close()
            self.dotask.close()
            self.trigger.stop()
            self.trigger.close()
            del self.trigger
            if self.recDevice is not None:
                self.recThread.stop()
            self.nidaq.reset_device()
            self.finalizeDone.emit()
            print("total Done")


class LaserCycle():

    def __init__(self, device, pxCycle, currDOchan):

        self.nidaq = device
        self.pxCycle = pxCycle
        self.currDOchan = currDOchan

    def run(self):
        self.dotask = nidaqmx.Task('dotask')

        tmpDOchan = copy.copy(self.currDOchan)
        fullDOsignal = np.zeros((len(tmpDOchan), self.pxCycle.cycleSamps),
                                dtype=bool)
        for i in range(0, len(tmpDOchan)):
            dev = min(tmpDOchan, key=tmpDOchan.get)
            chanstring = 'Dev1/port0/line%s' % tmpDOchan[dev]
            self.dotask.do_channels.add_do_chan(
                lines=chanstring,
                name_to_assign_to_lines='chan%s' % dev)
            tmpDOchan.pop(dev)
            fullDOsignal[i] = self.pxCycle.sigDict[dev]

        self.dotask.timing.cfg_samp_clk_timing(
           source=r'100kHzTimeBase',
           rate=self.pxCycle.sampleRate,
           sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        self.dotask.write(fullDOsignal, auto_start=False)

        self.dotask.start()

    def stop(self):
        self.dotask.stop()
        self.dotask.close()
        del self.dotask
        self.nidaq.reset()


class StageScan():
    """Scanning in xy

    :param ScanWidget main: main scan GUI
    :param float sampleRate: sample rate in samples per second"""

    def __init__(self, main, sampleRate):
        self.scanMode = 'xy scan'
        self.sigDict = {'x': [], 'y': [], 'z': []}
        self.sampleRate = sampleRate
        self.corrStepSize = None
        self.seqSamps = None
        self.nScanSteps = 0

        self.ax1 = self.scanMode[0]  # x if in '[x]y scan'
        self.ax2 = self.scanMode[1]  # y if in 'x[y] scan'

        self.size1 = 0
        self.size2 = 0

        self.scanWidget = main

        self.steps1 = 0
        self.steps2 = 0
        self.sampsPerLine = 0

        self.freq = 0

    def setScanMode(self, mode):
        self.scanMode = mode
        self.ax1 = self.scanMode[0]  # x for instance
        self.ax2 = self.scanMode[1]  # y for instance

    def updateFrames(self, parValues):
        self.steps1 = int(np.ceil(parValues['width']/parValues['stepSize']))
        self.steps2 = int(np.ceil(parValues['height']/parValues['stepSize']))
        self.nScanSteps = self.steps1 * self.steps2

#        stepSize1 = parValues['stepSize'] / corrFactors[self.ax1]
        self.seqSamps = int(np.ceil(self.sampleRate*parValues['seqTime']))
        if self.seqSamps == 1:
            self.seqSamps += 1
        rowSamps = self.steps1 * self.seqSamps
        self.freq = round(self.sampleRate / (rowSamps * 2), 1)

    def update(self, parValues):
        """creates the signals inside the dictionnary self.sigDict

        :param dict parValues:contains the name and value of the scanning
        parameters"""
        # Create signals
        try:
            start1 = getattr(self.scanWidget.positionner, self.ax1)
            start2 = getattr(self.scanWidget.positionner, self.ax2)
        except BaseException:
            start1 = 0
            start2 = 0
            print("couldn't access to the positionner")
        self.size1 = parValues['width'] / corrFactors[self.ax1]
        self.size2 = parValues['height'] / corrFactors[self.ax2]
        stepSize1 = parValues['stepSize'] / corrFactors[self.ax1]
        stepSize2 = parValues['stepSize'] / corrFactors[self.ax2]

        # We want at least two samples per point
        self.seqSamps = int(np.ceil(self.sampleRate*parValues['seqTime']))
        if self.seqSamps == 1:
            self.seqSamps += 1
            print("not enough samples")
        self.steps1 = int(np.ceil(self.size1 / stepSize1))
        self.steps2 = int(np.ceil(self.size2 / stepSize2))
#        self.nScanSteps = self.steps1 * self.steps2

        # Step size compatible with width
        self.corrStepSize = self.size2 / self.steps2
        rowSamps = self.steps1 * self.seqSamps
        sig1 = []
        sig2 = []

        newValue = start2
        nSamplesRamp = int(2 * fracRemoved * rowSamps)
        nSampsFlat = int((rowSamps - nSamplesRamp) / 2)
#        nSampsFlat_2 = rowSamps - nSampsFlat - nSamplesRamp
        rampAx2 = makeRamp(0, self.corrStepSize, nSamplesRamp)[0]

        self.freq = self.sampleRate / (rowSamps * 2)

        # Not sure whether I need the APD or PMT version. Here it's APD
        nSamplesRamp = int(2 * fracRemoved * rowSamps)
        nSampsFlat = int((2 * rowSamps - nSamplesRamp))
        # sine scanning
        sine = np.arange(0, 2*rowSamps*self.steps2)/(rowSamps*2)*2*np.pi
        # Sine varies from -1 to 1 so need to divide by 2
        sine = np.sin(sine) * self.size1 / 2
        for i in range(0, self.steps2):
            sig2 = np.concatenate(
                (sig2, newValue*np.ones(nSampsFlat), newValue + rampAx2))
            newValue = newValue + self.corrStepSize
            self.sampsPerLine = 2 * rowSamps

#        if self.scanWidget.currRecDevice == "APD":
#            nSamplesRamp = int(2 * fracRemoved * rowSamps)
#            nSampsFlat = int((2 * rowSamps - nSamplesRamp))
#            # sine scanning
#            sine = np.arange(0, 2*rowSamps*self.steps2)/(rowSamps*2)*2*np.pi
#            # Sine varies from -1 to 1 so need to divide by 2
#            sine = np.sin(sine) * self.size1 / 2
#            for i in range(0, self.steps2):
#                sig2 = np.concatenate(
#                    (sig2, newValue * np.ones(nSampsFlat),
#                     newValue + rampAx2))
#                newValue = newValue + self.corrStepSize
#                self.sampsPerLine = 2 * rowSamps
#        else self.scanWidget.currRecDevice == "PMT":
#            for i in range(0, self.steps2):
#                # sine scanning
#                sine = np.arange(0, rowSamps*self.steps2)/(rowSamps*2)*2*np.pi
#                # Sine varies from -1 to 1 so need to divide by 2
#                sine = np.sin(sine) * self.size1 / 2
#                sig2 = np.concatenate(
#                    (sig2,
#                     newValue * np.ones(nSampsFlat),
#                     newValue + rampAx2,
#                     self.corrStepSize + newValue*np.ones(nSampsFlat_2))
#                    )
#                newValue = newValue + self.corrStepSize
#                self.sampsPerLine = rowSamps
#
#        print("sequence samples:", self.seqSamps)
#        # Correction for amplitude:
        sig1 = sine * ampCorrection(fracRemoved, self.freq)
        sig1 += start1
        # Assign signal to axis 1
        self.sigDict[self.ax1] = sig1
        # Assign signal to axis 2
        self.sigDict[self.ax2] = sig2


class PixelCycle():
    """Contains the digital signals for the pixel cycle, ie the process
    repeated for the acquisition of each pixel.
    The update function takes a parameter_values dict and updates the signal
    accordingly.

    :param float sampleRate: sample rate in samples per seconds"""

    def __init__(self, sampleRate):
        self.sigDict = {'405': [], '473': [], '488': [], 'CAM': []}
        self.sampleRate = sampleRate
        self.cycleSamps = 0

    def update(self, devices, parValues, cycleSamps):
        self.cycleSamps = cycleSamps
        for device in devices:
            signal = np.zeros(self.cycleSamps)
            start_name = 'start' + device
            end_name = 'end' + device
            start_pos = parValues[start_name] * self.sampleRate
            start_pos = int(min(start_pos, self.cycleSamps - 1))
            end_pos = parValues[end_name] * self.sampleRate
            end_pos = int(min(end_pos, self.cycleSamps))
            signal[range(start_pos, end_pos)] = 1
            self.sigDict[device] = signal


class GraphFrame(pg.GraphicsWindow):
    """Class is child of pg.GraphicsWindow and creats the plot that plots the
    preview of the pulses.
    Fcn update() updates the plot of "device" with signal "signal"  """

    def __init__(self, pxCycle, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pxCycle = pxCycle
        self.plot = self.addPlot(row=1, col=0)
        self.plot.showGrid(x=False, y=False)
        self.plotSigDict = {'405': self.plot.plot(pen=pg.mkPen(73, 0, 188)),
                            '473': self.plot.plot(pen=pg.mkPen(0, 247, 255)),
                            '488': self.plot.plot(pen=pg.mkPen(97, 0, 97)),
                            'CAM': self.plot.plot(pen='w')}

    def update(self, devices=None):
        if devices is None:
            devices = self.plotSigDict

        for device in devices:
            signal = self.pxCycle.sigDict[device]
            self.plotSigDict[device].setData(signal)


class ImageDisplay(QtGui.QWidget):
    """Class creating a display for the images obtained either with an APD or
    PMT

    :param ScanWidget main: main scan GUI
    :param tuple shape: width and height of the image."""

    def __init__(self, main, shape):
        super().__init__()

        self.setWindowTitle("Image from scanning")

        self.array = np.zeros((shape[1], shape[0]))
        self.shape = (shape[1], shape[0])
        self.pos = [0, 0]
        self.scanWidget = main

        # File management
        self.initialDir = os.getcwd()
        self.saveButton = QtGui.QPushButton("Save image")
        self.saveButton.clicked.connect(self.saveImage)

        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        self.browseButton = QtGui.QPushButton("Choose folder")
        self.browseButton.clicked.connect(self.loadFolder)

        # Visualisation widget
        self.graph = pg.GraphicsLayoutWidget()
        self.vb = self.graph.addPlot(row=1, col=1)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.img.translate(-0.5, -0.5)
        self.vb.setAspectLocked(True)
        self.img.setImage(self.array)
        self.vb.setMinimumHeight(200)

        self.ROI = guitools.ROI((10, 10), self.vb, (0, 0), handlePos=(1, 0),
                                handleCenter=(0, 1), scaleSnap=True,
                                translateSnap=True, color='w')
#        self.ROI.sigRegionChangeFinished.connect(self.ROIchanged)
        self.ROI.hide()
        self.ROI_is_displayed = False
        # the position of the Positionner when start recording, in V
        self.initPosition = [0, 0]

        self.ROI_show_button = QtGui.QPushButton("Select zone")
        self.ROI_show_button.clicked.connect(self.showROI)
        self.ROI_go_button = QtGui.QPushButton("Prep. scan")
        self.ROI_go_button.clicked.connect(self.setScanArea)

        # To get intensity profile along a line
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.graph.addItem(self.hist, row=1, col=2)

        self.profile_plot = self.graph.addPlot(row=2, col=1)
        self.profile_plot.setMaximumHeight(150)
        self.line = pg.LineSegmentROI([[0, 0], [10, 0]], pen='r')
        self.vb.addItem(self.line)
        self.line.sigRegionChanged.connect(self.updatePlot)
        self.updatePlot()

        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updatePlot)
        self.viewtimer.start(50)

        # The size of one pixel
        self.pixel_size = -1
        self.scan_axes = ["x", "y"]
        self.isTurning = False

        layout = QtGui.QGridLayout()
        self.setLayout(layout)

        layout.addWidget(self.graph, 0, 0, 5, 5)
        layout.setRowMinimumHeight(0, 300)
        layout.setColumnMinimumWidth(0, 300)
        layout.addWidget(self.saveButton, 5, 1)
        layout.addWidget(self.browseButton, 5, 2)
        layout.addWidget(self.folderEdit, 5, 0)

        layout.addWidget(self.ROI_show_button, 5, 3)
        layout.addWidget(self.ROI_go_button, 5, 4)

    def updateParameters(self, shape):
        """reshapes the array with the proper dimensions before acquisition

        :param tuple shape: width and height of the image."""
        self.array = np.zeros((shape[1], shape[0]))
        self.shape = (shape[0], shape[1])
        self.pos = [0, 0]
        self.img.setImage(self.array)
        self.pixel_size = float(self.scanWidget.stepSizePar.text())
        scanMode = self.scanWidget.scanMode.currentText()
        self.scan_axes = [scanMode[0], scanMode[1]]
        x_init = getattr(self.scanWidget.positionner, self.scan_axes[0])
        y_init = getattr(self.scanWidget.positionner, self.scan_axes[1])
        self.initPosition = [x_init, y_init]

    def updatePlot(self):
        selected = self.line.getArrayRegion(self.array, self.img)
        self.profile_plot.plot(selected, clear=True, pen=(100, 100, 100),
                               symbolBrush=(255, 0, 0), symbolPen='w')

    def loadFolder(self):
        """Open a window to browse folders, and to select the folder in which
        images are to be saved."""
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if(folder != ''):
                self.folderEdit.setText(folder)
        except BaseException:
            print("We are trying to solve the problem, please wait...")

    def saveImage(self, filename="default"):
        """Saves the current image contained in *self.array* under a predefined
        name containing the main scanning parameters along with the date, under
        the tiff format."""
        im = Image.fromarray(self.array)

        width = self.scanWidget.scanParValues["width"]
        height = self.scanWidget.scanParValues["height"]
        seqTime = self.scanWidget.scanParValues["seqTime"]
        stepSize = self.scanWidget.scanParValues["stepSize"]

        print("filename", filename)
        if filename == "default":
            now = datetime.datetime.now()
            instant_string = str(now.day) + "_" + str(now.month) + "_" + str(
                now.hour) + "h" + str(now.minute) + "_" + str(now.second) + \
                "s_"
            name = instant_string + "fov" + str(width) + "x" + str(height) + \
                "um_seqtime_" + str(seqTime) + "s_stepSize" + \
                str(stepSize) + ".tif"
            print("in default")
        else:
            print("not in default")
        name = filename
        now = datetime.datetime.now()
        instant_string = str(now.day) + "_" + str(now.month) + "_" + \
            str(now.hour) + "h" + str(now.minute) + "_" + str(now.second) + \
            "s_"
        name = instant_string + "fov" + str(width) + "x" + str(height) + \
            "um_seqtime_" + str(seqTime) + "s_stepSize" + \
            str(stepSize) + ".tif"

        print(type(self.folderEdit.text()), type(name), type("\\"))
        im.save(self.folderEdit.text() + "\\" + name)

    def showROI(self):
        """Shows or hide an ROI on the current image to select a specific zone
        to scan"""
        if self.ROI_is_displayed:
            self.ROI.hide()
            self.ROI_show_button.setText("Select Zone")
            self.ROI_is_displayed = False
        else:
            self.ROI.show()
            self.ROI_show_button.setText("Hide ROI")
            self.ROI_is_displayed = True

    def setScanArea(self):
        """gets the position of the ROI and computes appropriate scan
        parameters to scan in this area"""
        if not self.ROI_is_displayed:
            return
        else:
            self.showROI()  # Makes the ROI disappear
        if self.pixel_size > 0:
            width = self.ROI.size()[0] * self.pixel_size
            height = self.ROI.size()[1] * self.pixel_size
            x0 = (self.ROI.pos()[0] - self.shape[0] / 2) * self.pixel_size
            y0 = self.ROI.pos()[1] * self.pixel_size
            print("width", width, height, "x n y", x0, y0)
            max0 = (x0 + width / 2) / corrFactors[self.scan_axes[0]]
            if max0 > maxVolt[self.scan_axes[0]]:
                print("invalid ROI")
                return
            min0 = (x0 - width / 2) / corrFactors[self.scan_axes[0]]
            if min0 < minVolt[self.scan_axes[0]]:
                print("invalid ROI")
                return
            max1 = (y0 + height) / corrFactors[self.scan_axes[1]]
            if max1 > maxVolt[self.scan_axes[1]]:
                print("invalid ROI")
                return
            min1 = y0 / corrFactors[self.scan_axes[1]]
            if min1 < minVolt[self.scan_axes[1]]:
                print("invalid ROI")
                return

            self.scanWidget.widthPar.setText(str(round(width, 2)))
            self.scanWidget.scanParameterChanged("width")
            self.scanWidget.heightPar.setText(str(round(height, 2)))
            self.scanWidget.scanParameterChanged("height")
            print("before")
            # Careful, the position values of the positionner are in V and not
            # in µm
            getattr(self.scanWidget.positionner, "set_" +
                    self.scan_axes[0])(self.initPosition[0] +
                                       (x0 + width / 2) /
                                       corrFactors[self.scan_axes[0]])
            getattr(self.scanWidget.positionner, "set_" +
                    self.scan_axes[1])(self.initPosition[1] +
                                       y0 /
                                       corrFactors[self.scan_axes[0]])
            print("after")

    def setPxValue(self, val):
        """sets the value of one pixel from an input array. Not used anymore:
        point by point filling is too slow, we use setLineValue instead."""
        if not self.isTurning:
            if(self.pos[1] % 2 == 0):
                self.array[self.pos[0], self.pos[1]] = val
            else:
                self.array[self.shape[0] - self.pos[0], self.pos[1]] = val
            self.pos[0] += 1
            if(self.pos[0] > self.shape[0]):
                self.pos[0] = 0
                self.pos[1] += 1
                self.pos[1] = self.pos[1] % (self.shape[1] + 1)
                self.isTurning = True
            self.img.setImage(self.array)
        else:
            self.isTurning = False

    def setLineValue(self, line):
        """Inserts *line* at the proper position in *self.array*

        :param numpy.ndarray line: line of data to insert in the image."""
        line = np.asarray(line)
        self.array[:, self.pos[1]] = line

        self.pos[1] += 1
        self.pos[1] = self.pos[1] % (self.shape[0])

        self.img.setImage(self.array)


class RecThreadAPD(QtCore.QThread):
    """Thread recording an image with an APD (Counter input) while the stage
    is scanning

    :param ImageDisplay display: ImageDisplay in scanwidget"""

    def __init__(self, main):
        super().__init__()
        self.imageDisplay = main
        self.exiting = True
        self.delay = 0

        # Initiation of the analog and counter input tasks necessary to avoid
        # crashing the program in certain cases (see stop)
        self.aitask = 0
        self.citask = 0

    def setParameters(self, seqSamps, sampsPerChan, sampleRate, sampsPerLine,
                      mainAx):
        """prepares the thread for data acquisition with the different
        parameters values

        :param int seqSamps: number of samples for acquisition of 1
        point
        :param int sampsPerChan: Total number of samples generated per
        channel in the scan
        :param int sampleRate: sample rate in number of samples per second
        :param int sampsPerLine: Total number of samples acquired or
        generate per line, to go back and forth.
        :param string mainAx: The axis driven with a sine
        """
        self.sampsPerLine = sampsPerLine
        self.seqSamps = seqSamps
        self.rate = sampleRate
        self.mainAx = mainAx  # Usually it is x
        self.freq = self.rate / self.sampsPerLine
        print("frequency in apd thread", self.freq)
        self.stepsPerLine = self.imageDisplay.shape[1]

        self.nFrames = self.imageDisplay.shape[0]

        self.sampsInScan = sampsPerChan

        try:
            # disables the oscilloscope if it is running
            if self.imageDisplay.scanWidget.main.oscilloscope.isRunning:
                self.imageDisplay.scanWidget.main.oscilloscope.start()
        except BaseException:
            print("error oscilloscope")

        # To record the sensor output
        recChan = 'Dev1/ai5'
        if self.mainAx == "y":
            recChan = 'Dev1/ai6'

        # elimination of 1/4 period at the beginning
        self.delay = self.rate / self.freq / 4
        self.delay += phaseCorr(self.freq) / self.freq * self.rate / 2 / np.pi
        self.delay = int(self.delay)

        self.aitask = nidaqmx.Task()
        self.aitask.create_voltage_channel(
            recChan, min_val=-0.5, max_val=10.0)
        self.aitask.configure_timing_sample_clock(
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            sampsPerChan=self.sampsInScan + self.delay)
        self.aitask.configure_trigger_digital_edge_start(refTrigger)
        print("init citask")
        self.citask = nidaqmx.Task()
        self.citask.create_channel_count_edges("Dev1/ctr0", init=0)
        self.citask.set_terminal_count_edges("Dev1/ctr0", "PFI0")
        self.citask.configure_timing_sample_clock(
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            sampsPerChan=self.sampsInScan + self.delay)
        self.citask.set_arm_start_trigger_source(refTrigger)
        self.citask.set_arm_start_trigger(trigger_type='digital_edge')

    def run(self):
        """runs this thread to acquire the data in task"""
        self.exiting = False
        self.aitask.start()
        self.citask.start()

        print("samples apd acquired before:",
              self.citask.getSampsPerChanAcquired())
        throw_apd_data = self.citask.read(self.delay)
        print("samples apd acquired after:",
              self.citask.getSampsPerChanAcquired())
        # To synchronize analog input and output

        print("self.delay", self.delay)
        counter = self.nFrames
        print("samples per line", self.sampsPerLine)
        last_value = throw_apd_data[-1]

        amplitude = float(self.imageDisplay.scanWidget.widthPar.text(
        )) / corrFactors[self.mainAx]
        initPosition = getattr(self.imageDisplay.scanWidget.positionner,
                               self.mainAx)

        while(counter > 0 and not self.exiting):
            apd_data = self.citask.read(self.sampsPerLine)
            sensor_data = self.aitask.read(self.sampsPerLine, timeout=10)
            sensor_data = sensor_data[:, 0]
            if counter == self.nFrames:
                np.save(
                    r"C:\Users\aurelien.barbotin\Documents\Data\signal5.npy",
                    sensor_data)
            substraction_array = np.concatenate(([last_value], apd_data[:-1]))
            last_value = apd_data[-1]
            # Now apd_data is an array contains at each position the number of
            # counts at this position
            apd_data = apd_data - substraction_array

            length_signal = len(sensor_data) // 2
            apd_data = np.absolute(apd_data[0:length_signal])
            sensor_data = sensor_data[0:length_signal]

            line = lineFromSine(apd_data, sensor_data, self.stepsPerLine,
                                amplitude, initPosition)
            self.emit(QtCore.SIGNAL("line(PyQt_PyObject)"), line)
            if counter < 6:
                print("counter", counter)
            counter -= 1
        self.exiting = True
        self.aitask.stop()
        self.citask.stop()

    def stop(self):
        self.exiting = True
        if self.aitask != 0:
            self.aitask.stop()
            del self.aitask
        if self.citask != 0:
            self.citask.stop()
            del self.citask


class RecThreadPMT(QtCore.QThread):
    """Thread to record an image with the PMT while the stage is scanning

    :param ImageDisplay main: ImageDisplay in scanwidget."""

    def __init__(self, main):
        super().__init__()

        self.imageDisplay = main
        self.exiting = True
        self.delay = 0

    def setParameters(self, seqSamps, sampsPerChan, sampleRate, sampsPerLine,
                      mainAx):
        """prepares the thread for data acquisition with the different
        parameters values

        :param int seqSamps: number of samples for acquisition of 1
        point
        :param int sampsPerChan: Total number of samples generated per
        channel in the scan
        :param int sampleRate: sample rate in number of samples per second
        :param int sampsPerLine: Total number of samples acquired or
        generate per line, to go back and forth.
        :param string mainAx: The axis driven with a sine
        """
        self.sampsPerLine = sampsPerLine
        self.seqSamps = seqSamps
        self.rate = sampleRate
        self.mainAx = mainAx  # Usually it is x
        self.freq = self.rate / self.sampsPerLine / 2

        self.stepsPerLine = self.imageDisplay.shape[1]

        self.nFrames = self.imageDisplay.shape[0]

        self.sampsInScan = sampsPerChan
        if(self.rate != sampleRate * self.sampsInScan / sampsPerChan):
            print("error arrondi")

        print("parameters for acquisition of data : sample rate",
              self.rate, "sampsPerChan:", self.sampsInScan)
        self.aitask = nidaqmx.Task()
        self.aitask.create_voltage_channel('Dev1/ai0', terminal='rse',
                                           min_val=-1, max_val=10.0)

        # To record the sensor output
        recChan = 'Dev1/ai5'
        if self.mainAx == "y":
            recChan = 'Dev1/ai6'

        self.aitask.create_voltage_channel(
            recChan, terminal='rse', min_val=-0.5, max_val=10.0)
        # elimination of 1/4 period at the beginning
        self.delay = self.rate / self.freq / 4
        self.delay += phaseCorr(self.freq) / self.freq * self.rate / 2 / np.pi
        print("delay", self.delay,
              "value 1", phaseCorr(self.freq)/self.freq * self.rate / 2/np.pi,
              "value 2", self.rate / self.freq / 4)
        self.delay = int(self.delay)

        self.aitask.configure_timing_sample_clock(
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            sampsPerChan=self.sampsInScan + self.delay)

    def run(self):
        """runs this thread to acquire the data in task"""
        self.exiting = False
        self.aitask.start()
        # To synchronize analog input and output
        dat = self.aitask.read(self.delay, timeout=30)
        # To change!!
        counter = self.nFrames
        if recSensorOutput:
            sensorVals = np.zeros(self.sampsInScan + self.delay)
            sensorVals[0:self.delay] = dat[:, 1]

        amplitude = float(self.imageDisplay.scanWidget.widthPar.text())
        amplitude /= corrFactors[self.mainAx]
        initPosition = getattr(
            self.imageDisplay.scanWidget.positionner, self.mainAx)

        while(counter > 0 and not self.exiting):
            data = self.aitask.read(self.sampsPerLine, timeout=10)
            if recSensorOutput:
                init = self.delay + (self.nFrames - counter)*self.sampsPerLine
                sensorVals[init:init + self.sampsPerLine] = data[:, 1]

            line = lineFromSine(data[:, 0], data[:, 1], self.stepsPerLine,
                                amplitude, initPosition)
            self.emit(QtCore.SIGNAL("line(PyQt_PyObject)"), line)
            if counter < 6:
                print("counter:", counter)
            counter -= 1

        if recSensorOutput:
            name = str(round(self.rate / self.sampsPerLine)) + "Hz"
            np.save(saveFolder + "\\" + "sensor_output_x" + name, sensorVals)
        self.aitask.stop()
        self.exiting = True

    def stop(self):
        """Stops the worker"""
        try:
            self.aitask.stop()
            self.aitask.close()
            del self.aitask
        except BaseException:
            pass


class Positionner(QtGui.QWidget):
    """This class communicates with the different analog outputs of the nidaq
    card. When not scanning, it drives the 3 axis x, y and z as well as the PMT
    sensitivity.

    :param ScanWidget main: main scan GUI"""

    def __init__(self, main):
        super().__init__()
        self.scanWidget = main

        # Position of the different devices in V
        self.x = 0
        self.y = 0
        self.z = 0
        self.PMTsensitivity = 0.3

        # Parameters for the ramp (driving signal for the different channels)
        self.rampTime = 800  # Time for each ramp in ms
        self.sampleRate = 10**5
        self.nSamples = int(self.rampTime * 10**-3 * self.sampleRate)

        # This boolean is set to False when tempesta is scanning to prevent
        # this positionner to access the analog output channels
        self.isActive = True
        self.active_channels = ["x", "y", "z"]
#        # PMT control
#        self.pmt_value_line = QtGui.QLineEdit()
#        self.PMTslider = QtGui.QSlider(QtCore.Qt.Horizontal)
#        self.PMTslider.valueChanged.connect(self.changePMTsensitivity)
#        self.PMTslider.setRange(0, 125)
#        self.PMTslider.setTickInterval(10)
#        self.PMTslider.setTickPosition(QtGui.QSlider.TicksBothSides)
#        self.PMTslider.setValue(self.PMTsensitivity * 100)
#
#        self.pmt_value_line.setText(str(self.PMTsensitivity))
#
#        self.pmt_minVal = QtGui.QLabel("0")
#        self.pmt_maxVal = QtGui.QLabel("1.25")
#        self.PMTsliderLabel = QtGui.QLabel("PMT sensitivity")
#
#        # Creating the analog output tasks
#        self.sensitivityTask = nidaqmx.AnalogOutputTask()
#        self.sensitivityTask.create_voltage_channel(
#            PMTsensitivityChan, min_val=0, max_val=1.25)
#        self.sensitivityTask.start()
#        self.sensitivityTask.write(self.PMTsensitivity, auto_start=True)

        self.aotask = nidaqmx.Task("positionnerTask")

        xchan = "Dev1/ao" + str(self.scanWidget.currAOchan["x"])
        self.aotask.ao_channels.add_ao_voltage_chan(
            physical_channel=xchan, name_to_assign_to_channel='x',
            min_val=minVolt['x'], max_val=maxVolt['x'])

        ychan = "Dev1/ao" + str(self.scanWidget.currAOchan["y"])
        self.aotask.ao_channels.add_ao_voltage_chan(
            physical_channel=ychan, name_to_assign_to_channel='y',
            min_val=minVolt['x'], max_val=maxVolt['x'])

        zchan = "Dev1/ao" + str(self.scanWidget.currAOchan["z"])
        self.aotask.ao_channels.add_ao_voltage_chan(
            physical_channel=zchan, name_to_assign_to_channel='z',
            min_val=minVolt['z'], max_val=maxVolt['z'])

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.nSamples)

        self.aotask.start()

        # Axes control
        self.x_value_line = QtGui.QLineEdit()
        self.x_value_line.setText(str(self.x))
        self.x_value_line.editingFinished.connect(self.editX)
        self.x_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.x_slider.sliderReleased.connect(self.moveX)
        self.x_slider.setRange(100*minVolt['x'], 100*maxVolt['x'])
        self.x_slider.setValue(self.x)
        self.x_minVal = QtGui.QLabel("-37.5")
        self.x_maxVal = QtGui.QLabel("37.5")
        self.x_sliderLabel = QtGui.QLabel("x position (µm)")

        self.y_value_line = QtGui.QLineEdit()
        self.y_value_line.setText(str(self.y))
        self.y_value_line.editingFinished.connect(self.editY)
        self.y_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.y_slider.sliderReleased.connect(self.moveY)
        self.y_slider.setRange(100*minVolt['x'], 100*maxVolt['x'])
        self.y_slider.setValue(self.y)
        self.y_minVal = QtGui.QLabel("-37.5")
        self.y_maxVal = QtGui.QLabel("37.5")
        self.y_sliderLabel = QtGui.QLabel("y position (µm)")

        self.z_value_line = QtGui.QLineEdit()
        self.z_value_line.setText(str(self.z))
        self.z_value_line.editingFinished.connect(self.editZ)
        self.z_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.z_slider.sliderReleased.connect(self.moveZ)
        self.z_slider.setRange(100 * minVolt['z'], 100 * maxVolt['z'])
        self.z_slider.setValue(self.z)
        self.z_minVal = QtGui.QLabel("0")
        self.z_maxVal = QtGui.QLabel("100")
        self.z_sliderLabel = QtGui.QLabel("z position (µm)")

        self.title = QtGui.QLabel()
        self.title.setText("Stage Positionner")
        self.title.setStyleSheet("font-size:18px")

        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.title, 0, 0)
#        layout.addWidget(self.PMTsliderLabel, 1, 0, 1, 1)
#        layout.addWidget(self.pmt_minVal, 2, 0, 1, 1)
#        layout.addWidget(self.pmt_maxVal, 2, 2, 1, 1)
#        layout.addWidget(self.PMTslider, 3, 0, 1, 3)
#        layout.addWidget(self.pmt_value_line, 3, 3, 1, 1)

        layout.addWidget(self.z_sliderLabel, 1, 8)
        layout.addWidget(self.z_minVal, 2, 8)
        layout.addWidget(self.z_maxVal, 2, 10)
        layout.addWidget(self.z_slider, 3, 8, 1, 3)
        layout.addWidget(self.z_value_line, 3, 11)

        layout.addWidget(self.x_sliderLabel, 1, 0)
        layout.addWidget(self.x_minVal, 2, 0)
        layout.addWidget(self.x_maxVal, 2, 2)
        layout.addWidget(self.x_slider, 3, 0, 1, 3)
        layout.addWidget(self.x_value_line, 3, 3)

        layout.addWidget(self.y_sliderLabel, 1, 4)
        layout.addWidget(self.y_minVal, 2, 4)
        layout.addWidget(self.y_maxVal, 2, 6)
        layout.addWidget(self.y_slider, 3, 4, 1, 3)
        layout.addWidget(self.y_value_line, 3, 7)

    def move(self):
        """moves the 3 axis to the positions specified by the sliders"""
        full_signal = np.zeros((len(self.active_channels), self.nSamples))
        for chan in self.active_channels:
            slider = getattr(self, chan + "_slider")

            new_pos = 0.01*slider.value()
            currPos = getattr(self, chan)
            if currPos != new_pos:
                signal = makeRamp(currPos, new_pos, self.nSamples)[0]
                print("in if, signal shape", signal.shape)
            else:
                signal = currPos * np.ones(self.nSamples)
                print("in else, shape", signal.shape)
            setattr(self, chan, new_pos)
            full_signal[self.active_channels.index(chan)] = signal

        self.aotask.write(full_signal, auto_start=True)

    def changePMTsensitivity(self):
        """Sets the sensitivity of the PMT to the value specified by the
        corresponding slider"""
        value = self.PMTslider.value() / 100
        self.pmt_value_line.setText(str(value))

        if self.PMTsensitivity != value:
            signalpmt = makeRamp(self.PMTsensitivity, value, self.nSamples)[0]
            self.sensitivityTask.write(signalpmt)
            self.PMTsensitivity = value

    def moveX(self):
        """Specifies the movement of the x axis."""
        value = self.x_slider.value() / 100
        self.x_value_line.setText(str(round(value*corrFactors['x'], 2)))
        print("move x")
        self.move()

    def moveY(self):
        """Specifies the movement of the y axis."""
        value = self.y_slider.value() / 100
        self.y_value_line.setText(str(round(value*corrFactors['y'], 2)))
        self.move()

    def moveZ(self):
        """Specifies the movement of the z axis."""
        value = self.z_slider.value() / 100
        self.z_value_line.setText(str(round(value*corrFactors['z'], 2)))
        self.move()

    def editX(self):
        """Method called when a position for x is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        value = 100*float(self.x_value_line.text())/corrFactors['x']
        self.x_slider.setValue(value)
        self.move()

    def editY(self):
        """Method called when a position for y is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        value = 100*float(self.y_value_line.text())/corrFactors['y']
        self.y_slider.setValue(value)
        self.move()

    def editZ(self):
        """Method called when a position for z is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        value = 100*float(self.z_value_line.text())/corrFactors['z']
        self.z_slider.setValue(value)
        self.move()

    def setX(self, value):
        """This method sets x to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        valueLine = round(value * corrFactors['x'], 2)
        print("in set x", value, valueLine)
        self.x_slider.setValue(value * 100)
        self.x_value_line.setText(str(valueLine))
        self.move()

    def setY(self, value):
        """This method sets y to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        self.y_slider.setValue(value * 100)
        self.y_value_line.setText(str(round(value*corrFactors['y'], 2)))
        self.move()

    def setZ(self, value):
        """This method sets x to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        self.z_slider.setValue(value * 100)
        self.z_value_line.setText(str(round(value*corrFactors['z'], 2)))
        self.move()

    def goToZero(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.PMTsensitivity = 0
        self.move()

    def resetChannels(self, channels):
        """Method called when the analog output channels need to be used by
        another resource, typically for scanning. Deactivates the Positionner
        when it is active and reactives it when it is not, typically after a
        scan.

        :param dict channels: the channels which are used or released by
        another object. The positionner does not touch the other channels"""
        if(self.isActive):
            print("disabling channels")
            self.aotask.stop()
            self.aotask.close()
            del self.aotask
            total_channels = ["x", "y", "z"]
            # returns a list containing the axis not in use
            self.active_channels = [
                x for x in total_channels if x not in channels]
            self.aotask = nidaqmx.Task("positionnerTask")
            axis = self.active_channels[0]
            channel = "Dev1/ao" + str(self.scanWidget.currAOchan[axis])
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel=channel, name_to_assign_to_channel=axis,
                min_val=minVolt[axis], max_val=maxVolt[axis])
            self.isActive = False

        else:
            # Restarting the analog channels
            print("restarting channels")
            self.aotask.stop()
            self.aotask.close()
            del self.aotask
            self.aotask = nidaqmx.Task("positionnerTask")

            total_channels = ["x", "y", "z"]
            self.active_channels = total_channels
            for elt in total_channels:
                channel = "Dev1/ao" + str(self.scanWidget.currAOchan[elt])
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel=channel, name_to_assign_to_channel=elt,
                min_val=minVolt[elt], max_val=maxVolt[elt])

            self.aotask.timing.cfg_samp_clk_timing(
                rate=self.sampleRate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=self.nSamples)
            self.aotask.start()
            self.isActive = True
            print("in reset:", self.aotask)

    def closeEvent(self, *args, **kwargs):
        if(self.isActive):
            # Resets the sliders, which will reset each channel to 0
            print("closeEvent positionner")
            self.PMTslider.setValue(0)
            self.x_slider.setValue(0)
            self.y_slider.setValue(0)
            self.z_slider.setValue(0)
            self.move()
            self.aotask.wait_until_done(timeout=2)
            self.aotask.stop()
            self.aotask.close()
            self.sensitivityTask.stop()
            self.sensitivityTask.close()


class ChannelManager(object):
    """Prototype of a class to manage the channels of the NiDaq card. It
    centralizes the calls the different channels and prevents them from being
    called from 2 or more parts of the code at the same time."""

    def __init_(self):
        super().__init__()
        self.channels = ["x", "y", "z", "pmt", "apd", "PMTsensitivity",
                         "x_sensor", "y_sensor"]
        self.available_channels = self.channels
        self.channels_used = []

    def reserve_channel(self, channel):
        pass

    def release_channel(self, channel):
        pass


def makeRamp(start, end, samples):
    k = (end - start) / (samples - 1)
    ramp = [start + k * i for i in range(0, samples)]
    return np.array([np.asarray(ramp), k])


def recFromSine(acquisition_signal, refSignal, nPoints):
    """Distributes the values acquired in acquisition_signal according to the
    corresponding position measured in refSignal"""
    volt_range = max(refSignal) - min(refSignal)
    results = np.zeros(nPoints)
    print("real amplitude:", volt_range * corrFactors["x"] * 2)
    if volt_range != max(refSignal) - min(refSignal):
        print("bad synchronization in voltage range")
        print("max", max(refSignal), refSignal[-1], "min", min(refSignal),
              refSignal[0])
    voltIncrement = -volt_range / nPoints
    print("voltage incr", voltIncrement)
    counter = 0
    pixel = 0
    nSamplesInPx = 0
    currPos = refSignal[0]
    """pixel corresponds to the pixel the loop fills.
    counter is the numero of the iteration, to know at which instance of the
    array we are at each moment nSamplesInPx is the number of samples
    which contributed to the current pixel, used for normalization
    currPos is in which voltage interval we are. the loop always works
    between currPos and currPos+voltIncrement"""
    lut = np.zeros(nPoints)
    print("acquisition signal shae", acquisition_signal.shape)
    print("number of points:", nPoints, "ref shape", refSignal.shape)
    for measure in acquisition_signal:
        results[pixel] += measure
        nSamplesInPx += 1
        if refSignal[counter] < currPos + voltIncrement and pixel < nPoints:
            currPos += voltIncrement
            results[pixel] /= nSamplesInPx
            pixel += 1
            nSamplesInPx = 0
        if pixel > lut.shape[0] - 1:
            print("pixel out of range in record from sine")
            pixel = nPoints - 1

        lut[pixel] = counter
        counter += 1
    lut = lut.astype(np.int)
    np.save(r"C:\Users\aurelien.barbotin\Documents\Data\signal_ref.npy",
            refSignal)
    return lut, results


def lineWithLUT(data, lut):
    """creates line from an LUT calculated with the method recFromSine"""
    spli = np.split(data, lut)
    result = [np.mean(x) for x in spli]
    result = np.asarray(result)
    return result


def phaseCorr(freq):
    """models the frequency-dependant phase shift of a Piezoconcept LFSH2
    stage, x axis(fast)

    :param float freq: frequency of the driving signal"""
    coeffs = [1.06208989e-09, 4.13293782e-07, -1.65441906e-04,
              2.35337482e-02, -3.73010981e-02]
    polynom = np.poly1d(coeffs)
    phase = polynom(freq)
    return phase


def lineFromSine(detectorSig, stageSensorSig, nPixels, scanAmp, initPosition):
    """This function takes mainly as input 2 arrays of same length, one
    corresponding to the position measured by a sensor at time t and the other
    giving the intensity measured by the detector at this same time. From this,
    we reconstruct the image measured while scanning over a line, no matter
    which driving signal we use.

    :param numpy.ndarray detectorSig: array containing the signal measured
    :param numpy.ndarray stageSensorSig: array cotianing the positions of
    the stage measured by its sensor
    :param int nPixels: the number of pixels into which the data will
    be split
    :param float scanAmp: the target scan amplitude converted in volts
    :param float initPosition: the initial position of the stage in
    volts"""
    min_val = initPosition - scanAmp / 2
    voltIncrement = scanAmp / (nPixels)
    nSamplesPerPixel = np.zeros(nPixels)
    image_line = np.zeros(nPixels)
    # Converts the stage sensor signal into a signal similar to the driving one
    stageSensorSig = (stageSensorSig - 5) * 2
    # Takes only positive values, normally from zero to max_val
    stageSensorSig = stageSensorSig - min_val
    for index, value in enumerate(detectorSig):
        pixelPos = np.trunc(stageSensorSig[index] / voltIncrement)
        if pixelPos < nPixels and pixelPos >= 0:
            image_line[pixelPos] += value
            nSamplesPerPixel[pixelPos] += 1
    image_line /= nSamplesPerPixel
    return np.nan_to_num(image_line)


def ampCorrection(fracRemoved, freq):
    """corrects the amplitude to take into account the sample we throw away and
    the response of the stage

    :param float fracRemoved: fraction of a cosine removed at the
    beginning and end of acquisition of a line
    :param float freq: scanning frequency in Hz"""
    cosFactor = 1 / np.cos(np.pi * fracRemoved)
    coeffs = [3.72521796e-12, -1.27313251e-09, 1.57438425e-07,
              -7.70042004e-06, 5.38779963e-05, -8.34837794e-04,
              1.00054532e+00]
    polynom = np.poly1d(coeffs)
    freqFactor = 1 / polynom(freq)
    return freqFactor * cosFactor
