# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import time

import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

import nidaqmx

import control.guitools as guitools

# These dictionnaries contain values specific to the different axis of our
# piezo motors.
# For each direction, there are the movements in µm induced by a command of 1V
convFactors = {'x': 4.06, 'y': 3.9, 'z': 10}
# Minimum and maximum voltages for the different piezos
minVolt = {'x': -10, 'y': -10, 'z': 0}
maxVolt = {'x': 10, 'y': 10, 'z': 10}


class ScanWidget(QtGui.QMainWindow):
    ''' This class is intended as a widget in the bigger GUI, Thus all the
    commented parameters etc. It contain an instance of stageScan and
    pixel_scan which in turn harbour the analog and digital signals
    respectively.
    The function run is the function that is supposed to start communication
    with the Nidaq through the Scanner object. This object was initially
    written as a QThread object but is not right now.
    As seen in the commened lines of run() I also tried running in a QThread
    created in run().
    The rest of the functions contain mosly GUI related code.'''
    def __init__(self, device, outChannels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.currDOchan, self.currAOchan = outChannels

        self.allDevices = list(self.currDOchan.keys())
        self.saveScanBtn = QtGui.QPushButton('Save Scan')

        def saveScanFcn(): return guitools.saveScan(self)
        self.saveScanBtn.clicked.connect(saveScanFcn)
        self.loadScanBtn = QtGui.QPushButton('Load Scan')

        def loadScanFcn(): return guitools.loadScan(self)
        self.loadScanBtn.clicked.connect(loadScanFcn)

        self.sampleRateEdit = QtGui.QLineEdit()

        self.sizeXPar = QtGui.QLineEdit('1')
        self.sizeXPar.editingFinished.connect(
            lambda: self.scanParameterChanged('sizeX'))
        self.sizeYPar = QtGui.QLineEdit('1')
        self.sizeYPar.editingFinished.connect(
            lambda: self.scanParameterChanged('sizeY'))
        self.sizeZPar = QtGui.QLineEdit('10')
        self.sizeZPar.editingFinished.connect(
            lambda: self.scanParameterChanged('sizeZ'))
        self.seqTimePar = QtGui.QLineEdit('100')  # ms
        self.seqTimePar.editingFinished.connect(
            lambda: self.scanParameterChanged('seqTime'))
        self.nrFramesPar = QtGui.QLabel()
        self.scanDuration = 0
        self.scanDurationLabel = QtGui.QLabel(str(self.scanDuration))
        self.stepSizeXYPar = QtGui.QLineEdit('0.05')
        self.stepSizeXYPar.editingFinished.connect(
            lambda: self.scanParameterChanged('stepSizeXY'))
        self.stepSizeZPar = QtGui.QLineEdit('0.05')
        self.stepSizeZPar.editingFinished.connect(
            lambda: self.scanParameterChanged('stepSizeZ'))
        self.sampleRate = 70000

        self.scanMode = QtGui.QComboBox()
        self.scanModes = ['FOV scan', 'VOL scan', 'Line scan']
        self.scanMode.addItems(self.scanModes)
        self.scanMode.currentIndexChanged.connect(
            lambda: self.setScanMode(self.scanMode.currentText()))

        self.primScanDim = QtGui.QComboBox()
        self.scanDims = ['x', 'y']
        self.primScanDim.addItems(self.scanDims)
        self.primScanDim.currentIndexChanged.connect(
            lambda: self.setPrimScanDim(self.primScanDim.currentText()))

        self.scanPar = {'sizeX': self.sizeXPar,
                        'sizeY': self.sizeYPar,
                        'sizeZ': self.sizeZPar,
                        'seqTime': self.seqTimePar,
                        'stepSizeXY': self.stepSizeXYPar,
                        'stepSizeZ': self.stepSizeZPar}

        self.scanParValues = {'sizeX': float(self.sizeXPar.text()),
                              'sizeY': float(self.sizeYPar.text()),
                              'sizeZ': float(self.sizeZPar.text()),
                              'seqTime': 0.001*float(self.seqTimePar.text()),
                              'stepSizeXY': float(self.stepSizeXYPar.text()),
                              'stepSizeZ': float(self.stepSizeZPar.text())}

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

        self.pxParValues = {'start405': 0.001*float(self.start405Par.text()),
                            'start473': 0.001*float(self.start473Par.text()),
                            'start488': 0.001*float(self.start488Par.text()),
                            'startCAM': 0.001*float(self.startCAMPar.text()),
                            'end488': 0.001*float(self.end488Par.text()),
                            'end473': 0.001*float(self.end473Par.text()),
                            'end405': 0.001*float(self.end405Par.text()),
                            'endCAM': 0.001*float(self.endCAMPar.text())}

        self.stageScan = StageScan(self.sampleRate)
        self.pxCycle = PixelCycle(self.sampleRate)
        self.graph = GraphFrame(self.pxCycle)
        self.graph.plot.getAxis('bottom').setScale(1/self.sampleRate)
        self.updateScan(self.allDevices)
        self.scanParameterChanged('seqTime')

        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(
            lambda: self.setScanOrNot(False))

        self.ScanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.ScanButton.clicked.connect(self.scanOrAbort)
        self.PreviewButton = QtGui.QPushButton('Preview')
        self.PreviewButton.clicked.connect(self.previewScan)

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        grid.addWidget(self.loadScanBtn, 0, 0)
        grid.addWidget(self.saveScanBtn, 0, 1)
        grid.addWidget(QtGui.QLabel('Size X (µm):'), 1, 0)
        grid.addWidget(self.sizeXPar, 1, 1)
        grid.addWidget(QtGui.QLabel('Size Y (µm):'), 2, 0)
        grid.addWidget(self.sizeYPar, 2, 1)
        grid.addWidget(QtGui.QLabel('Size Z (µm):'), 3, 0)
        grid.addWidget(self.sizeZPar, 3, 1)
        grid.addWidget(QtGui.QLabel('Step size XY (µm):'), 1, 2)
        grid.addWidget(self.stepSizeXYPar, 1, 3)
        grid.addWidget(QtGui.QLabel('Step size Z (µm):'), 3, 2)
        grid.addWidget(self.stepSizeZPar, 3, 3)

        grid.addWidget(QtGui.QLabel('Frames in scan:'), 4, 3)
        grid.addWidget(self.nrFramesPar, 4, 4)
        grid.addWidget(QtGui.QLabel('Scan duration (s):'), 5, 3)
        grid.addWidget(self.scanDurationLabel, 5, 4)
        grid.setColumnMinimumWidth(4, 50)

        grid.addWidget(self.scanRadio, 4, 0)
        grid.addWidget(QtGui.QLabel('Scan mode:'), 4, 1)
        grid.addWidget(self.scanMode, 4, 2)
        grid.addWidget(QtGui.QLabel('Primary scan dim:'), 5, 1)
        grid.addWidget(self.primScanDim, 5, 2)
        grid.addWidget(self.contLaserPulsesRadio, 6, 0)

        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 8, 0)
        grid.addWidget(self.seqTimePar, 8, 1)
        grid.addWidget(QtGui.QLabel('Start (ms):'), 9, 1)
        grid.addWidget(QtGui.QLabel('End (ms):'), 9, 2)
        grid.addWidget(QtGui.QLabel('405:'), 10, 0)
        grid.addWidget(self.start405Par, 10, 1)
        grid.addWidget(self.end405Par, 10, 2)
        grid.addWidget(QtGui.QLabel('473:'), 11, 0)
        grid.addWidget(self.start473Par, 11, 1)
        grid.addWidget(self.end473Par, 11, 2)
        grid.addWidget(QtGui.QLabel('488:'), 12, 0)
        grid.addWidget(self.start488Par, 12, 1)
        grid.addWidget(self.end488Par, 12, 2)
        grid.addWidget(QtGui.QLabel('Camera:'), 13, 0)
        grid.addWidget(self.startCAMPar, 13, 1)
        grid.addWidget(self.endCAMPar, 13, 2)

        grid.addWidget(self.graph, 14, 0, 1, 5)
        grid.addWidget(self.PreviewButton, 15, 0)
        grid.addWidget(self.ScanButton, 15, 1)

    @property
    def scanOrNot(self):
        return self._scanOrNot

    @scanOrNot.setter
    def scanOrNot(self, value):
        self.enableScanPars(value)
        self.ScanButton.setCheckable(not value)

    def enableScanPars(self, value):
        self.sizeXPar.setEnabled(value)
        self.sizeYPar.setEnabled(value)
        self.stepSizeXYPar.setEnabled(value)
        self.scanMode.setEnabled(value)
        self.primScanDim.setEnabled(value)
        if value:
            self.ScanButton.setText('Scan')
        else:
            self.ScanButton.setText('Run')

    def setScanOrNot(self, value):
        self.scanOrNot = value

    def setScanMode(self, mode):
        self.stageScan.setScanMode(mode)
        self.scanParameterChanged('scanMode')

    def setPrimScanDim(self, dim):
        self.stageScan.setPrimScanDim(dim)
        self.scanParameterChanged('primScanDim')

    def AOchanChanged(self):
        """Function is obsolete since we never change channels this way,
        Z-channel not implemented"""
#        Xchan = self.XchanPar.currentIndex()
#        Ychan = self.YchanPar.currentIndex()
#        if Xchan == Ychan:
#            Ychan = (Ychan + 1)%4
#            self.YchanPar.setCurrentIndex(Ychan)
#        self.currAOchan['x'] = Xchan
#        self.currAOchan['y'] = Ychan

    def DOchanChanged(self, sig, new_index):
        for i in self.currDOchan:
            if i != sig and new_index == self.currDOchan[i]:
                self.DOchanParsDict[sig].setCurrentIndex(self.currDOchan[sig])

        self.currDOchan[sig] = self.DOchanParsDict[sig].currentIndex()

    def scanParameterChanged(self, p):
        if p not in ('scanMode', 'primScanDim'):
            if p == 'seqTime':
                # To get in seconds
                self.scanParValues[p] = 0.001*float(self.scanPar[p].text())
            else:
                self.scanParValues[p] = float(self.scanPar[p].text())

        if p == 'seqTime':
            self.updateScan(self.allDevices)
            self.graph.update(self.allDevices)
        self.stageScan.updateFrames(self.scanParValues)
        self.nrFramesPar.setText(str(self.stageScan.frames))
        self.scanDuration = (self.stageScan.frames *
                             self.scanParValues['seqTime'])
        self.scanDurationLabel.setText(str(np.round(self.scanDuration, 2)))

    def pxParameterChanged(self, p):
        self.pxParValues[p] = 0.001*float(self.pxParameters[p].text())
        device = [p[-3] + p[-2] + p[-1]]
        self.pxCycle.update(device, self.pxParValues, self.stageScan.seqSamps)
        self.graph.update(device)

    def previewScan(self):
        self.stageScan.update(self.scanParValues)
        plt.plot(self.stageScan.sigDict['x'] * convFactors['x'],
                 self.stageScan.sigDict['y'] * convFactors['y'])
        mx = max(self.scanParValues['sizeX'], self.scanParValues['sizeY'])
        plt.margins(0.1*mx)
        plt.axis('scaled')

        plt.xlabel("x axis [µm]")
        plt.ylabel("y axis [µm]")

    def scanOrAbort(self):
        if not self.scanning:
            self.prepAndRun()
        else:
            self.scanner.abort()

    def prepAndRun(self):
        ''' Only called if scanner is not running (See scanOrAbort funciton).
        '''
        if self.scanRadio.isChecked():
            self.stageScan.update(self.scanParValues)
            self.ScanButton.setText('Abort')
            self.scanner = Scanner(self.nidaq, self.stageScan, self.pxCycle,
                                   self.currAOchan, self.currDOchan, self)
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
        print('in scanDone()')
        self.ScanButton.setEnabled(False)

    def finalizeDone(self):
        self.ScanButton.setText('Scan')
        self.ScanButton.setEnabled(True)
        print('Scan Done')
        del self.scanner
        self.scanning = False

    def updateScan(self, devices):
        self.stageScan.update(self.scanParValues)
        self.pxCycle.update(devices, self.pxParValues, self.stageScan.seqSamps)

    def closeEvent(self, *args, **kwargs):
        try:
            self.scanner.waiter.terminate()
        except BaseException:
            pass


class WaitThread(QtCore.QThread):
    waitdoneSignal = QtCore.pyqtSignal()

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.wait = True

    def run(self):
        if self.wait:
            self.task.wait_until_done(nidaqmx.constants.WAIT_INFINITELY)
        self.wait = True
        self.waitdoneSignal.emit()
        print(self.task.name + ' is done')

    def stop(self):
        self.wait = False


class Scanner(QtCore.QObject):

    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()

    def __init__(self, device, stageScan, pxCycle, currAOchan, currDOchan,
                 main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pxCycle = pxCycle
        # Dict containing channel numbers to be written to for each signal
        self.currAOchan = currAOchan
        # Dict containing channel numbers to be written to for each device
        self.currDOchan = currDOchan
        self.sampsInScan = len(self.stageScan.sigDict['x'])
        self.main = main

        self.aotask = nidaqmx.Task('aotask')
        self.dotask = nidaqmx.Task('dotask')

        self.waiter = WaitThread(self.aotask)

        self.scanTimeW = QtGui.QMessageBox()
        self.scanTimeW.setInformativeText("Are you sure you want to continue?")
        self.scanTimeW.setStandardButtons(QtGui.QMessageBox.Yes |
                                          QtGui.QMessageBox.No)

    def runScan(self):
        scanTime = self.sampsInScan / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scanTimeW.setText("Scan will take %s seconds" % scanTime)
        if scanTime > 10:
            ret = self.scanTimeW.exec_()

        if ret == QtGui.QMessageBox.No:
            self.done()
            return

        fullAOsignal = np.zeros((len(self.currAOchan),
                                 len(self.stageScan.sigDict['x'])))
        tempAOchan = copy.copy(self.currAOchan)
        # Following loop creates the voltage channels in smallest to largest
        # order and places signals in same order.
        for i in range(0, 3):
            # dim = dimension ('x' or 'y') containing smallest channel nr.
            dim = min(tempAOchan, key=tempAOchan.get)
            chanstring = 'Dev1/ao%s' % tempAOchan[dim]
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel=chanstring,
                name_to_assign_to_channel='chan%s' % dim,
                min_val=minVolt[dim], max_val=maxVolt[dim])
            tempAOchan.pop(dim)
            fullAOsignal[i] = self.stageScan.sigDict[dim]

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan)

        # Same as above but for the digital signals/devices
        tempDOchan = copy.copy(self.currDOchan)
        fullDOsignal = np.zeros(
            (len(tempDOchan), len(self.pxCycle.sigDict['405'])), dtype=bool)
        for i in range(0, len(tempDOchan)):
            dev = min(tempDOchan, key=tempDOchan.get)
            chanstring = 'Dev1/port0/line%s' % tempDOchan[dev]
            self.dotask.do_channels.add_do_chan(
                lines=chanstring,
                name_to_assign_to_lines='chan%s' % dev)
            tempDOchan.pop(dev)

            if self.stageScan.scanMode == 'VOLscan':
                signal = np.tile(self.pxCycle.sigDict[dev],
                                 self.stageScan.VOLscan.cyclesPerSlice)
                signal = np.concatenate((signal,
                                         np.zeros(self.stageScan.seqSamps)))
            else:
                signal = self.pxCycle.sigDict[dev]

            fullDOsignal[i] = self.pxCycle.sigDict[dev]

        self.dotask.timing.cfg_samp_clk_timing(
            rate=self.pxCycle.sampleRate,
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan)

        # Following is to create ramps back to zero for the analog channels
        # during one second.
#        returnRamps = np.array([])
#        for i in range(0,2):
#            rampAndK = makeRamp(finalSamps[i], 0, self.stageScan.sampleRate)
#            returnRamps = np.append(returnRamps, rampAndK)
#
#        print(np.ones(1)) # This line alone fixes the problem...

        self.waiter.waitdoneSignal.connect(self.finalize)

        self.dotask.write(fullDOsignal, auto_start=False)
        self.aotask.write(fullAOsignal, auto_start=False)

        self.dotask.start()
        self.aotask.start()
        self.waiter.start()

    def abort(self):
        print('Aborting scan')
        self.waiter.stop()
        self.aotask.stop()
        self.aotask.close()
        self.dotask.stop()
        self.dotask.close()
        self.finalize()

    def finalize(self):
        print('in finalize')
        self.scanDone.emit()
        # Apparently important, otherwise finalize is called again when next
        # waiting finishes.
        try:
            self.waiter.waitdoneSignal.disconnect(self.finalize)
        except TypeError:
            # This happens when the scan is aborted after the warning
            pass
        self.waiter.waitdoneSignal.connect(self.done)

        # Following code should correct channels mentioned in Buglist. Not
        # correct though, assumes channels are 0, 1 and 2.
        # TODO: Test abort (task function)
        writtenSamps = int(np.round(self.aotask.out_stream.curr_write_pos))
        finalSamps = [0, 0, 0]
        tempAOchan = copy.copy(self.currAOchan)
        for i in range(0, 3):
            # dim = dimension ('x', 'y' or 'z') containing smallest channel nr.
            dim = min(tempAOchan, key=tempAOchan.get)
            finalSamps[i] = self.stageScan.sigDict[dim][writtenSamps - 1]
            tempAOchan.pop(dim)

        returnRamps = [makeRamp(finalSamps[i], 0, self.stageScan.sampleRate)
                       for i in range(0, 3)]

        print('task is: ', self.aotask.name)
        self.aotask.stop()
        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.stageScan.sampleRate)

        self.aotask.write(returnRamps, auto_start=False)
        self.aotask.start()
        self.waiter.start()

    def done(self):
        self.aotask.stop()
        self.aotask.close()
        self.dotask.stop()
        self.dotask.close()
        self.nidaq.reset_device()
        self.finalizeDone.emit()


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
        self.nidaq.reset_device()


class StageScan():
    '''Contains the analog signals in sig_dict. The update function takes the
    parameter_values and updates the signals accordingly.'''
    def __init__(self, sampleRate):
        self.scanMode = 'FOV scan'
        self.primScanDim = 'x'
        self.sigDict = {'x': [], 'y': [], 'z': []}
        self.sampleRate = sampleRate
        self.seqSamps = None
        self.FOVscan = FOVscan(self.sampleRate)
        self.VOLscan = VOLscan(self.sampleRate)
        self.lineScan = LineScan(self.sampleRate)
        self.scans = {'FOV scan': self.FOVscan,
                      'VOL scan': self.VOLscan,
                      'Line scan': self.lineScan}
        self.frames = 0

    def setScanMode(self, mode):
        self.scanMode = mode

    def setPrimScanDim(self, dim):
        self.primScanDim = dim

    def updateFrames(self, parValues):
        self.scans[self.scanMode].updateFrames(parValues)
        self.frames = self.scans[self.scanMode].frames

    def update(self, parValues):
        self.scans[self.scanMode].update(parValues, self.primScanDim)
        self.sigDict = self.scans[self.scanMode].sigDict
        self.seqSamps = self.scans[self.scanMode].seqSamps
        self.frames = self.scans[self.scanMode].frames


class LineScan():

    def __init__(self, sampleRate):
        self.sigDict = {'x': [], 'y': [], 'z': []}
        self.sampleRate = sampleRate
        self.corrStepSize = None
        self.seqSamps = None
        self.frames = 0

    def updateFrames(self, parValues):
        sizeY = parValues['sizeY'] / convFactors['y']
        stepSize = parValues['stepSizeXY'] / convFactors['y']
        stepsY = int(np.ceil(sizeY / stepSize))
        # +1 because nr of frames per line is one more than nr of steps
        self.frames = stepsY + 1

    def update(self, parValues, primScanDim):
        '''Create signals.
        First, distances are converted to voltages.'''
        startY = 0
        sizeY = parValues['sizeY'] / convFactors['y']
        seqSamps = np.round(self.sampleRate * parValues['seqTime'])
        stepSize = parValues['stepSizeXY'] / convFactors['y']
        self.stepsY = int(np.ceil(sizeY / stepSize))
        # Step size compatible with width
        self.corrStepSize = sizeY / self.stepsY
        self.seqSamps = int(seqSamps)

        ramp = makeRamp(startY, sizeY, self.stepsY * self.seqSamps)
        self.sigDict[primScanDim] = 1.14 * ramp
        for key in self.sigDict:
            if not key[0] == primScanDim:
                self.sigDict[key] = np.zeros(len(ramp))


class FOVscan():

    def __init__(self, sampleRate):
        self.sigDict = {'x': [], 'y': [], 'z': []}
        self.sampleRate = sampleRate
        self.corrStepSize = None
        self.seqSamps = None
        self.frames = 0

    def updateFrames(self, parValues):
        '''Update signals according to parameters.
        Note that rounding floats to ints may cause actual scan to differ
        slightly from expected scan. Maybe either limit input parameters to
        numbers that "fit each other" or find other solution, eg step size has
        to be width divided by an integer. Maybe not a problem ???'''
        stepSizeX = parValues['stepSizeXY'] / convFactors['x']
        stepSizeY = parValues['stepSizeXY'] / convFactors['y']
        sizeX = parValues['sizeX'] / convFactors['x']
        sizeY = parValues['sizeY'] / convFactors['y']
        stepsX = int(np.ceil(sizeX / stepSizeX))
        stepsY = int(np.ceil(sizeY / stepSizeY))
        # +1 because nr of frames per line is one more than nr of steps
        self.frames = (stepsY + 1) * (stepsX + 1)

    def update(self, parValues, primScanDim):
        '''Create signals.
        First, distances are converted to voltages.'''
        # Create signals
        startX = 0
        startY = 0
        sizeX = parValues['sizeX'] / convFactors['x']
        sizeY = parValues['sizeY'] / 2
        stepSizeX = parValues['stepSizeXY'] / convFactors['x']
        stepSizeY = parValues['stepSizeXY'] / convFactors['y']
        self.seqSamps = int(np.round(self.sampleRate*parValues['seqTime']))
        self.stepsX = int(np.ceil(sizeX / stepSizeX))
        self.stepsY = int(np.ceil(sizeY / stepSizeY))
        # Step size compatible with width
        self.corrStepSize = sizeX / self.stepsX
        rowSamps = self.stepsX * self.seqSamps

        LTRramp = makeRamp(startX, sizeX, rowSamps)
        RTLramp = LTRramp[::-1]

        primDimSig = []
        secDimSig = []
        newValue = startY
        for i in range(0, self.stepsY):
            if i % 2 == 0:
                primDimSig = np.concatenate((primDimSig, LTRramp))
            else:
                primDimSig = np.concatenate((primDimSig, RTLramp))
            secDimSig = np.concatenate((secDimSig, newValue*np.ones(rowSamps)))
            newValue = newValue + self.corrStepSize

        # Assign primary scan dir
        # 1.14 is emperically measured correction factor
        self.sigDict[primScanDim] = 1.14 * primDimSig
        # Assign second and third dim
        for key in self.sigDict:
            if not key[0] == primScanDim and not key[0] == 'z':
                self.sigDict[key] = 1.14 * secDimSig
            elif not key[0] == primScanDim:
                self.sigDict[key] = np.zeros(len(secDimSig))


class VOLscan():
    """Class representing the scanning movement for a volumetric scan i.e.
    multiple conscutive XY-planes with a certain delta z distance."""

    def __init__(self, sampleRate):
        self.sigDict = {'x': [], 'y': [], 'z': []}
        self.sampleRate = sampleRate
        self.corrStepSizeXY = None
        self.corrStepSizeZ = None
        self.seqSamps = None
        self.frames = 0

    def updateFrames(self, parValues):
        pass

    def update(self, parValues, primScanDim):
        print('Updating VOL scan')
        # Create signals
        startX = 0
        startY = 0
        startZ = 0
        # Division by 2 to convert from distance to voltage
        sizeX = parValues['sizeX'] / convFactors['x']
        sizeY = parValues['sizeY'] / convFactors['y']
        sizeZ = parValues['sizeZ'] / convFactors['z']
        stepSizeX = parValues['stepSizeXY'] / convFactors['x']
        stepSizeY = parValues['stepSizeXY'] / convFactors['y']
        stepSizeZ = parValues['stepSizeZ'] / convFactors['z']
        # WARNING: Correct for units of the time, now seconds!!!!
        self.seqSamps = int(np.round(self.sampleRate * parValues['seqTime']))
        self.stepsX = int(np.ceil(sizeX / stepSizeX))
        self.stepsY = int(np.ceil(sizeY / stepSizeY))
        self.stepsZ = int(np.ceil(sizeZ / stepSizeZ))

        # Step size compatible with width
        self.corrStepSizeXY = sizeX / self.stepsX
        # Step size compatible with range
        self.corrStepSizeZ = sizeZ / self.stepsZ
        rowSamps = self.stepsX * self.seqSamps

        # rampAndK contains [ramp, k]
        LTRramp = makeRamp(startX, sizeX, rowSamps)
        # RTLramp contains only ramp, no k since same k = -k
        RTLramp = LTRramp[::-1]

        primDimSig = []
        secDimSig = []
        newValue = startY
        for i in range(0, self.stepsY):
            if i % 2 == 0:
                primDimSig = np.concatenate((primDimSig, LTRramp))
            else:
                primDimSig = np.concatenate((primDimSig, RTLramp))
            secDimSig = np.concatenate((secDimSig, newValue*np.ones(rowSamps)))
            newValue = newValue + self.corrStepSizeXY

        sampsPerSlice = len(primDimSig)  # Used in Scanner->runScan
        self.cyclesPerSlice = sampsPerSlice / self.seqSamps
        """Below we make the concatenation along the third dimension, between
        the "slices" we add a smooth transition to avoid too rapid motion that
        seems to cause strange movent. This needs to be syncronized with the
        pixel cycle signal"""
        fullZprimDimSig = primDimSig
        fullZsecDimSig = secDimSig
        fullZthirdDimSig = startZ * np.ones(len(primDimSig))
        newValue = startZ + 1
        primDimTransition = makeSmoothStep(
            primDimSig[-1], primDimSig[0], self.seqSamps)
        secDimTransition = makeSmoothStep(
            secDimSig[-1], secDimSig[0], self.seqSamps)
        thirdDimTransition = makeSmoothStep(
            0, self.corrStepSizeZ, self.seqSamps)

        for i in range(1, self.stepsZ - 1):
            fullZprimDimSig = np.concatenate(
                (fullZprimDimSig, primDimTransition))
            fullZsecDimSig = np.concatenate(
                (fullZsecDimSig, secDimTransition))
            fullZthirdDimSig = np.concatenate(
                (fullZthirdDimSig, newValue + thirdDimTransition))

            fullZprimDimSig = np.concatenate((fullZprimDimSig, primDimSig))
            fullZsecDimSig = np.concatenate((fullZsecDimSig, secDimSig))
            fullZthirdDimSig = np.concatenate(
                (fullZthirdDimSig, newValue * np.ones(len(primDimSig))))
            newValue = newValue + self.corrStepSizeZ

        fullZprimDimSig = np.concatenate((fullZprimDimSig, primDimSig))
        fullZsecDimSig = np.concatenate((fullZsecDimSig, secDimSig))
        fullZthirdDimSig = np.concatenate((fullZthirdDimSig,
                                           newValue*np.ones(len(primDimSig))))
        # Assign primary scan dir
        # 1.14 is an emperically measured correction factor
        self.sigDict[primScanDim] = 1.14 * fullZprimDimSig
        # Assign second and third dim
        for key in self.sigDict:
            if not key[0] == primScanDim and not key[0] == 'z':
                self.sigDict[key] = 1.14 * fullZsecDimSig
            elif not key[0] == primScanDim:
                self.sigDict[key] = 1.14 * fullZthirdDimSig

        print('Final X: ' + str(fullZprimDimSig[-1]))
        print('Final Y: ' + str(fullZsecDimSig[-1]))
        print('Final Z: ' + str(fullZthirdDimSig[-1]))


class PixelCycle():
    ''' Contains the digital signals for the pixel cycle. The update function
    takes a parameter_values dict and updates the signal accordingly.'''
    def __init__(self, sampleRate):
        self.sigDict = {'405': [], '473': [], '488': [], 'CAM': []}
        self.sampleRate = sampleRate

    def update(self, devices, parValues, cycleSamps):
        for device in devices:
            signal = np.zeros(cycleSamps)
            start_name = 'start' + device
            end_name = 'end' + device
            start_pos = parValues[start_name] * self.sampleRate
            start_pos = int(min(start_pos, cycleSamps - 1))
            end_pos = parValues[end_name] * self.sampleRate
            end_pos = int(min(end_pos, cycleSamps))
            signal[range(start_pos, end_pos)] = 1
            self.sigDict[device] = signal


class GraphFrame(pg.GraphicsWindow):
    """Creates the plot that plots the preview of the pulses.
    Fcn update() updates the plot of "device" with signal "signal"."""
    def __init__(self, pxCycle, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pxCycle = pxCycle
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setYRange(0, 1)
        self.plot.showGrid(x=False, y=False)
        self.plotSigDict = {'405': self.plot.plot(pen=pg.mkPen(130, 0, 200)),
                            '473': self.plot.plot(pen=pg.mkPen(0, 183, 255)),
                            '488': self.plot.plot(pen=pg.mkPen(0, 247, 255)),
                            'CAM': self.plot.plot(pen='w')}
        self.resize(600, 200)

    def update(self, devices=None):
        if devices is None:
            devices = self.plotSigDict

        for device in devices:
            signal = self.pxCycle.sigDict[device]
            self.plotSigDict[device].setData(signal)


def makeRamp(start, end, samples):
    return np.linspace(start, end, num=samples)


def makeSmoothStep(start, end, samples):
    x = np.linspace(start, end, num=samples, endpoint=True)
    x = 0.5*(1 - np.cos(x * np.pi))
    signal = start + (end - start) * x
    return signal


def distToVoltY(d):
    a1 = 0.6524
    a2 = -0.0175
    a3 = 0.0004
    samples = len(d)
    Vsignal = np.zeros(samples)
    now = time.time()
    for i in range(0, samples):
        V_value = a1 * d[i] + a2 * d[i]**2 + a3 * np.power(d[i], 3)
        Vsignal[i] = V_value

    elapsed = time.time() - now
    print('Elapsed time: ', elapsed)
    return Vsignal


def distToVoltX(d):
    a1 = 0.6149
    a2 = -0.0146
    a3 = 0.0003
    samples = len(d)
    Vsignal = np.zeros(samples)
    now = time.time()
    for i in range(0, samples):
        V_value = a1 * d[i] + a2 * d[i]**2 + a3 * np.power(d[i], 3)
        Vsignal[i] = V_value

    elapsed = time.time() - now
    print('Elapsed time: ', elapsed)
    return Vsignal
