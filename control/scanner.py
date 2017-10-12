# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
"""

import numpy as np
import time
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt
import collections
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
    def __init__(self, device, main, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DigModW = QtGui.QMessageBox()
        self.DigModW.setInformativeText(
            "You need to be in digital modulation to scan")

        self.nidaq = device
        self.main = main
        self.allDevices = ['405', '473', '488', 'CAM']
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
        self.stepSizeXYPar = QtGui.QLineEdit('0.1')
        self.stepSizeXYPar.editingFinished.connect(
            lambda: self.scanParameterChanged('stepSizeXY'))
        self.stepSizeZPar = QtGui.QLineEdit('1')
        self.stepSizeZPar.editingFinished.connect(
            lambda: self.scanParameterChanged('stepSizeZ'))
        self.sampleRate = 100000

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
        self.graph.plot.getAxis('bottom').setScale(1000/self.sampleRate)
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
        self.continuousCheck = QtGui.QCheckBox('Continuous Scan')

        self.scanImage = ImageWidget()

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
        grid.addWidget(self.contLaserPulsesRadio, 5, 0)

        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 7, 0)
        grid.addWidget(self.seqTimePar, 7, 1)
        grid.addWidget(QtGui.QLabel('Start (ms):'), 6, 3)
        grid.addWidget(QtGui.QLabel('End (ms):'), 6, 4)
        grid.addWidget(QtGui.QLabel('405:'), 7, 2)
        grid.addWidget(self.start405Par, 7, 3)
        grid.addWidget(self.end405Par, 7, 4)
        grid.addWidget(QtGui.QLabel('473:'), 8, 2)
        grid.addWidget(self.start473Par, 8, 3)
        grid.addWidget(self.end473Par, 8, 4)
        grid.addWidget(QtGui.QLabel('488:'), 9, 2)
        grid.addWidget(self.start488Par, 9, 3)
        grid.addWidget(self.end488Par, 9, 4)
        grid.addWidget(QtGui.QLabel('Camera:'), 10, 2)
        grid.addWidget(self.startCAMPar, 10, 3)
        grid.addWidget(self.endCAMPar, 10, 4)

        grid.addWidget(self.graph, 11, 0, 1, 5)
        self.graph.setFixedHeight(100)
        grid.addWidget(self.scanImage, 12, 0, 1, 5)
        grid.addWidget(self.PreviewButton, 13, 0)
        grid.addWidget(self.ScanButton, 13, 1)
        grid.addWidget(self.continuousCheck, 13, 2)

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
        plt.show()

    def scanOrAbort(self):
        if not self.scanning:
            if self.main.laserWidgets.DigCtrl.DigitalControlButton.isChecked():
                self.prepAndRun()
            else:
                self.DigModW.exec_()
        else:
            self.scanner.abort()

    def prepAndRun(self):
        ''' Only called if scanner is not running (See scanOrAbort funciton).
        '''
        if self.scanRadio.isChecked():
            self.stageScan.update(self.scanParValues)
            self.ScanButton.setText('Abort')
            self.scanner = Scanner(self.nidaq, self.stageScan,
                                   self.pxCycle, self)
            self.scanner.finalizeDone.connect(self.finalizeDone)
            self.scanner.scanDone.connect(self.scanDone)
            self.scanning = True

            self.start_f = self.main.lvworkers[0].f_ind

            self.scanner.runScan()

        elif self.ScanButton.isChecked():
            self.lasercycle = LaserCycle(self.nidaq, self.pxCycle)
            self.ScanButton.setText('Stop')
            self.lasercycle.run()

        else:
            self.lasercycle.stop()
            self.ScanButton.setText('Run')
            del self.lasercycle

    def scanDone(self):
        print('in scanDone()')
        self.ScanButton.setEnabled(False)

        if not self.scanner.aborted:
            self.end_f = self.main.lvworkers[0].f_ind
            if self.end_f >= self.start_f - 1:
                f_range = range(self.start_f, self.end_f + 1)
            else:
                buffer_size = self.main.cameras[0].number_image_buffers
                f_range = np.append(range(self.start_f, buffer_size),
                                    range(0, self.end_f + 1))
            data = []
            for j in f_range:
                data.append(self.main.cameras[0].hcam_data[j].getData())
            datashape = (
                len(f_range), self.main.shapes[0][1], self.main.shapes[0][0])
            reshapeddata = np.reshape(data, datashape, order='C')
            z_stack = []
            for j in range(0, len(f_range)):
                z_stack.append(np.mean(reshapeddata[j, :, :]))

            if not np.floor(np.sqrt(len(z_stack))) == np.sqrt(len(z_stack)):
                del z_stack[0]
            imside = int(np.sqrt(np.size(z_stack)))
            print('Imside = ', imside)
            z_stack = np.reshape(z_stack, [imside, imside])
            z_stack[::2] = np.fliplr(z_stack[::2])
            self.scanImage.img.setImage(z_stack)

    def finalizeDone(self):
        if (not self.continuousCheck.isChecked()) or self.scanner.aborted:
            self.ScanButton.setText('Scan')
            self.ScanButton.setEnabled(True)
            del self.scanner
            self.scanning = False
        else:
            self.ScanButton.setEnabled(True)
            self.prepAndRun()

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

    def __init__(self, device, stageScan, pxCycle, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pxCycle = pxCycle

        self.sampsInScan = len(self.stageScan.sigDict['x'])
        self.main = main

        self.aotask = nidaqmx.Task('aotask')
        self.dotask = nidaqmx.Task('dotask')
        self.waiter = WaitThread(self.aotask)

        self.scanTimeW = QtGui.QMessageBox()
        self.scanTimeW.setInformativeText("Are you sure you want to continue?")
        self.scanTimeW.setStandardButtons(QtGui.QMessageBox.Yes |
                                          QtGui.QMessageBox.No)
        self.channel_order = ['x', 'y', 'z']

        self.aborted = False

    def runScan(self):
        self.aborted = False
        scanTime = self.sampsInScan / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scanTimeW.setText("Scan will take %s seconds" % scanTime)
        if scanTime > 10:
            ret = self.scanTimeW.exec_()

        if ret == QtGui.QMessageBox.No:
            self.done()
            return

        # Following loop creates the voltage channels
        AOchans = [0, 1, 2]
        for n in AOchans:
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel='Dev1/ao%s' % n,
                name_to_assign_to_channel='chan_%s' % self.channel_order[n],
                min_val=minVolt[self.channel_order[n]],
                max_val=maxVolt[self.channel_order[n]])

        fullAOsignal = np.array([self.stageScan.sigDict[self.channel_order[i]]
                                 for i in AOchans])

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            source=r'100kHzTimeBase',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan)

        # Same as above but for the digital signals/devices
        devs = list(self.pxCycle.sigDict.keys())
        DOchans = range(0, 4)
        for d in DOchans:
            chanstring = 'Dev1/port0/line%s' % d
            self.dotask.do_channels.add_do_chan(
                lines=chanstring, name_to_assign_to_lines='chan%s' % devs[d])

        fullDOsignal = np.array([self.pxCycle.sigDict[devs[i]]
                                for i in DOchans])

        """If doing VOLume scan, the time needed for the stage to move
        between z-planes needs to be filled with zeros/False. This time is
        equal to one "sequence-time". To do so we first have to repeat the
        sequence for the whole scan in one plane and then append with zeros."""
        if self.stageScan.scanMode == 'VOLscan':
            fullDOsignal = np.tile(fullDOsignal,
                                   self.stageScan.VOLscan.cyclesPerSlice)
            fullDOsignal = np.concatenate(
                (fullDOsignal, np.zeros(4, self.stageScan.seqSamps)))

        self.dotask.timing.cfg_samp_clk_timing(
            rate=self.pxCycle.sampleRate,
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan)

        self.waiter.waitdoneSignal.connect(self.finalize)

        self.aotask.write(fullAOsignal, auto_start=False)
        self.dotask.write(fullDOsignal, auto_start=False)

        self.dotask.start()
        self.aotask.start()

        self.waiter.start()

    def abort(self):
        print('Aborting scan')
        self.aborted = True
        self.waiter.stop()
        self.aotask.stop()
        self.dotask.stop()
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
        chans = [0, 1, 2]
        dim = [self.channel_order[i] for i in chans]
        finalSamps = [self.stageScan.sigDict[dim[i]][writtenSamps - 1]
                      for i in chans]
        print(finalSamps)
        returnRamps = np.array(
            [makeRamp(finalSamps[i], 0,
                      self.stageScan.sampleRate) for i in chans])

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

    def __init__(self, device, pxCycle):
        self.nidaq = device
        self.pxCycle = pxCycle

    def run(self):
        self.dotask = nidaqmx.Task('dotaskLaser')

        devs = list(self.pxCycle.sigDict.keys())
        DOchans = range(0, 4)
        for d in DOchans:
            chanstring = 'Dev1/port0/line%s' % d
            self.dotask.do_channels.add_do_chan(
                lines=chanstring, name_to_assign_to_lines='chan%s' % devs[d])

        DOchans = [0, 1, 2, 3]
        fullDOsignal = np.array([self.pxCycle.sigDict[devs[i]]
                                for i in DOchans])

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
        self.stepsX = 0
        self.stepsY = int(np.ceil(sizeY / stepSize))
        self.stepsZ = 0
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
        stepSizeX = parValues['stepSizeXY']
        stepSizeY = parValues['stepSizeXY']
        sizeX = parValues['sizeX']
        sizeY = parValues['sizeY']
        stepsX = int(np.ceil(sizeX / stepSizeX))
        stepsY = int(np.ceil(sizeY / stepSizeY))
        # +1 because nr of frames per line is one more than nr of steps
        self.frames = stepsY * stepsX

    def update(self, parValues, primScanDim):
        '''Create signals.
        Signals are first created in units of distance and converted to voltage
        at the end.'''
        # Create signals
        startX = 0
        startY = 0
        sizeX = parValues['sizeX']
        sizeY = parValues['sizeY']
        stepSizeX = parValues['stepSizeXY']
        stepSizeY = parValues['stepSizeXY']
        self.seqSamps = int(np.round(self.sampleRate*parValues['seqTime']))
        self.stepsX = int(np.ceil(sizeX / stepSizeX))
        self.stepsY = int(np.ceil(sizeY / stepSizeY))
        self.stepsZ = 0
        # Step size compatible with width
        self.corrStepSize = sizeX / self.stepsX
        rowSamps = self.stepsX * self.seqSamps

        LTRramp = makeRamp(startX, sizeX, rowSamps)
        RTLramp = LTRramp[::-1]

        Xsig = []
        Ysig = []
        newValue = startY
        for i in range(0, self.stepsY):
            if i % 2 == 0:
                Xsig = np.concatenate((Xsig, LTRramp))
            else:
                Xsig = np.concatenate((Xsig, RTLramp))
            Ysig = np.concatenate((Ysig, newValue*np.ones(rowSamps)))
            newValue = newValue + self.corrStepSize

        self.sigDict['x'] = Xsig / convFactors['x']
        self.sigDict['y'] = Ysig / convFactors['y']
        self.sigDict['z'] = np.zeros(len(Ysig))


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
        """Updates the VOL-scan signals, units of length are used when creating
        the scan signals and is converted to voltages at the end """
        print('Updating VOL scan')
        # Create signals
        startX = 0
        startY = 0
        startZ = 0
        # Division by 2 to convert from distance to voltage
        sizeX = parValues['sizeX']
        sizeY = parValues['sizeY']
        sizeZ = parValues['sizeZ']
        stepSizeX = parValues['stepSizeXY']
        stepSizeY = parValues['stepSizeXY']
        stepSizeZ = parValues['stepSizeZ']
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

        Xsig = []
        Ysig = []
        newValue = startY
        for i in range(0, self.stepsY):
            if i % 2 == 0:
                Xsig = np.concatenate((Xsig, LTRramp))
            else:
                Xsig = np.concatenate((Xsig, RTLramp))
            Ysig = np.concatenate((Ysig, newValue*np.ones(rowSamps)))
            newValue = newValue + self.corrStepSize

        sampsPerSlice = len(Xsig)  # Used in Scanner->runScan
        self.cyclesPerSlice = sampsPerSlice / self.seqSamps
        """Below we make the concatenation along the third dimension, between
        the "slices" we add a smooth transition to avoid too rapid motion that
        seems to cause strange movent. This needs to be syncronized with the
        pixel cycle signal"""
        fullXsig = Xsig
        fullYsig = Ysig
        fullZsig = startZ * np.ones(len(Xsig))
        newValue = startZ + 1
        XTransition = makeSmoothStep(
            Xsig[-1], Xsig[0], self.seqSamps)
        YTransition = makeSmoothStep(
            Ysig[-1], Ysig[0], self.seqSamps)
        ZTransition = makeSmoothStep(
            0, self.corrStepSizeZ, self.seqSamps)

        for i in range(1, self.stepsZ - 1):
            fullXsig = np.concatenate(
                (fullXsig, XTransition))
            fullYsig = np.concatenate(
                (fullYsig, YTransition))
            fullZsig = np.concatenate(
                (fullZsig, newValue + ZTransition))

            fullXsig = np.concatenate((fullXsig, Xsig))
            fullYsig = np.concatenate((fullYsig, Ysig))
            fullZsig = np.concatenate(
                (fullZsig, newValue * np.ones(len(Xsig))))
            newValue = newValue + self.corrStepSizeZ

        fullXsig = np.concatenate((fullXsig, Xsig))
        fullYsig = np.concatenate((fullYsig, Ysig))
        fullZsig = np.concatenate((fullZsig,
                                   newValue*np.ones(len(primDimSig))))
        # Assign signals to scanDict
        self.sigDict['x'] = fullXsig / convFactors['x']
        self.sigDict['y'] = fullXsig / convFactors['y']
        self.sigDict['z'] = fullXsig / convFactors['z']

        print('Final X: ' + str(fullZprimDimSig[-1]))
        print('Final Y: ' + str(fullZsecDimSig[-1]))
        print('Final Z: ' + str(fullZthirdDimSig[-1]))


class ImageWidget(pg.GraphicsLayoutWidget):

    def __init__(self):
        super().__init__()

        self.vb = self.addViewBox(row=1, col=1)
        self.img = pg.ImageItem()
        self.lut = guitools.cubehelix()
        self.img.setLookupTable(self.lut)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.addItem(self.hist, row=1, col=2)


class PixelCycle():
    ''' Contains the digital signals for the pixel cycle. The update function
    takes a parameter_values dict and updates the signal accordingly.'''
    def __init__(self, sampleRate):
        self.sigDict = collections.OrderedDict([('405', []),
                                                ('473', []),
                                                ('488', []),
                                                ('CAM', [])])
        self.sampleRate = sampleRate
        self.cycleSamps = None

    def update(self, devices, parValues, cycleSamps):
        self.cycleSamps = cycleSamps
        for device in devices:
            signal = np.zeros(cycleSamps, dtype='bool')
            start_name = 'start' + device
            end_name = 'end' + device
            start_pos = parValues[start_name] * self.sampleRate
            start_pos = int(min(start_pos, cycleSamps - 1))
            end_pos = parValues[end_name] * self.sampleRate
            end_pos = int(min(end_pos, cycleSamps))
            signal[range(start_pos, end_pos)] = True
            self.sigDict[device] = signal


class GraphFrame(pg.GraphicsWindow):
    """Creates the plot that plots the preview of the pulses.
    Fcn update() updates the plot of "device" with signal "signal"."""
    def __init__(self, pxCycle, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pxCycle = pxCycle
        self.plot = self.addPlot(row=1, col=0)
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
