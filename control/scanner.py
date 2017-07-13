# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
"""

import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import copy
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

import control.guitools as guitools


class ScanWidget(QtGui.QFrame):
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

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.all_devices = ['405', '473', '488', 'CAM']
        self.saved_signal = np.array([0, 0])
        self.times_run = 0
        self.saveScanBtn = QtGui.QPushButton('Save Scan')
        self.saveScanBtn.clicked.connect(lambda: guitools.saveScan(self))
        self.loadScanBtn = QtGui.QPushButton('Load Scan')
        self.loadScanBtn.clicked.connect(lambda: guitools.loadScan(self))

        self.sampleRateEdit = QtGui.QLineEdit()

        self.widthPar = QtGui.QLineEdit('10')
        self.widthPar.editingFinished.connect(
            lambda: self.ScanParameterChanged('width'))
        self.heightPar = QtGui.QLineEdit('10')
        self.heightPar.editingFinished.connect(
            lambda: self.ScanParameterChanged('height'))
        self.seqTimePar = QtGui.QLineEdit('100')  # Milliseconds
        self.seqTimePar.editingFinished.connect(
            lambda: self.ScanParameterChanged('sequence_time'))
        self.nrFramesPar = QtGui.QLabel()
        self.scanDuration = QtGui.QLabel()
        self.step_sizePar = QtGui.QLineEdit('0.5')
        self.step_sizePar.editingFinished.connect(
            lambda: self.ScanParameterChanged('step_size'))
        self.sampleRate = 100000
#        self.sampleRate = np.float(self.sampleRateEdit.text())

        self.Scan_Mode = QtGui.QComboBox()
        self.scan_modes = ['FOV scan', 'Line scan']
        self.Scan_Mode.addItems(self.scan_modes)
        self.Scan_Mode.currentIndexChanged.connect(
            lambda: self.setScanMode(self.Scan_Mode.currentText()))

        self.prim_scan_dim = QtGui.QComboBox()
        self.scan_dims = ['x', 'y']
        self.prim_scan_dim.addItems(self.scan_dims)
        self.prim_scan_dim.currentIndexChanged.connect(
            lambda: self.setPrimScanDim(self.prim_scan_dim.currentText()))

        self.scan_parameters = {'width': self.widthPar,
                                'height': self.heightPar,
                                'sequence_time': self.seqTimePar,
                                'step_size': self.step_sizePar}

        self.scanParValues = {
            'width': float(self.widthPar.text()),
            'height': float(self.heightPar.text()),
            'sequence_time': float(self.seqTimePar.text()) / 1000,
            'step_size': float(self.step_sizePar.text())}

        self.start473Par = QtGui.QLineEdit('0')
        self.start473Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('start473'))
        self.start405Par = QtGui.QLineEdit('0')
        self.start405Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('start405'))
        self.start488Par = QtGui.QLineEdit('0')
        self.start488Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('start488'))
        self.startCAMPar = QtGui.QLineEdit('0')
        self.startCAMPar.editingFinished.connect(
            lambda: self.PixelParameterChanged('startCAM'))

        self.end473Par = QtGui.QLineEdit('0')
        self.end473Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('end473'))
        self.end405Par = QtGui.QLineEdit('0')
        self.end405Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('end405'))
        self.end488Par = QtGui.QLineEdit('0')
        self.end488Par.editingFinished.connect(
            lambda: self.PixelParameterChanged('end488'))
        self.endCAMPar = QtGui.QLineEdit('0')
        self.endCAMPar.editingFinished.connect(
            lambda: self.PixelParameterChanged('endCAM'))

        self.pxParameters = {'start405': self.start405Par,
                             'start473': self.start473Par,
                             'start488': self.start488Par,
                             'startCAM': self.startCAMPar,
                             'end405': self.end405Par,
                             'end473': self.end473Par,
                             'end488': self.end488Par,
                             'endCAM': self.endCAMPar}

        self.pxParValues = {
            'start405': float(self.start488Par.text()) / 1000,
            'start473': float(self.start405Par.text()) / 1000,
            'start488': float(self.start473Par.text()) / 1000,
            'startCAM': float(self.startCAMPar.text()) / 1000,
            'end405': float(self.end473Par.text()) / 1000,
            'end473': float(self.end405Par.text()) / 1000,
            'end488': float(self.end488Par.text()) / 1000,
            'endCAM': float(self.endCAMPar.text()) / 1000}

        self.currDOchan = {'405': 0, '473': 2, '488': 3, 'CAM': 4}
        self.currAOchan = {'x': 0, 'y': 1}

        self.stageScan = StageScan(self.sampleRate)
        self.pxCycle = PixelCycle(self.sampleRate)
        self.graph = GraphFrame(self.pxCycle)
        self.update_Scan(self.all_devices)

        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(
            lambda: self.setScanOrNot(False))

        self.ScanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.ScanButton.clicked.connect(self.ScanOrAbort)
        self.PreviewButton = QtGui.QPushButton('Preview')
        self.PreviewButton.clicked.connect(self.PreviewScan)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        grid.setRowMinimumHeight(5, 10)
        grid.addWidget(self.loadScanBtn, 0, 0)
        grid.addWidget(self.saveScanBtn, 0, 1)
        grid.addWidget(QtGui.QLabel('Width (um):'), 2, 0, 2, 1)
        grid.addWidget(self.widthPar, 2, 1, 2, 1)
        grid.addWidget(QtGui.QLabel('Height (um):'), 2, 2, 2, 1)
        grid.addWidget(self.heightPar, 2, 3, 2, 1)
        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 4, 0)
        grid.addWidget(self.seqTimePar, 4, 1)
        grid.addWidget(QtGui.QLabel('Frames in scan:'), 4, 2)
        grid.addWidget(self.nrFramesPar, 4, 3)
        grid.addWidget(QtGui.QLabel('Step size (um):'), 3, 4)
        grid.addWidget(self.step_sizePar, 3, 5)
        grid.addWidget(QtGui.QLabel('Scan duration (s):'), 4, 4)
        grid.addWidget(self.scanDuration, 4, 5)
        grid.addWidget(QtGui.QLabel('Start (ms):'), 6, 1)
        grid.addWidget(QtGui.QLabel('End (ms):'), 6, 2)
        grid.addWidget(QtGui.QLabel('405:'), 7, 0)
        grid.addWidget(self.start405Par, 7, 1)
        grid.addWidget(self.end405Par, 7, 2)
        grid.addWidget(self.scanRadio, 7, 4, 2, 1)
        grid.addWidget(QtGui.QLabel('Scan mode:'), 6, 5)
        grid.addWidget(self.Scan_Mode, 7, 5)
        grid.addWidget(QtGui.QLabel('Primary scan dim:'), 8, 5)
        grid.addWidget(self.prim_scan_dim, 9, 5)
        grid.addWidget(QtGui.QLabel('473:'), 8, 0)
        grid.addWidget(self.start473Par, 8, 1)
        grid.addWidget(self.end473Par, 8, 2)
        grid.addWidget(self.contLaserPulsesRadio, 8, 4, 2, 1)
        grid.addWidget(QtGui.QLabel('488:'), 9, 0)
        grid.addWidget(self.start488Par, 9, 1)
        grid.addWidget(self.end488Par, 9, 2)
        grid.addWidget(QtGui.QLabel('CAM:'), 10, 0)
        grid.addWidget(self.startCAMPar, 10, 1)
        grid.addWidget(self.endCAMPar, 10, 2)
        grid.addWidget(self.graph, 11, 0, 1, 6)
        grid.addWidget(self.ScanButton, 12, 3, 1, 3)
        grid.addWidget(self.PreviewButton, 12, 0, 1, 3)

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
        self.step_sizePar.setEnabled(value)
        self.Scan_Mode.setEnabled(value)
        self.prim_scan_dim.setEnabled(value)
        if value:
            self.ScanButton.setText('Scan')
        else:
            self.ScanButton.setText('Run')

    def setScanOrNot(self, value):
        self.scanOrNot = value

    def setScanMode(self, mode):
        self.stageScan.set_scan_mode(mode)
        self.ScanParameterChanged('scan_mode')

    def setPrimScanDim(self, dim):
        self.stageScan.set_prim_scan_dim(dim)
        self.ScanParameterChanged('prim_scan_dim')

    def ScanParameterChanged(self, par):
        if par not in ('scan_mode', 'prim_scan_dim'):
            if par == 'sequence_time':
                # To get in seconds
                self.scanParValues[par] = float(self.scan_pars[par].text()) / 1000
            else:
                self.scanParValues[par] = float(self.scan_pars[par].text())

        if par == 'sequence_time':
            self.update_Scan(self.all_devices)
            self.graph.update(self.all_devices)
        self.stageScan.update_frames(self.scanParValues)
        self.nrFramesPar.setText(str(self.stageScan.frames))
        self.scanDuration.setText(str((1 / 1000) * self.stageScan.frames *
                                  float(self.seqTimePar.text())))

    def PixelParameterChanged(self, par):
        self.pxParValues[par] = float(self.pxParameters[par].text()) / 1000
        dev = [par[-3] + par[-2] + par[-1]]
        self.pxCycle.update(dev, self.pxParValues, self.stageScan.seqSamples)
        self.graph.update(dev)

    def PreviewScan(self):
        self.stageScan.update(self.scanParValues)
        x_sig = self.stageScan.sig_dict['x_sig']
        y_sig = self.stageScan.sig_dict['y_sig']
        plt.plot(x_sig, y_sig)
        plt.axis([-0.2, self.scanParValues['width'] + 0.2, -
                  0.2, self.scanParValues['height'] + 0.2])

    def ScanOrAbort(self):
        if not self.scanning:
            self.PrepAndRun()
        else:
            self.scanner.abort()

    # PrepAndRun is only called if scanner is not running (See ScanOrAbort
    # funciton)
    def PrepAndRun(self):

        if self.scanRadio.isChecked():
            self.stageScan.update(self.scanParValues)
            self.ScanButton.setText('Abort')
            self.scanner = Scanner(self.nidaq,
                                   self.stageScan,
                                   self.pxCycle,
                                   self.currAOchan,
                                   self.currDOchan,
                                   self)
            self.scanner.finalizeDone.connect(self.FinalizeDone)
            self.scanner.scanDone.connect(self.ScanDone)
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
        self.stageScan.update(self.scanParValues)
        self.pxCycle.update(devices,
                            self.pxParValues,
                            self.stageScan.seqSamples)

    def closeEvent(self, *args, **kwargs):
        try:
            self.scanner.waiter.terminate()
        except BaseException:
            pass


# This can probably replaced by nidaqmx.task.register_done_event
class Wait_Thread(QtCore.QThread):
    waitdoneSignal = QtCore.pyqtSignal()

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.wait = True

    def run(self):
        print('will wait for ' + self.task.name)
        if self.wait:
            self.task.wait_until_done()
        self.wait = True
        self.waitdoneSignal.emit()
        print(self.task.name + ' is done')

    def stop(self):
        self.wait = False


class Scanner(QtCore.QObject):

    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()

    def __init__(self, device, stageScan, pxCycle, currAOchan,
                 currDOchan, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pxCycle = pxCycle
        # Dict containing channel numbers to be written to for each signal
        self.currAOchan = currAOchan
        # Dict containing channel numbers to be written to for each device
        self.currDOchan = currDOchan
        self.main = main

        self.aotask = nidaqmx.Task('aotask')
        self.dotask = nidaqmx.Task('dotask')

        self.waiter = Wait_Thread(self.aotask)

        self.warningTime = 10
        self.scanTimeWar = QtGui.QMessageBox()
        self.scanTimeWar.setInformativeText(
            "Are you sure you want to continue?")
        self.scanTimeWar.setStandardButtons(QtGui.QMessageBox.Yes |
                                            QtGui.QMessageBox.No)

    def runScan(self):
        scan_time = self.stageScan.seqSamples / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scanTimeWar.setText("Scan will take %s seconds" % scan_time)
        if scan_time > self.warningTime:
            ret = self.scanTimeWar.exec_()

        if ret == QtGui.QMessageBox.No:
            self.done()
            return

        fullAOsignal = np.zeros((2, len(self.stageScan.sig_dict['x_sig'])))
        tempAOchan = copy.copy(self.currAOchan)
        min_ao = -10
        max_ao = 10

        # Following loop creates the voltage channels in smallest to largest
        # order and places signals in same order.
        for i in range(0, 2):
            # dim = dimension ('x' or 'y') containing smallest channel nr.
            dim = min(tempAOchan, key=tempAOchan.get)
            chanstring = 'Dev1/ao%s' % tempAOchan[dim]
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel=chanstring,
                name_to_assign_to_channel='chan%s' % dim,
                min_val=min_ao, max_val=max_ao)
            tempAOchan.pop(dim)
            fullAOsignal[i] = self.stageScan.sig_dict[dim + '_sig']

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.stageScan.seqSamples)

        # Same as above but for the digital signals/devices
        tmpDOchan = copy.copy(self.currDOchan)
        fullDOsignal = np.zeros((len(tmpDOchan), self.pxCycle.cycleSamples),
                                dtype=bool)
        for i in range(0, len(tmpDOchan)):
            dev = min(tmpDOchan, key=tmpDOchan.get)
            chanstring = 'Dev1/port0/line%s' % tmpDOchan[dev]
            self.dotask.do_channels.add_do_chan(
                lines=chanstring,
                name_to_assign_to_lines='chan%s' % dev)
            tmpDOchan.pop(dev)
            fullDOsignal[i] = self.pxCycle.sig_dict[dev + 'sig']

        self.dotask.timing.cfg_samp_clk_timing(
            rate=self.pxCycle.sampleRate,
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.stageScan.seqSamples)

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
        self.dotask.stop()
        self.finalize()

    def finalize(self):
        print('in finalize')
        self.scanDone.emit()
        # Apparently important, otherwise finalize is called again when next
        # waiting finishes.
        self.waiter.waitdoneSignal.disconnect(self.finalize)
        self.waiter.waitdoneSignal.connect(self.done)
        # TODO: Test abort
        writtenSamps = np.round(self.aotask.out_stream.curr_write_pos)
        final_x = self.stageScan.sig_dict['x_sig'][int(writtenSamps) - 1]
        final_y = self.stageScan.sig_dict['y_sig'][int(writtenSamps) - 1]
        finalSamps = [final_x, final_y]

        # Following code should correct channels mentioned in Buglist.
        finalSamps = [0, 0]
        finalSamps[self.currAOchan['x']] = final_x
        finalSamps[self.currAOchan['y']] = final_y

        returnRamps = np.zeros((2, self.stageScan.sampleRate))
        returnRamps[0] = makeRamp(
            finalSamps[0], 0, self.stageScan.sampleRate)[0]
        returnRamps[1] = makeRamp(
            finalSamps[1], 0, self.stageScan.sampleRate)[0]

        # Seems to decrease frequency of Invalid task errors.
        # magic = np.ones(100)
        self.aotask.stop()
        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.stageScan.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.stageScan.sampleRate)

        self.aotask.write(returnRamps, auto_start=False)
        self.aotask.start()
        self.waiter.start()

    def done(self):
        print('in self.done()')
        self.aotask.clear()
        self.dotask.clear()
        self.nidaq.reset_device()
        self.finalizeDone.emit()


class LaserCycle():

    def __init__(self, device, pxCycle, curren_dochannels):

        self.nidaq = device
        self.pxCycle = pxCycle
        self.currDOchan = curren_dochannels

    def run(self):
        self.dotask = nidaqmx.Task('dotask')

        tmpDOchan = copy.copy(self.currDOchan)
        fullDOsignal = np.zeros((len(tmpDOchan), self.pxCycle.cycleSamples),
                                dtype=bool)
        for i in range(0, len(tmpDOchan)):
            dev = min(tmpDOchan, key=tmpDOchan.get)
            chanstring = 'Dev1/port0/line%s' % tmpDOchan[dev]
            self.dotask.do_channels.add_do_chan(
                lines=chanstring,
                name_to_assign_to_lines='chan%s' % dev)
            tmpDOchan.pop(dev)
            fullDOsignal[i] = self.pxCycle.sig_dict[dev + 'sig']

        self.dotask.timing.cfg_samp_clk_timing(source=r'100kHzTimeBase',
                                               rate=self.pxCycle.sampleRate,
                                               sample_mode='continuous')

        self.dotask.write(fullDOsignal, auto_start=False)

        self.dotask.start()

    def stop(self):
        self.dotask.stop()
        self.dotask.clear()
        del self.dotask
        self.nidaq.reset()


class StageScan():
    '''Contains the analog signals in sig_dict. The update function takes the
    parameter_values and updates the signals accordingly.'''
    def __init__(self, sampleRate):
        self.scan_mode = 'FOV scan'
        self.prim_scan_dim = 'x'
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sampleRate = sampleRate
        self.seqSamples = None
        self.FOV_scan = FOV_Scan(self.sampleRate)
        self.line_scan = Line_scan(self.sampleRate)
        self.scans = {'FOV scan': self.FOV_scan, 'Line scan': self.line_scan}
        self.frames = 0

    def set_scan_mode(self, mode):
        self.scan_mode = mode

    def set_prim_scan_dim(self, dim):
        self.prim_scan_dim = dim

    def update_frames(self, par_values):
        self.scans[self.scan_mode].update_frames(par_values)
        self.frames = self.scans[self.scan_mode].frames

    def update(self, par_values):
        self.scans[self.scan_mode].update(par_values, self.prim_scan_dim)
        self.sig_dict = self.scans[self.scan_mode].sig_dict
        self.seqSamples = self.scans[self.scan_mode].seqSamples
        self.frames = self.scans[self.scan_mode].frames


class Line_scan():

    def __init__(self, sampleRate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sampleRate = sampleRate
        self.corr_step_size = None
        self.seqSamples = None
        self.frames = 0

    def update_frames(self, par_values):
        size_y = par_values['height'] / 2
        step_size = par_values['step_size'] / 2
        steps_y = int(np.ceil(size_y / step_size))
        # +1 because nr of frames per line is one more than nr of steps
        self.frames = steps_y + 1

    def update(self, par_values, prim_scan_dim):

        # Create signals
        start_y = 0
#        start_x = 0
        size_y = par_values['height'] / 2
        seqSamples = np.round(self.sampleRate * par_values['sequence_time'])
        step_size = par_values['step_size'] / 2
        self.steps_y = int(np.ceil(size_y / step_size))
        # Step size compatible with width
        self.corr_step_size = size_y / self.steps_y
        self.seqSamples = int(seqSamples)
        colSamples = self.steps_y * self.seqSamples
        # rampAndK contains [ramp, k]
        rampAndK = makeRamp(start_y, size_y, colSamples)
        ramp = rampAndK[0]

        self.sig_dict[prim_scan_dim + '_sig'] = 1.14 * ramp
        self.sig_dict[chr(121 - (ord(prim_scan_dim) % 2)) +
                      '_sig'] = np.zeros(len(ramp))


class FOV_Scan():

    def __init__(self, sampleRate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sampleRate = sampleRate
        self.corr_step_size = None
        self.seqSamples = None
        self.frames = 0

    # Update signals according to parameters.
    # Note that rounding floats to ints may cause actual scan to differ slighly
    # from expected scan. Maybe either limit input parameters to numbers that
    # "fit each other" or find other solution, eg step size has to be width
    # divided by an integer. Maybe not a problem ???
    def update_frames(self, par_values):
        step_size = par_values['step_size'] / 2
        size_x = par_values['width'] / 2
        size_y = par_values['height'] / 2
        steps_x = int(np.ceil(size_x / step_size))
        steps_y = int(np.ceil(size_y / step_size))
        # +1 because nr of frames per line is one more than nr of steps
        self.frames = (steps_y + 1) * (steps_x + 1)

    def update(self, par_values, prim_scan_dim):

        # Create signals
        start_x = 0
        start_y = 0
        size_x = par_values['width'] / 2
        size_y = par_values['height'] / 2
        step_size = par_values['step_size'] / 2
        # WARNING: Correct for units of the time, now seconds!!!!
        self.seqSamples = int(np.round(
            self.sampleRate * par_values['sequence_time']))
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        # Step size compatible with width
        self.corr_step_size = size_x / self.steps_x
        row_samples = self.steps_x * self.seqSamples

        # rampAndK contains [ramp, k]
        rampAndK = makeRamp(start_x, size_x, row_samples)
        k = rampAndK[1]
        ltr_ramp = rampAndK[0]
        # rtl_ramp contains only ramp, no k since same k = -k
        rtl_ramp = ltr_ramp[::-1]
        gradual_k = makeRamp(k, -k, self.seqSamples)
        turn_rtl = np.cumsum(gradual_k[0])
        turn_ltr = -turn_rtl
        max_turn = np.max(turn_rtl)
        adjustor = 1 - self.seqSamples % 2

        # Create first and last part by flipping and turnign the turn_rtl array
        first_part = max_turn - \
            turn_rtl[range(int(np.ceil(self.seqSamples / 2)),
                           self.seqSamples)]
        mx = int(np.floor(self.seqSamples / 2) - adjustor + 1)
        last_part = max_turn + turn_rtl[range(0, mx)]
        y_ramp_smooth = np.append(first_part, last_part)
        # adjust scale and offset of ramp
        y_ramp_smooth = (self.corr_step_size / (2 * max_turn)) * y_ramp_smooth

        turn_rtl = ltr_ramp[-1] + turn_rtl

        prim_dim_sig = []
        sec_dim_sig = []
        new_value = start_y
        for i in range(0, self.steps_y):
            if i % 2 == 0:
                prim_dim_sig = np.concatenate(
                    (prim_dim_sig, ltr_ramp, turn_rtl))
            else:
                prim_dim_sig = np.concatenate(
                    (prim_dim_sig, rtl_ramp, turn_ltr))
            sec_dim_sig = np.concatenate(
                (sec_dim_sig, new_value * np.ones(row_samples),
                 new_value + y_ramp_smooth))
            new_value = new_value + self.corr_step_size

        i = i + 1
        if i % 2 == 0:
            prim_dim_sig = np.concatenate((prim_dim_sig, ltr_ramp))
        else:
            prim_dim_sig = np.concatenate((prim_dim_sig, rtl_ramp))
        sec_dim_sig = np.concatenate(
            (sec_dim_sig, new_value * np.ones(row_samples)))

        # Assign x_sig
        self.sig_dict[prim_scan_dim + '_sig'] = 1.14 * prim_dim_sig
        # Assign y_sig
        self.sig_dict[chr(121 - (ord(prim_scan_dim) % 2)) +
                      '_sig'] = 1.14 * sec_dim_sig


class PixelCycle():
    ''' Contains the digital signals for the pixel cycle. The update function
    takes a parameter_values dict and updates the signal accordingly.'''

    def __init__(self, sampleRate):
        self.sig_dict = {'405sig': [], '473sig': [], '488sig': [],
                         'CAMsig': []}
        self.sampleRate = sampleRate
        self.cycleSamples = 0

    def update(self, devices, par_values, cycleSamples):
        self.cycleSamples = cycleSamples
        for device in devices:
            signal = np.zeros(self.cycleSamples)
            start_name = 'start' + device
            end_name = 'end' + device
            start_pos = par_values[start_name] * self.sampleRate
            start_pos = int(min(start_pos, self.cycleSamples - 1))
            end_pos = par_values[end_name] * self.sampleRate
            end_pos = int(min(end_pos, self.cycleSamples))
            signal[range(start_pos, end_pos)] = 1
            self.sig_dict[device + 'sig'] = signal


class GraphFrame(pg.GraphicsWindow):
    """Child of pg.GraphicsWindow and creats the plot that plots the preview of
    the pulses.
    Fcn update() updates the plot of "device" with signal "signal"  """

    def __init__(self, pxCycle, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pxCycle = pxCycle
        self.plot = self.addPlot(row=1, col=0)
        self.plot.showGrid(x=False, y=False)
        self.plot_sig_dict = {'405': self.plot.plot(pen=pg.mkPen(73, 0, 188)),
                              '473': self.plot.plot(pen=pg.mkPen(0, 247, 255)),
                              '488': self.plot.plot(pen=pg.mkPen(97, 0, 97)),
                              'CAM': self.plot.plot(pen='w')}

    def update(self, devices=None):

        if devices is None:
            devices = self.plot_sig_dict

        for device in devices:
            signal = self.pxCycle.sig_dict[device + 'sig']
            self.plot_sig_dict[device].setData(signal)


def makeRamp(start, end, samples):
    k = (end - start) / (samples - 1)
    ramp = [start + k * i for i in range(0, samples)]
    return np.array([np.asarray(ramp), k])
