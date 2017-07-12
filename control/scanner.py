# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@author: testaRES
"""

import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore

import control.guitools as guitools


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
        self.sequence_timePar = QtGui.QLineEdit('100')  # Milliseconds
        self.sequence_timePar.editingFinished.connect(
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
                                'sequence_time': self.sequence_timePar,
                                'step_size': self.step_sizePar}

        self.scan_par_values = {
            'width': float(self.widthPar.text()),
            'height': float(self.heightPar.text()),
            'sequence_time': float(self.sequence_timePar.text()) / 1000,
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

        self.pixel_parameters = {'start405': self.start405Par,
                                 'start473': self.start473Par,
                                 'start488': self.start488Par,
                                 'startCAM': self.startCAMPar,
                                 'end405': self.end405Par,
                                 'end473': self.end473Par,
                                 'end488': self.end488Par,
                                 'endCAM': self.endCAMPar}

        self.pixel_par_values = {
            'start405': float(self.start488Par.text()) / 1000,
            'start473': float(self.start405Par.text()) / 1000,
            'start488': float(self.start473Par.text()) / 1000,
            'startCAM': float(self.startCAMPar.text()) / 1000,
            'end405': float(self.end473Par.text()) / 1000,
            'end473': float(self.end405Par.text()) / 1000,
            'end488': float(self.end488Par.text()) / 1000,
            'endCAM': float(self.endCAMPar.text()) / 1000}

        self.current_dochannels = {'405': 0, '473': 2, '488': 3, 'CAM': 4}
        self.current_aochannels = {'x': 0, 'y': 1}

        self.stageScan = StageScan(self.sampleRate)
        self.pixel_cycle = PixelCycle(self.sampleRate)
        self.graph = GraphFrame(self.pixel_cycle)
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

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        grid.setRowMinimumHeight(5, 10)
        grid.addWidget(self.loadScanBtn, 0, 0)
        grid.addWidget(self.saveScanBtn, 0, 1)
        grid.addWidget(QtGui.QLabel('Width (um):'), 2, 0, 2, 1)
        grid.addWidget(self.widthPar, 2, 1, 2, 1)
        grid.addWidget(QtGui.QLabel('Height (um):'), 2, 2, 2, 1)
        grid.addWidget(self.heightPar, 2, 3, 2, 1)
        grid.addWidget(QtGui.QLabel('Sequence Time (ms):'), 4, 0)
        grid.addWidget(self.sequence_timePar, 4, 1)
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

    def ScanParameterChanged(self, parameter):
        if parameter not in ('scan_mode', 'prim_scan_dim'):
            if parameter == 'sequence_time':
                # To get in seconds
                self.scan_par_values[parameter] = float(
                    self.scan_parameters[parameter].text()) / 1000
            else:
                self.scan_par_values[parameter] = float(
                    self.scan_parameters[parameter].text())

        if parameter == 'sequence_time':
            self.update_Scan(self.all_devices)
            self.graph.update(self.all_devices)
        self.stageScan.update_frames(self.scan_par_values)
        self.nrFramesPar.setText(str(self.stageScan.frames))
        self.scanDuration.setText(str((1 / 1000) * self.stageScan.frames *
                                  float(self.sequence_timePar.text())))

    def PixelParameterChanged(self, parameter):
        self.pixel_par_values[parameter] = float(
            self.pixel_parameters[parameter].text()) / 1000
        device = parameter[-3] + parameter[-2] + parameter[-1]
        self.pixel_cycle.update(
            [device],
            self.pixel_par_values,
            self.stageScan.sequence_samples)
        self.graph.update([device])

    def PreviewScan(self):

        self.stageScan.update(self.scan_par_values)
        x_sig = self.stageScan.sig_dict['x_sig']
        y_sig = self.stageScan.sig_dict['y_sig']
        plt.plot(x_sig, y_sig)
        plt.axis([-0.2, self.scan_par_values['width'] + 0.2, -
                  0.2, self.scan_par_values['height'] + 0.2])

    def ScanOrAbort(self):
        if not self.scanning:
            self.PrepAndRun()
        else:
            self.scanner.abort()

    # PrepAndRun is only called if scanner is not running (See ScanOrAbort
    # funciton)
    def PrepAndRun(self):

        if self.scanRadio.isChecked():
            self.stageScan.update(self.scan_par_values)
            self.ScanButton.setText('Abort')
            self.scanner = Scanner(self.nidaq,
                                   self.stageScan,
                                   self.pixel_cycle,
                                   self.current_aochannels,
                                   self.current_dochannels,
                                   self)
            self.scanner.finalizeDone.connect(self.FinalizeDone)
            self.scanner.scanDone.connect(self.ScanDone)
            self.scanning = True
            self.scanner.runScan()

        elif self.ScanButton.isChecked():
            self.lasercycle = LaserCycle(self.nidaq,
                                         self.pixel_cycle,
                                         self.current_dochannels)
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
        self.stageScan.update(self.scan_par_values)
        self.pixel_cycle.update(devices,
                                self.pixel_par_values,
                                self.stageScan.sequence_samples)

    def closeEvent(self, *args, **kwargs):
        try:
            self.scanner.waiter.terminate()
        except BaseException:
            pass


class Wait_Thread(QtCore.QThread):
    waitdoneSignal = QtCore.pyqtSignal()

    def __init__(self, task):
        super().__init__()
        self.task = task
        self.wait = True

    def run(self):
        print('will wait for aotask')
        while not self.task.is_task_done() and self.wait:
            pass
        self.wait = True
        self.waitdoneSignal.emit()
        print(self.task.name + ' is done')

    def stop(self):
        self.wait = False


class Scanner(QtCore.QObject):

    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()

    def __init__(self, device, stageScan, pixel_cycle, current_aochannels,
                 current_dochannels, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pixel_cycle = pixel_cycle
        # Dict containing channel numbers to be written to for each signal
        self.current_aochannels = current_aochannels
        # Dict containing channel numbers to be written to for each device
        self.current_dochannels = current_dochannels
        self.samples_in_scan = len(self.stageScan.sig_dict['x_sig'])
        self.main = main

#        self.aotask = libnidaqmx.AnalogOutputTask('aotask')
        self.aotask = nidaqmx.Task('aotask')
#        self.dotask = libnidaqmx.DigitalOutputTask('dotask')
        self.dotask = nidaqmx.Task('dotask')

        self.waiter = Wait_Thread(self.aotask)

        self.warningTime = 10
        self.scanTimeWar = QtGui.QMessageBox()
        self.scanTimeWar.setInformativeText(
            "Are you sure you want to continue?")
        self.scanTimeWar.setStandardButtons(QtGui.QMessageBox.Yes |
                                            QtGui.QMessageBox.No)

    def finalize(self):
        print('in finalize')
        self.scanDone.emit()
        # Apparently important, otherwise finalize is called again when next
        # waiting finishes.
        self.waiter.waitdoneSignal.disconnect(self.finalize)
        self.waiter.waitdoneSignal.connect(self.done)
#        writtenSamps = self.aotask.get_samples_per_channel_generated()
        writtenSamps = self.aotask.out_stream.curr_write_pos  # TODO: Test
        final_x = self.stageScan.sig_dict['x_sig'][writtenSamps - 1]
        final_y = self.stageScan.sig_dict['y_sig'][writtenSamps - 1]
        finalSamps = [final_x, final_y]

        # Following code should correct channels mentioned in Buglist.
        finalSamps = [0, 0]
        finalSamps[self.current_aochannels['x']] = final_x
        finalSamps[self.current_aochannels['y']] = final_y

        return_ramps = np.array([])
        for i in range(0, 2):
            rampAndK = makeRamp(finalSamps[i], 0, self.stageScan.sampleRate)
            return_ramps = np.append(return_ramps, rampAndK[0])

        # Seems to decrease frequency of Invalid task errors.
        # magic = np.ones(100)
        print('aotask is: ', self.aotask)
        self.aotask.stop()
        self.aotask.configure_timing_sample_clock(
            rate=self.stageScan.sampleRate,
            sample_mode='finite',
            samples_per_channel=self.stageScan.sampleRate)

        self.aotask.write(return_ramps,
                          layout='group_by_channel',
                          auto_start=False)
        self.aotask.start()

        self.waiter.start()

    def done(self):
        print('in self.done()')
        self.aotask.close()
        self.dotask.close()
        self.nidaq.reset_device()
        self.finalizeDone.emit()

    def runScan(self):
        scan_time = self.samples_in_scan / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scanTimeWar.setText("Scan will take %s seconds" % scan_time)
        if scan_time > self.warningTime:
            ret = self.scanTimeWar.exec_()

        if ret == QtGui.QMessageBox.No:
            self.done()
            return

        full_ao_signal = []
        temp_aochannels = copy.copy(self.current_aochannels)
        min_ao = -10
        max_ao = 10

        # Following loop creates the voltage channels in smallest to largest
        # order and places signals in same order.
        for i in range(0, 2):
            # dim = dimension ('x' or 'y') containing smallest channel nr.
            dim = min(temp_aochannels, key=temp_aochannels.get)
            chanstring = 'Dev1/ao%s' % temp_aochannels[dim]
            self.aotask.create_voltage_channel(
                phys_channel=chanstring, channel_name='chan%s' %
                dim, min_val=min_ao, max_val=max_ao)
            temp_aochannels.pop(dim)
            signal = self.stageScan.sig_dict[dim + '_sig']
            if i == 1 and len(full_ao_signal) != len(signal):
                tx = 'Length of signals are not equal (printed from RunScan()'
                print(tx)
            full_ao_signal = np.append(full_ao_signal, signal)
#            finalSamps = np.append(finalSamps, signal[-1])

        # Same as above but for the digital signals/devices
        full_do_signal = []
        temp_dochannels = copy.copy(self.current_dochannels)
        for i in range(0, len(temp_dochannels)):
            dev = min(temp_dochannels, key=temp_dochannels.get)
            chanstring = 'Dev1/port0/line%s' % temp_dochannels[dev]
            self.dotask.create_channel(lines=chanstring, name='chan%s' % dev)
            temp_dochannels.pop(dev)
            signal = self.pixel_cycle.sig_dict[dev + 'sig']
            if len(full_ao_signal) % len(signal) != 0 and len(
                    full_do_signal) % len(signal) != 0:
                print('Signal lengths does not match (printed from run)')
            full_do_signal = np.append(full_do_signal, signal)

        self.aotask.configure_timing_sample_clock(
            rate=self.stageScan.sampleRate,
            sample_mode='finite',
            samples_per_channel=self.samples_in_scan)

        self.dotask.configure_timing_sample_clock(
            source=r'ao/SampleClock',
            rate=self.pixel_cycle.sampleRate,
            sample_mode='finite',
            samples_per_channel=self.samples_in_scan)

        # Following is to create ramps back to zero for the analog channels
        # during one second.

#        return_ramps = np.array([])
#        for i in range(0,2):
#            rampAndK = makeRamp(finalSamps[i], 0,
#                                   self.stageScan.sampleRate)
#            return_ramps = np.append(return_ramps, rampAndK[0])
#
#        print(np.ones(1)) # This line alone fixes the problem...

        self.waiter.waitdoneSignal.connect(self.finalize)

        self.dotask.write(full_do_signal,
                          layout='group_by_channel',
                          auto_start=False)
        self.aotask.write(full_ao_signal,
                          layout='group_by_channel',
                          auto_start=False)

        self.dotask.start()
        self.aotask.start()

        self.waiter.start()
        # Need to wait for task to finish, otherwise aotask will be deleted
#        self.aotask.wait_until_done()
#        self.aotask.stop()
#
#        self.aotask.configure_timing_sample_clock(rate=self.stageScan.sampleRate,
#                                                  sample_mode='finite',
#                                                  samples_per_channel=self.stageScan.sampleRate)
#
#
#
#        self.aotask.write(return_ramps, layout='group_by_channel',
#                          auto_start=False)
#        self.aotask.start()
#        self.aotask.wait_until_done()
#
#
#        self.aotask.clear()      # when function is finished and task aborted
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
        self.dotask = nidaqmx.Task('dotask')

        full_do_signal = []
        temp_dochannels = copy.copy(self.current_dochannels)
        for i in range(0, len(temp_dochannels)):
            dev = min(temp_dochannels, key=temp_dochannels.get)
            chanstring = 'Dev1/port0/line%s' % temp_dochannels[dev]
            self.dotask.create_channel(lines=chanstring, name='chan%s' % dev)
            temp_dochannels.pop(dev)
            signal = self.pixel_cycle.sig_dict[dev + 'sig']
            full_do_signal = np.append(full_do_signal, signal)

        self.dotask.configure_timing_sample_clock(
            source=r'100kHzTimeBase',
            rate=self.pixel_cycle.sampleRate,
            sample_mode='continuous')

        self.dotask.write(full_do_signal,
                          layout='group_by_channel',
                          auto_start=False)

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
        self.sequence_samples = None
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
        print('in update stageScan')
        print('self.scan_mode = ', self.scan_mode)
        self.scans[self.scan_mode].update(par_values, self.prim_scan_dim)
        self.sig_dict = self.scans[self.scan_mode].sig_dict
        self.sequence_samples = self.scans[self.scan_mode].sequence_samples
        self.frames = self.scans[self.scan_mode].frames


class Line_scan():

    def __init__(self, sampleRate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sampleRate = sampleRate
        self.corr_step_size = None
        self.sequence_samples = None
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
        sequence_samples = np.round(
            self.sampleRate *
            par_values['sequence_time'])
        step_size = par_values['step_size'] / 2
        self.steps_y = int(np.ceil(size_y / step_size))
        # Step size compatible with width
        self.corr_step_size = size_y / self.steps_y
        self.sequence_samples = int(sequence_samples)
        column_samples = self.steps_y * self.sequence_samples
        # rampAndK contains [ramp, k]
        rampAndK = makeRamp(start_y, size_y, column_samples)
        ramp = rampAndK[0]

        self.sig_dict[prim_scan_dim + '_sig'] = 1.14 * ramp
        self.sig_dict[chr(121 - (ord(prim_scan_dim) % 2)) +
                      '_sig'] = np.zeros(len(ramp))


class FOV_Scan():

    def __init__(self, sampleRate):
        self.sig_dict = {'x_sig': [], 'y_sig': []}
        self.sampleRate = sampleRate
        self.corr_step_size = None
        self.sequence_samples = None
        self.frames = 0

        # Update signals according to parameters.
        # Note that rounding floats to ints may cause actual scan to differ
        # slighly from expected scan.
        # Maybe either limit input parameters to numbers that "fit each other"
        # or find other solution, eg step size has to be width divided by an
        # integer. Maybe not a problem ???
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
        self.sequence_samples = int(
            np.round(
                self.sampleRate *
                par_values['sequence_time']))
        self.steps_x = int(np.ceil(size_x / step_size))
        self.steps_y = int(np.ceil(size_y / step_size))
        # Step size compatible with width
        self.corr_step_size = size_x / self.steps_x
        row_samples = self.steps_x * self.sequence_samples

        # rampAndK contains [ramp, k]
        rampAndK = makeRamp(start_x, size_x, row_samples)
        k = rampAndK[1]
        ltr_ramp = rampAndK[0]
        # rtl_ramp contains only ramp, no k since same k = -k
        rtl_ramp = ltr_ramp[::-1]
        gradual_k = makeRamp(k, -k, self.sequence_samples)
        turn_rtl = np.cumsum(gradual_k[0])
        turn_ltr = -turn_rtl
        max_turn = np.max(turn_rtl)
        adjustor = 1 - self.sequence_samples % 2

        # Create first and last part by flipping and turnign the turn_rtl array
        first_part = max_turn - \
            turn_rtl[range(int(np.ceil(self.sequence_samples / 2)),
                           self.sequence_samples)]
        mx = int(np.floor(self.sequence_samples / 2) - adjustor + 1)
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

    def update(self, devices, par_values, cycle_samples):
        for device in devices:
            signal = np.zeros(cycle_samples)
            start_name = 'start' + device
            end_name = 'end' + device
            start_pos = par_values[start_name] * self.sampleRate
            start_pos = int(min(start_pos, cycle_samples - 1))
            end_pos = par_values[end_name] * self.sampleRate
            end_pos = int(min(end_pos, cycle_samples))
            signal[range(start_pos, end_pos)] = 1
            self.sig_dict[device + 'sig'] = signal


class GraphFrame(pg.GraphicsWindow):
    """Child of pg.GraphicsWindow and creats the plot that plots the preview of
    the pulses.
    Fcn update() updates the plot of "device" with signal "signal"  """

    def __init__(self, pixel_cycle, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pixel_cycle = pixel_cycle
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
            signal = self.pixel_cycle.sig_dict[device + 'sig']
            self.plot_sig_dict[device].setData(signal)


def makeRamp(start, end, samples):

    k = (end - start) / (samples - 1)
    ramp = [start + k * i for i in range(0, samples)]
#    ramp = []
#    for i in range(0, samples):
#        ramp.append(start + k * i)
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
        V_value = a1 * D_value + a2 * D_value**2 + a3 * np.power(D_value, 3)
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
        V_value = a1 * D_value + a2 * D_value**2 + a3 * np.power(D_value, 3)
        V_signal[i] = V_value

    elapsed = time.time() - now
    print('Elapsed time: ', elapsed)
    return V_signal
