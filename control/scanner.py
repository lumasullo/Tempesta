# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:32:36 2016

@authors: Luciano Masullo, Andreas Bodén, Shusei Masuda, Federico Barabas,
    Aurelién Barbotin.
"""

import numpy as np
import time
import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt
import collections
import nidaqmx

import control.guitools as guitools
import control.mpl_to_pyqtgraph as mplToPg

from cv2 import rectangle, goodFeaturesToTrack, moments

# These dictionnaries contain values specific to the different axis of our
# piezo motors.
# They are the movements in µm induced by a command of 1V
convFactors = {'x': 4.06, 'y': 3.9, 'z': 10}
# Minimum and maximum voltages for the different piezos
minVolt = {'x': -10, 'y': -10, 'z': 0}
maxVolt = {'x': 10, 'y': 10, 'z': 10}


class Positionner(QtGui.QWidget):
    """This class communicates with the different analog outputs of the nidaq
    card. When not scanning, it drives the 3 axis x, y and z.

    :param ScanWidget main: main scan GUI"""

    def __init__(self, main):
        super().__init__()
        self.scanWidget = main

        # Position of the different devices in V
        self.x = 0
        self.y = 0
        self.z = 0

        # Parameters for the ramp (driving signal for the different channels)
        self.rampTime = 800  # Time for each ramp in ms
        self.sampleRate = 10**5
        self.nSamples = int(self.rampTime * 10**-3 * self.sampleRate)

        # This boolean is set to False when tempesta is scanning to prevent
        # this positionner to access the analog output channels
        self.isActive = True
        self.activeChannels = ["x", "y", "z"]

        # Axes control
        self.xEdit = QtGui.QLineEdit()
        self.xEdit.setText(str(self.x))
        self.xEdit.editingFinished.connect(self.editX)
        self.xSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.xSlider.sliderReleased.connect(self.moveX)
        self.xSlider.setRange(100*minVolt['x'], 100*maxVolt['x'])
        self.xSlider.setValue(self.x)
        xMin = "{:.2f}".format(minVolt['x']*convFactors['x'])
        self.xMinLabel = QtGui.QLabel(xMin)
        xMax = "{:.2f}".format(maxVolt['x']*convFactors['x'])
        self.xMaxLabel = QtGui.QLabel(xMax)
        self.xMaxLabel.setAlignment(QtCore.Qt.AlignRight |
                                    QtCore.Qt.AlignVCenter)
        self.xSliderLabel = QtGui.QLabel("<strong>x position (µm)</strong>")
        self.xSliderLabel.setTextFormat(QtCore.Qt.RichText)

        self.yEdit = QtGui.QLineEdit()
        self.yEdit.setText(str(self.y))
        self.yEdit.editingFinished.connect(self.editY)
        self.ySlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.ySlider.sliderReleased.connect(self.moveY)
        self.ySlider.setRange(100*minVolt['x'], 100*maxVolt['x'])
        self.ySlider.setValue(self.y)
        self.yMinLabel = QtGui.QLabel(str(minVolt['y']*convFactors['y']))
        self.yMaxLabel = QtGui.QLabel(str(maxVolt['y']*convFactors['y']))
        self.yMaxLabel.setAlignment(QtCore.Qt.AlignRight |
                                    QtCore.Qt.AlignVCenter)
        self.ySliderLabel = QtGui.QLabel("<strong>y position (µm)</strong>")
        self.ySliderLabel.setTextFormat(QtCore.Qt.RichText)

        self.zEdit = QtGui.QLineEdit()
        self.zEdit.setText(str(self.z))
        self.zEdit.editingFinished.connect(self.editZ)
        self.zSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.zSlider.sliderReleased.connect(self.moveZ)
        self.zSlider.setRange(100*minVolt['z'], 100*maxVolt['z'])
        self.zSlider.setValue(self.z)
        self.zMinLabel = QtGui.QLabel(str(minVolt['z']*convFactors['z']))
        self.zMaxLabel = QtGui.QLabel(str(maxVolt['z']*convFactors['z']))
        self.zMaxLabel.setAlignment(QtCore.Qt.AlignRight |
                                    QtCore.Qt.AlignVCenter)
        self.zSliderLabel = QtGui.QLabel("<strong>z position (µm)</strong>")
        self.zSliderLabel.setTextFormat(QtCore.Qt.RichText)

        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.xSliderLabel, 1, 0)
        layout.addWidget(self.xEdit, 1, 1, 1, 2)
        layout.addWidget(self.xMinLabel, 2, 0)
        layout.addWidget(self.xMaxLabel, 2, 2)
        layout.addWidget(self.xSlider, 3, 0, 1, 3)
        layout.addWidget(self.ySliderLabel, 5, 0)
        layout.addWidget(self.yEdit, 5, 1, 1, 2)
        layout.addWidget(self.yMinLabel, 6, 0)
        layout.addWidget(self.yMaxLabel, 6, 2)
        layout.addWidget(self.ySlider, 7, 0, 1, 3)
        layout.addWidget(self.zSliderLabel, 9, 0)
        layout.addWidget(self.zEdit, 9, 1, 1, 2)
        layout.addWidget(self.zMinLabel, 10, 0)
        layout.addWidget(self.zMaxLabel, 10, 2)
        layout.addWidget(self.zSlider, 11, 0, 1, 3)

    def move(self):
        """moves the 3 axis to the positions specified by the sliders"""
        fullSignal = np.zeros((len(self.activeChannels), self.nSamples))
        for chan in self.activeChannels:
            slider = getattr(self, chan + "Slider")
            newPos = 0.01*slider.value()
            currPos = getattr(self, chan)
            print(currPos, newPos)
            signal = makeRamp(currPos, newPos, self.nSamples)
            setattr(self, chan, newPos)
            fullSignal[self.activeChannels.index(chan)] = signal

        self.aotask = nidaqmx.Task("positionnerTask")
        # Following loop creates the voltage channels
        AOchans = [0, 1, 2]     # Order corresponds to self.channelOrder
        for n in AOchans:
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel='Dev1/ao%s' % n,
                name_to_assign_to_channel=self.activeChannels[n],
                min_val=minVolt[self.activeChannels[n]],
                max_val=maxVolt[self.activeChannels[n]])

        self.aotask.timing.cfg_samp_clk_timing(
            rate=self.sampleRate,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.nSamples)

        self.aotask.write(fullSignal, auto_start=True)
        self.aotask.wait_until_done()
        self.aotask.stop()
        self.aotask.close()

    def moveX(self):
        """Specifies the movement of the x axis."""
        value = self.xSlider.value() / 100
        self.xEdit.setText(str(round(value*convFactors['x'], 2)))
        self.move()

    def moveY(self):
        """Specifies the movement of the y axis."""
        value = self.ySlider.value() / 100
        self.yEdit.setText(str(round(value*convFactors['y'], 2)))
        self.move()

    def moveZ(self):
        """Specifies the movement of the z axis."""
        value = self.zSlider.value() / 100
        self.zEdit.setText(str(round(value*convFactors['z'], 2)))
        self.move()

    def editX(self):
        """Method called when a position for x is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        self.xSlider.setValue(100*float(self.xEdit.text())/convFactors['x'])
        self.move()

    def editY(self):
        """Method called when a position for y is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        self.ySlider.setValue(100*float(self.yEdit.text())/convFactors['y'])
        self.move()

    def editZ(self):
        """Method called when a position for z is entered manually. Repositions
        the slider and initiates the movement of the stage"""
        self.zSlider.setValue(100*float(self.zEdit.text())/convFactors['z'])
        self.move()

    def setX(self, value):
        """This method sets x to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        valueLine = round(value * convFactors['x'], 2)
        print("in set x", value, valueLine)
        self.xSlider.setValue(value * 100)
        self.xEdit.setText(str(valueLine))
        self.move()

    def setY(self, value):
        """This method sets y to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        self.ySlider.setValue(value * 100)
        self.yEdit.setText(str(round(value*convFactors['y'], 2)))
        self.move()

    def setZ(self, value):
        """This method sets x to value in Volts and moves accordingly the
        slider and the corresponding value line"""
        self.zSlider.setValue(value * 100)
        self.zEdit.setText(str(round(value*convFactors['z'], 2)))
        self.move()

    def goToZero(self):
        self.x = 0
        self.y = 0
        self.z = 0
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
            totalChannels = ["x", "y", "z"]
            # returns a list containing the axis not in use
            self.activeChannels = [
                x for x in totalChannels if x not in channels]
            self.aotask = nidaqmx.Task("positionnerTask")
            axis = self.activeChannels[0]
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

            totalChannels = ["x", "y", "z"]
            self.activeChannels = totalChannels
            for elt in totalChannels:
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
            self.xSlider.setValue(0)
            self.ySlider.setValue(0)
            self.zSlider.setValue(0)
            self.move()
            self.aotask.wait_until_done(timeout=2)
            self.aotask.stop()
            self.aotask.close()


class ScanWidget(QtGui.QMainWindow):
    ''' This class is intended as a widget in the bigger GUI, Thus all the
    commented parameters etc. It contain an instance of stageScan and
    pixel_scan which in turn harbour the analog and digital signals
    respectively.
    The function run starts the communication with the Nidaq through the
    Scanner object. This object was initially written as a QThread object but
    is not right now.
    As seen in the commened lines of run() I also tried running in a QThread
    created in run().
    The rest of the functions contain mostly GUI related code.'''
    def __init__(self, device, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scanInLiveviewWar = QtGui.QMessageBox()
        self.scanInLiveviewWar.setInformativeText(
            "You need to be in liveview to scan")

        self.digModWarning = QtGui.QMessageBox()
        self.digModWarning.setInformativeText(
            "You need to be in digital laser modulation and external "
            "frame-trigger acquisition mode")

        self.nidaq = device
        self.main = main

        # The port order in the NIDAQ follows this same order.
        # We chose to follow the temporal sequence order
        self.allDevices = ['405', '488', '473', 'CAM']
        self.channelOrder = ['x', 'y', 'z']

        self.saveScanBtn = QtGui.QPushButton('Save Scan')

        def saveScanFcn(): return guitools.saveScan(self)
        self.saveScanBtn.clicked.connect(saveScanFcn)
        self.loadScanBtn = QtGui.QPushButton('Load Scan')

        def loadScanFcn(): return guitools.loadScan(self)
        self.loadScanBtn.clicked.connect(loadScanFcn)

        self.sampleRateEdit = QtGui.QLineEdit()

        self.sizeXPar = QtGui.QLineEdit('2')
        self.sizeXPar.textChanged.connect(
            lambda: self.scanParameterChanged('sizeX'))
        self.sizeYPar = QtGui.QLineEdit('2')
        self.sizeYPar.textChanged.connect(
            lambda: self.scanParameterChanged('sizeY'))
        self.sizeZPar = QtGui.QLineEdit('10')
        self.sizeZPar.textChanged.connect(
            lambda: self.scanParameterChanged('sizeZ'))
        self.seqTimePar = QtGui.QLineEdit('10')     # ms
        self.seqTimePar.textChanged.connect(
            lambda: self.scanParameterChanged('seqTime'))
        self.nrFramesPar = QtGui.QLabel()
        self.scanDuration = 0
        self.scanDurationLabel = QtGui.QLabel(str(self.scanDuration))
        self.stepSizeXYPar = QtGui.QLineEdit('0.1')
        self.stepSizeXYPar.textChanged.connect(
            lambda: self.scanParameterChanged('stepSizeXY'))
        self.stepSizeZPar = QtGui.QLineEdit('1')
        self.stepSizeZPar.textChanged.connect(
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

        self.start405Par = QtGui.QLineEdit('0')
        self.start405Par.textChanged.connect(
            lambda: self.pxParameterChanged('start405'))
        self.start488Par = QtGui.QLineEdit('2.5')
        self.start488Par.textChanged.connect(
            lambda: self.pxParameterChanged('start488'))
        self.start473Par = QtGui.QLineEdit('0')
        self.start473Par.textChanged.connect(
            lambda: self.pxParameterChanged('start473'))
        self.startCAMPar = QtGui.QLineEdit('2.5')
        self.startCAMPar.textChanged.connect(
            lambda: self.pxParameterChanged('startCAM'))

        self.end405Par = QtGui.QLineEdit('0')
        self.end405Par.textChanged.connect(
            lambda: self.pxParameterChanged('end405'))
        self.end488Par = QtGui.QLineEdit('7.5')
        self.end488Par.textChanged.connect(
            lambda: self.pxParameterChanged('end488'))
        self.end473Par = QtGui.QLineEdit('0')
        self.end473Par.textChanged.connect(
            lambda: self.pxParameterChanged('end473'))
        self.endCAMPar = QtGui.QLineEdit('7.5')
        self.endCAMPar.textChanged.connect(
            lambda: self.pxParameterChanged('endCAM'))

        self.pxParameters = {'start405': self.start405Par,
                             'start488': self.start488Par,
                             'start473': self.start473Par,
                             'startCAM': self.startCAMPar,
                             'end405': self.end405Par,
                             'end488': self.end488Par,
                             'end473': self.end473Par,
                             'endCAM': self.endCAMPar}

        self.pxParValues = {'start405': 0.001*float(self.start405Par.text()),
                            'start488': 0.001*float(self.start488Par.text()),
                            'start473': 0.001*float(self.start473Par.text()),
                            'startCAM': 0.001*float(self.startCAMPar.text()),
                            'end405': 0.001*float(self.end405Par.text()),
                            'end488': 0.001*float(self.end488Par.text()),
                            'end473': 0.001*float(self.end473Par.text()),
                            'endCAM': 0.001*float(self.endCAMPar.text())}

        self.stageScan = StageScan(self.sampleRate)
        self.pxCycle = PixelCycle(self.sampleRate)
        self.graph = GraphFrame(self.pxCycle)
        self.graph.plot.getAxis('bottom').setScale(1000/self.sampleRate)
        self.graph.setFixedHeight(100)
        self.updateScan(self.allDevices)
        self.scanParameterChanged('seqTime')

        self.multiScanWgt = MultipleScanWidget(self)

        self.scanRadio = QtGui.QRadioButton('Scan')
        self.scanRadio.clicked.connect(lambda: self.setScanOrNot(True))
        self.scanRadio.setChecked(True)
        self.contLaserPulsesRadio = QtGui.QRadioButton('Cont. Laser Pulses')
        self.contLaserPulsesRadio.clicked.connect(
            lambda: self.setScanOrNot(False))

        self.scanButton = QtGui.QPushButton('Scan')
        self.scanning = False
        self.scanButton.clicked.connect(self.scanOrAbort)
        self.previewButton = QtGui.QPushButton('Plot scan path')
        self.previewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                         QtGui.QSizePolicy.Expanding)
        self.previewButton.clicked.connect(self.previewScan)
        self.continuousCheck = QtGui.QCheckBox('Repeat')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        grid.addWidget(self.loadScanBtn, 0, 0)
        grid.addWidget(self.saveScanBtn, 0, 1)
        grid.addWidget(self.scanRadio, 0, 2)
        grid.addWidget(self.contLaserPulsesRadio, 0, 3)
        grid.addWidget(self.scanButton, 0, 4, 1, 2)
        grid.addWidget(self.continuousCheck, 0, 6)

        grid.addWidget(QtGui.QLabel('Size X (µm):'), 2, 0)
        grid.addWidget(self.sizeXPar, 2, 1)
        grid.addWidget(QtGui.QLabel('Size Y (µm):'), 3, 0)
        grid.addWidget(self.sizeYPar, 3, 1)
        grid.addWidget(QtGui.QLabel('Size Z (µm):'), 4, 0)
        grid.addWidget(self.sizeZPar, 4, 1)
        grid.addWidget(QtGui.QLabel('Step XY (µm):'), 2, 2)
        grid.addWidget(self.stepSizeXYPar, 2, 3)
        grid.addWidget(QtGui.QLabel('Step Z (µm):'), 4, 2)
        grid.addWidget(self.stepSizeZPar, 4, 3)

        grid.addWidget(QtGui.QLabel('Mode:'), 2, 4)
        grid.addWidget(self.scanMode, 2, 5)
        grid.addWidget(QtGui.QLabel('Primary dimension:'), 3, 4)
        grid.addWidget(self.primScanDim, 3, 5)
        grid.addWidget(QtGui.QLabel('Number of frames:'), 4, 4)
        grid.addWidget(self.nrFramesPar, 4, 5)
        grid.addWidget(self.previewButton, 2, 6, 3, 2)

        grid.addWidget(QtGui.QLabel('Dwell time (ms):'), 7, 0)
        grid.addWidget(self.seqTimePar, 7, 1)
        grid.addWidget(QtGui.QLabel('Total time (s):'), 7, 2)
        grid.addWidget(self.scanDurationLabel, 7, 3)
        grid.addWidget(QtGui.QLabel('Start (ms):'), 8, 1)
        grid.addWidget(QtGui.QLabel('End (ms):'), 8, 2)
        grid.addWidget(QtGui.QLabel('405:'), 9, 0)
        grid.addWidget(self.start405Par, 9, 1)
        grid.addWidget(self.end405Par, 9, 2)
        grid.addWidget(QtGui.QLabel('488:'), 10, 0)
        grid.addWidget(self.start488Par, 10, 1)
        grid.addWidget(self.end488Par, 10, 2)
        grid.addWidget(QtGui.QLabel('473:'), 11, 0)
        grid.addWidget(self.start473Par, 11, 1)
        grid.addWidget(self.end473Par, 11, 2)
        grid.addWidget(QtGui.QLabel('Camera:'), 12, 0)
        grid.addWidget(self.startCAMPar, 12, 1)
        grid.addWidget(self.endCAMPar, 12, 2)
        grid.addWidget(self.graph, 8, 3, 5, 5)

        grid.addWidget(self.multiScanWgt, 13, 0, 4, 9)

        grid.setColumnMinimumWidth(6, 160)
        grid.setRowMinimumHeight(1, 10)
        grid.setRowMinimumHeight(6, 10)
        grid.setRowMinimumHeight(13, 10)

    @property
    def scanOrNot(self):
        return self._scanOrNot

    @scanOrNot.setter
    def scanOrNot(self, value):
        self.enableScanPars(value)
        self.scanButton.setCheckable(not value)

    def enableScanPars(self, value):
        self.sizeXPar.setEnabled(value)
        self.sizeYPar.setEnabled(value)
        self.stepSizeXYPar.setEnabled(value)
        self.scanMode.setEnabled(value)
        self.primScanDim.setEnabled(value)
        if value:
            self.scanButton.setText('Scan')
        else:
            self.scanButton.setText('Run')

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
        self.scanDuration = self.stageScan.frames*self.scanParValues['seqTime']
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
            main = self.main
            lasers = main.laserWidgets
            if not(lasers.DigCtrl.DigitalControlButton.isChecked() and
                   main.trigsourceparam.value() == 'External "frame-trigger"'):
                main.trigsourceparam.setValue('External "frame-trigger"')
                main.laserWidgets.DigCtrl.DigitalControlButton.setChecked(True)
                main.laserWidgets.DigCtrl.GlobalDigitalMod()

            self.prepAndRun()
        else:
            self.scanner.abort()

    def prepAndRun(self, continuous=False):
        ''' Only called if scanner is not running (See scanOrAbort function).
        '''
        if self.scanRadio.isChecked():
            self.stageScan.update(self.scanParValues)
            self.scanButton.setText('Abort')
            self.scanner = Scanner(
               self.nidaq, self.stageScan, self.pxCycle, self, continuous)
            self.scanner.finalizeDone.connect(self.finalizeDone)
            self.scanner.scanDone.connect(self.scanDone)
            self.scanning = True

            self.main.lvworkers[0].startRecording()

            self.scanner.runScan()

        elif self.scanButton.isChecked():
            self.lasercycle = LaserCycle(self.nidaq, self.pxCycle)
            self.scanButton.setText('Stop')
            self.lasercycle.run()

        else:
            self.lasercycle.stop()
            self.scanButton.setText('Run')
            del self.lasercycle

    def scanDone(self):
        self.scanButton.setEnabled(False)

        buildImg = self.multiScanWgt.makeImgBox.isChecked()
        if not self.scanner.aborted:
            time.sleep(0.1)

            self.main.lvworkers[0].stopRecording()
            data = self.main.lvworkers[0].framesRecorded

            datashape = (len(data), *self.main.shapes[0][::-1])
            reshapeddata = np.reshape(data, datashape, order='C')

#            self.multiScanWgt.worker.set_images(reshapeddata[1:])
            self.multiScanWgt.worker.set_images(reshapeddata)
            if buildImg:
                self.multiScanWgt.worker.find_fp()
                self.multiScanWgt.worker.analyze()

    def finalizeDone(self):
        if (not self.continuousCheck.isChecked()) or self.scanner.aborted:
            self.scanButton.setText('Scan')
            self.scanButton.setEnabled(True)
            del self.scanner
            self.scanning = False
        elif self.continuousCheck.isChecked():
            self.scanButton.setEnabled(True)
            self.prepAndRun(True)
        else:
            self.scanButton.setEnabled(True)
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

    def stop(self):
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
    :param ScanWidget main: main scan GUI."""

    scanDone = QtCore.pyqtSignal()
    finalizeDone = QtCore.pyqtSignal()

    def __init__(self, device, stageScan, pxCycle, main, continuous=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nidaq = device
        self.stageScan = stageScan
        self.pxCycle = pxCycle
        self.continuous = continuous

        self.sampsInScan = len(self.stageScan.sigDict['x'])
        self.main = main

        self.aotask = nidaqmx.Task('aotask')
        self.dotask = nidaqmx.Task('dotask')
        self.waiter = WaitThread(self.aotask)

        self.scanTimeW = QtGui.QMessageBox()
        self.scanTimeW.setInformativeText("Are you sure you want to continue?")
        self.scanTimeW.setStandardButtons(QtGui.QMessageBox.Yes |
                                          QtGui.QMessageBox.No)
        self.channelOrder = main.channelOrder

        self.aborted = False

    def runScan(self):
        self.aborted = False
        scanTime = self.sampsInScan / self.main.sampleRate
        ret = QtGui.QMessageBox.Yes
        self.scanTimeW.setText("Scan will take %s seconds" % scanTime)
        if scanTime > 10 and not(self.continuous):
            ret = self.scanTimeW.exec_()

        if ret == QtGui.QMessageBox.No:
            self.done()
            return

        # Following loop creates the voltage channels
        AOchans = [0, 1, 2]     # Order corresponds to self.channelOrder
        for n in AOchans:
            self.aotask.ao_channels.add_ao_voltage_chan(
                physical_channel='Dev1/ao%s' % n,
                name_to_assign_to_channel='chan_%s' % self.channelOrder[n],
                min_val=minVolt[self.channelOrder[n]],
                max_val=maxVolt[self.channelOrder[n]])

        fullAOsig = np.array(
            [self.stageScan.sigDict[self.channelOrder[i]] for i in AOchans])

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

        fullDOsig = np.array(
            [self.pxCycle.sigDict[devs[i]] for i in DOchans])

        """When doing unidirectional scan, the time needed for the stage to
        move back to the initial x needs to be filled with zeros/False.
        This time is now set to the time spanned by 1 sequence.
        Therefore, the digital signal is assambled as the repetition of the
        sequence for the whole scan in one row and then append zeros for 1
        sequence time. THIS IS NOW INCOMPATIBLE WITH VOLUMETRIC SCAN, maybe."""
        if self.stageScan.primScanDim == 'x':
            primSteps = self.stageScan.FOVscan.stepsX
        else:
            primSteps = self.stageScan.FOVscan.stepsY
        # Signal for a single line
        lineSig = np.tile(fullDOsig, primSteps)
        emptySig = np.zeros((4, int(self.stageScan.seqSamps)), dtype=bool)
        fullDOsig = np.concatenate((emptySig, lineSig, emptySig), axis=1)

        # TODO: adapt this to work with unidirectional scanning (see before)
        """If doing VOLume scan, the time needed for the stage to move
        between z-planes needs to be filled with zeros/False. This time is
        equal to one "sequence-time". To do so we first have to repeat the
        sequence for the whole scan in one plane and then append zeros."""
        if self.stageScan.scanMode == 'VOLscan':
            fullDOsig = np.tile(
                fullDOsig, self.stageScan.VOLscan.cyclesPerSlice)
            fullDOsig = np.concatenate(
                (fullDOsig, np.zeros(4, self.stageScan.seqSamps)))

        self.dotask.timing.cfg_samp_clk_timing(
            rate=self.pxCycle.sampleRate,
            source=r'ao/SampleClock',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=self.sampsInScan)

        self.waiter.waitdoneSignal.connect(self.finalize)

        self.aotask.write(fullAOsig, auto_start=False)
        self.dotask.write(fullDOsig, auto_start=False)

        self.dotask.start()
        self.aotask.start()

        self.waiter.start()

    def abort(self):
        self.aborted = True
        self.waiter.stop()
        self.aotask.stop()
        self.dotask.stop()
        self.finalize()

    def finalize(self):
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
        dim = [self.channelOrder[i] for i in chans]
        finalSamps = [
            self.stageScan.sigDict[dim[i]][writtenSamps - 1] for i in chans]
        returnRamps = np.array(
            [makeRamp(finalSamps[i], 0, self.stageScan.sampleRate)
             for i in chans])

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


class MultipleScanWidget(QtGui.QFrame):

    def __init__(self, main):
        super().__init__()

        self.main = main

        # make illumination image widget
        self.illum_wgt = IllumImageWidget()

        self.makeImgBox = QtGui.QCheckBox('Build scan image')

        # Crosshair
        self.crosshair = guitools.Crosshair(self.illum_wgt.vb)
        self.crossButton = QtGui.QPushButton('Crosshair')
        self.crossButton.setCheckable(True)
        self.crossButton.pressed.connect(self.crosshair.toggle)

        # make worker
        self.worker = MultiScanWorker(self, self.main)

        # make other GUI componentsa
        self.analysis_btn = QtGui.QPushButton('Analyze')
        self.analysis_btn.clicked.connect(self.worker.analyze)
        self.analysis_btn.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                        QtGui.QSizePolicy.Expanding)
        self.show_beads_btn = QtGui.QPushButton('Show beads')
        self.show_beads_btn.clicked.connect(self.worker.find_fp)
        self.quality_label = QtGui.QLabel('Quality level of points')
        self.quality_edit = QtGui.QLineEdit('0.05')
        self.quality_edit.editingFinished.connect(self.worker.find_fp)
        self.win_size_label = QtGui.QLabel('Window size [px]')
        self.win_size_edit = QtGui.QLineEdit('10')
        self.win_size_edit.editingFinished.connect(self.worker.find_fp)

        self.beads_label = QtGui.QLabel('Bead number')
        self.beads_box = QtGui.QComboBox()
        self.beads_box.activated.connect(self.change_illum_image)
        self.change_beads_button = QtGui.QPushButton('Change')
        self.change_beads_button.clicked.connect(self.next_bead)
        self.overlay_box = QtGui.QComboBox()
        self.overlay_box.activated.connect(self.worker.overlay)
        self.overlay_check = QtGui.QCheckBox('Overlay')
        self.overlay_check.stateChanged.connect(self.worker.overlay)
        self.clear_btn = QtGui.QPushButton('Clear')
        self.clear_btn.clicked.connect(self.clear)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        grid.addWidget(self.makeImgBox, 0, 0)
        grid.addWidget(self.crossButton, 0, 1)
        grid.addWidget(self.illum_wgt, 1, 0, 1, 8)

        grid.addWidget(self.quality_label, 2, 0)
        grid.addWidget(self.quality_edit, 2, 1)
        grid.addWidget(self.win_size_label, 3, 0)
        grid.addWidget(self.win_size_edit, 3, 1)
        grid.addWidget(self.show_beads_btn, 2, 2)
        grid.addWidget(self.analysis_btn, 3, 2)

        grid.addWidget(self.beads_label, 2, 4)
        grid.addWidget(self.beads_box, 2, 5)
        grid.addWidget(self.change_beads_button, 3, 4, 1, 2)
        grid.addWidget(self.overlay_check, 2, 6)
        grid.addWidget(self.overlay_box, 2, 7)
        grid.addWidget(self.clear_btn, 3, 6, 1, 2)

        grid.setColumnMinimumWidth(3, 100)

    def change_illum_image(self):
        self.worker.delete_label()
        curr_ind = self.beads_box.currentIndex()
        self.illum_wgt.update(self.worker.illum_images[curr_ind])
        self.illum_wgt.vb.autoRange()
        if self.overlay_check.isChecked():
            self.illum_wgt.update_back(self.worker.illum_images_back[curr_ind])

    def next_bead(self):
        self.worker.delete_label()
        curr_ind = self.beads_box.currentIndex()
        if len(self.worker.illum_images) == curr_ind + 1:
            next_ind = 0
        else:
            next_ind = curr_ind + 1
        self.illum_wgt.update(self.worker.illum_images[next_ind])
        self.beads_box.setCurrentIndex(next_ind)
        if self.overlay_check.isChecked():
            self.illum_wgt.update_back(self.worker.illum_images_back[next_ind])
        self.illum_wgt.vb.autoRange()

    def clear(self):
        self.worker.illum_images_stocked = []
        self.overlay_box.clear()


class MultiScanWorker(QtCore.QObject):

    def __init__(self, main_wgt, mainScanWid):
        super().__init__()

        self.main = main_wgt
        self.mainScanWid = mainScanWid
        self.illum_images = []
        self.illum_images_stocked = []
        self.labels = []

        # corner detection parameter of Shi-Tomasi
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.1,
                                   minDistance=7,
                                   blockSize=7)

    def set_images(self, images):
        self.primScanDim = self.mainScanWid.scanner.stageScan.primScanDim
        if self.primScanDim == 'x':
            self.steps = [self.mainScanWid.scanner.stageScan.FOVscan.stepsX,
                          self.mainScanWid.scanner.stageScan.FOVscan.stepsY]
        else:
            self.steps = [self.mainScanWid.scanner.stageScan.FOVscan.stepsY,
                          self.mainScanWid.scanner.stageScan.FOVscan.stepsX]
        self.images = images

    def find_fp(self):
        self.main.illum_wgt.delete_back()

        # find feature points
        ql = float(self.main.quality_edit.text())
        self.feature_params['qualityLevel'] = ql
        self.radius = int(self.main.win_size_edit.text())
        self.nor_const = 255 / (np.max(self.images))
        self.f_frame = (self.images[1] * self.nor_const).astype(np.uint8)
        self.l_frame = (self.images[-1] * self.nor_const).astype(np.uint8)
        fps_f = goodFeaturesToTrack(
            self.f_frame, mask=None, **self.feature_params)
        self.fps_f = np.array([point[0] for point in fps_f])
        fps_l = goodFeaturesToTrack(
            self.l_frame, mask=None, **self.feature_params)
        self.fps_l = np.array([point[0] for point in fps_l])

        # make frame for visualizing feature point detection
        self.frame_view = (self.f_frame + self.l_frame) / 2

        # draw feature points image
        self.delete_label()
        self.centers = []   # center points between first FPs and last FPs
        self.fps_ll = []    # FPs of last frame that match ones of fist frame
        for i, fp_f in enumerate(self.fps_f):
            distances = [np.linalg.norm(fp_l - fp_f) for fp_l in self.fps_l]
            ind = np.argmin(distances)
            self.fps_ll.append(self.fps_l[ind])
            center = (fp_f + self.fps_l[ind]) / 2
            self.centers.append(center)
            # draw calculating window
            rectangle(self.frame_view,
                      (int(center[0]-self.radius), int(center[1]-self.radius)),
                      (int(center[0]+self.radius), int(center[1]+self.radius)),
                      255, 1)
            # make labels for each window
            label = pg.TextItem()
            label.setPos(center[0] + self.radius, center[1] + self.radius)
            label.setText(str(i))
            self.labels.append(label)
            self.main.illum_wgt.vb.addItem(label)
        self.main.illum_wgt.update(self.frame_view.T, invert=False)
        self.main.illum_wgt.vb.autoRange()

    def analyze(self):
        self.main.beads_box.clear()
        self.illum_images = []

        self.delete_label()

        data_mean = []  # means of calculating window for each images
        cps_f = []  # center points of beads in first frame
        cps_l = []  # center points of beads in last frame
        for i in range(len(self.centers)):
            data_mean.append([])
            # record the center point of gravity
            cps_f.append(self.find_cp(
                self.f_frame, self.fps_f[i].astype(np.uint16), self.radius))
            cps_l.append(self.find_cp(
                self.l_frame, self.fps_ll[i].astype(np.uint16), self.radius))

            # calculate the mean of calculating window
            for image in self.images:
                mean = self.mean_roi(
                    image, self.centers[i].astype(np.uint16), self.radius)
                data_mean[i].append(mean)

        # reconstruct the illumination image
        for i in range(len(data_mean)):
            data_r = np.reshape(data_mean[i], self.steps)
            if self.primScanDim == 'y':
                data_r = data_r.T
            self.illum_images.append(data_r)
            self.main.beads_box.addItem(str(i))

        # stock images for overlaying
        self.illum_images_stocked.append(self.illum_images)
        self.main.overlay_box.addItem(str(self.main.overlay_box.count()))

        # make large field of view of illumination image
        # expand beads image
        dif = []
        for i in range(len(cps_f)):
            dif_x = cps_f[i][0] - cps_l[i][0]
            dif_y = cps_f[i][1] - cps_l[i][1]
            dif.append(max(dif_x, dif_y))
        rate = max(self.steps) / np.average(dif)
        img_large = np.zeros((int(self.f_frame.shape[0]*rate),
                              int(self.f_frame.shape[1]*rate))).T

        self.points_large = []  # start and end points of illumination image
        for point, illum_image in zip(self.fps_f, self.illum_images):
            px = img_large.shape[1] - (point[1] * rate).astype(int)
            py = img_large.shape[0] - (point[0] * rate).astype(int)
            if self.primScanDim == 'x':
                pxe = min(px+self.steps[1], img_large.shape[1])
                pye = min(py+self.steps[0], img_large.shape[0])
            else:
                pxe = min(px+self.steps[0], img_large.shape[1])
                pye = min(py+self.steps[1], img_large.shape[0])
            img_large[py:pye, px:pxe] = illum_image[0:pye-py, 0:pxe-px]
            self.points_large.append([px, py, pxe, pye])
        self.illum_images.append(img_large)

        # update illumination image
        self.main.illum_wgt.update(img_large)
        self.main.beads_box.addItem('large field of view')
        self.main.beads_box.setCurrentIndex(len(self.illum_images)-1)
        self.main.illum_wgt.vb.autoRange()

        # do not display large view if bead is only one
        if len(self.illum_images) == 2:
            self.main.next_bead()

    def overlay(self):
        ind = self.main.overlay_box.currentIndex()
        self.illum_images_back = [] # illumination images for overlay

        # overlay previous image to current image
        if self.main.overlay_check.isChecked():

            # process the large field of view
            illum_image_large_pre = np.zeros(self.illum_images[-1].shape)
            for i, point in enumerate(self.points_large):
                px, py, pxe, pye = point
                illum_image_pre = self.illum_images_stocked[ind][i]
                illum_image_large_pre[py:pye, px:pxe] = illum_image_pre[0:pye-py, 0:pxe-px]

            # process each image
            for i in range(len(self.illum_images)-1):
                illum_image_pre = self.illum_images_stocked[ind][i]
                self.illum_images_back.append(illum_image_pre)

            self.illum_images_back.append(illum_image_large_pre)

            # update the background image
            img = self.illum_images_back[self.main.beads_box.currentIndex()]
            self.main.illum_wgt.update_back(img)

        else:
            self.illum_images_back.clear()
            self.main.illum_wgt.delete_back()

    def delete_label(self):
        # delete beads label
        if len(self.labels) != 0:
            for label in self.labels:
                self.main.illum_wgt.vb.removeItem(label)
            self.labels = []

    @staticmethod
    def mean_roi(array, p, r):
        xs = max(p[0] - r, 0)
        xe = min(p[0] + r, array.shape[1])
        ys = max(p[1] - r, 0)
        ye = min(p[1] + r, array.shape[0])
        roi = array[ys: ye, xs: xe]
        return np.average(roi)

    @staticmethod
    def find_cp(array, point, r):
        xs = max(point[0] - r, 0)
        xe = min(point[0] + r, array.shape[1])
        ys = max(point[1] - r, 0)
        ye = min(point[1] + r, array.shape[0])
        roi = array[ys: ye, xs: xe]
        M = moments(roi, False)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return [int(cx + point[0] - r), int(cy + point[1] - r)]


class IllumImageWidget(pg.GraphicsLayoutWidget):

    def __init__(self):
        super().__init__()

        self.vb = self.addViewBox(row=1, col=1)
        self.vb.setAspectLocked(True)
        self.vb.enableAutoRange()

        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.vb.setLimits(yMin=0, yMax=66000)
#        pos, rgba = zip(*mplToPg.cmapToColormap(plt.get_cmap('hot')))
#        redsColormap = pg.ColorMap(pos, rgba)
        redsColormap = pg.ColorMap([0, 1], [(0, 0, 0), (255, 0, 0)])
        self.hist.gradient.setColorMap(redsColormap)
        for tick in self.hist.gradient.ticks:
            tick.hide()
        self.addItem(self.hist, row=1, col=2)

        self.img_back = pg.ImageItem()
        self.vb.addItem(self.img_back)
        self.img_back.setZValue(10)
        self.img_back.setOpacity(0.5)
        self.hist_back = pg.HistogramLUTItem(image=self.img_back)
        self.hist_back.vb.setLimits(yMin=0, yMax=66000)
        pos, rgba = zip(*mplToPg.cmapToColormap(plt.get_cmap('viridis')))
        greensColormap = pg.ColorMap(pos, rgba)
#        greensColormap = pg.ColorMap([0, 1], [(0, 0, 0), (0, 250, 0)])
        self.hist_back.gradient.setColorMap(greensColormap)
        for tick in self.hist_back.gradient.ticks:
            tick.hide()
        self.addItem(self.hist_back, row=1, col=3)

    def update(self, img, invert=True):
        self.img.setImage(img)
        if invert:
            self.vb.invertX(True)
            self.vb.invertY(True)
        else:
            self.vb.invertX(False)
            self.vb.invertY(False)
        self.hist.setLevels(*guitools.bestLimits(img))

    def update_back(self, img):
        self.img_back.setImage(img)
        self.hist_back.setLevels(*guitools.bestLimits(img))

    def delete_back(self):
        self.img_back.clear()


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
        fullDOsig = np.array(
            [self.pxCycle.sigDict[devs[i]] for i in DOchans])

        self.dotask.timing.cfg_samp_clk_timing(
           source=r'100kHzTimeBase',
           rate=self.pxCycle.sampleRate,
           sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        self.dotask.write(fullDOsig, auto_start=False)

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
        startX = startY = 0
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

        if primScanDim == 'x':
            self.makePrimDimSig('x', startX, sizeX, self.stepsX, self.stepsY)
            self.makeSecDimSig('y', startY, sizeY, self.stepsY, self.stepsX)
        elif primScanDim == 'y':
            self.makePrimDimSig('y', startY, sizeY, self.stepsY, self.stepsX)
            self.makeSecDimSig('x', startX, sizeX, self.stepsX, self.stepsY)

        self.sigDict['z'] = np.zeros(len(self.secSig))

    def makePrimDimSig(self, dim, start, size, steps, otherSteps):
        rowSamps = steps * self.seqSamps
        LTRramp = makeRamp(start, size, rowSamps)
        # Fast return to startX
        RTLramp = makeRamp(size, start, self.seqSamps)
        LTRramp = np.concatenate((start*np.ones(self.seqSamps), LTRramp))
        LTRTLramp = np.concatenate((LTRramp, RTLramp))
        self.primSig = np.tile(LTRTLramp, otherSteps)
        self.sigDict[dim] = self.primSig / convFactors[dim]

    def makeSecDimSig(self, dim, start, size, steps, otherSteps):
        # y axis scan signal
        colSamps = steps * self.seqSamps
        Yramp = makeRamp(start, size, colSamps)
        Yramps = np.split(Yramp, steps)
        constant = np.ones((otherSteps + 1)*self.seqSamps)
        Sig = np.array([np.concatenate((i[0]*constant, i)) for i in Yramps])
        self.secSig = Sig.ravel()
        self.sigDict[dim] = self.secSig / convFactors[dim]

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
        XTransition = smoothRamp(Xsig[-1], Xsig[0], self.seqSamps)
        YTransition = smoothRamp(Ysig[-1], Ysig[0], self.seqSamps)
        ZTransition = smoothRamp(0, self.corrStepSizeZ, self.seqSamps)

        for i in range(1, self.stepsZ - 1):
            fullXsig = np.concatenate((fullXsig, XTransition))
            fullYsig = np.concatenate((fullYsig, YTransition))
            fullZsig = np.concatenate((fullZsig, newValue + ZTransition))

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


class PixelCycle():
    ''' Contains the digital signals for the pixel cycle. The update function
    takes a parameter_values dict and updates the signal accordingly.'''
    def __init__(self, sampleRate):
        self.sigDict = collections.OrderedDict(
            [('405', []), ('488', []), ('473', []), ('CAM', [])])
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
                            '488': self.plot.plot(pen=pg.mkPen(0, 247, 255)),
                            '473': self.plot.plot(pen=pg.mkPen(0, 183, 255)),
                            'CAM': self.plot.plot(pen='w')}

    def update(self, devices=None):
        if devices is None:
            devices = self.plotSigDict

        for device in devices:
            signal = self.pxCycle.sigDict[device]
            self.plotSigDict[device].setData(signal)


def makeRamp(start, end, samples):
    return np.linspace(start, end, num=samples)


def smoothRamp(start, end, samples):
    x = np.linspace(start, end, num=samples, endpoint=True)
    signal = start + (end - start) * 0.5*(1 - np.cos(x * np.pi))
    return signal
