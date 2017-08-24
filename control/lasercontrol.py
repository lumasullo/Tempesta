# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from lantz import Q_


class UpdatePowers(QtCore.QObject):

    def __init__(self, laserwidget, *args, **kwargs):

        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.widget = laserwidget

    def update(self):
#        bluepower = '{:~}'.format(self.widget.bluelaser.power)
#        lilapower = '{:~}'.format(self.widget.lilalaser.power)
#        uvpower = '{:~}'.format(self.widget.uvlaser.power)
#        self.widget.blueControl.powerIndicator.setText(bluepower)
#        self.widget.lilaControl.powerIndicator.setText(lilapower)
#        self.widget.uvControl.powerIndicator.setText(uvpower)
#        time.sleep(1)
#        QtCore.QTimer.singleShot(0.01, self.update)
        pass


class LaserWidget(QtGui.QFrame):

    def __init__(self, lasers, daq, *args, **kwargs):

        super().__init__(*args, **kwargs)

#        self.lilalaser, self.exclaser, self.offlaser1, self.offlaser2 = lasers
        self.lilalaser, self.exclaser, self.offlaser, = lasers
        self.mW = Q_(1, 'mW')
        self.daq = daq

        self.lilaControl = LaserControl(self.lilalaser, '<h3>405<h3>',
                                        color=(130, 0, 200), prange=(0, 200),
                                        tickInterval=5, singleStep=0.1)

        self.excControl = LaserControl(self.exclaser, '<h3>473<h3>',
                                       color=(0, 183, 255), prange=(0, 200),
                                       tickInterval=100, singleStep=10,
                                       daq=self.daq, port=0)

        self.offControl = LaserControl(self.offlaser, '<h3>488<h3>',
                                       color=(0, 247, 255), prange=(0, 200),
                                       tickInterval=100, singleStep=10,
                                       daq=self.daq, port=0)

#        self.offControl2 = LaserControl(self.offlaser2, '<h3>488 2<h3>',
#                                        color=(0, 247, 255), prange=(0, 200),
#                                        tickInterval=100, singleStep=10,
#                                        daq=self.daq, port=0)

#        self.offControl = LaserLinkedControl(
#             [self.offlaser1, self.offlaser2], '<h3>488 Linked<h3>',
#             color=(0, 247, 255), prange=(0, 200), tickInterval=100,
#             singleStep=10, daq=self.daq, port=0)

        self.lilalaser.autostart = False
        self.offlaser.autostart = False
#        self.offlaser2.autostart = False

        self.controls = (self.lilaControl, self.excControl, self.offControl)
#                         self.offControl2, self.offControl)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        self.DigCtrl = DigitalControl(lasers=(self.lilalaser, self.exclaser,
                                              self.offlaser))

        grid.addWidget(self.lilaControl, 0, 0, 4, 1)
        grid.addWidget(self.excControl, 0, 1, 4, 1)
        grid.addWidget(self.offControl, 0, 2, 4, 1)
#        grid.addWidget(self.offControl2, 0, 3, 4, 1)
#        grid.addWidget(self.offControl, 0, 4, 4, 1)
        grid.addWidget(self.DigCtrl, 4, 0, 2, 3)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)

    def closeEvent(self, *args, **kwargs):
        self.updateThread.terminate()
        super().closeEvent(*args, **kwargs)


# TODO: adapt this to two off lasers
class DigitalControl(QtGui.QFrame):

    def __init__(self, lasers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mW = Q_(1, 'mW')
        self.lasers = lasers
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.blueOffLabel = QtGui.QLabel('<h3>488 (OFF pattern)<h3>')
        self.blueOffPower = QtGui.QLineEdit('0')
        self.blueOffPower.textChanged.connect(self.updateDigitalPowers)
        self.blueReadoutLabel = QtGui.QLabel('<h3>488 readout<h3>')
        self.blueReadoutPower = QtGui.QLineEdit('0')
        self.blueReadoutPower.textChanged.connect(self.updateDigitalPowers)
        self.lilaOnLabel = QtGui.QLabel('<h3>405 (ON pattern)<h3>')
        self.lilaOnPower = QtGui.QLineEdit('0')
        self.lilaOnPower.textChanged.connect(self.updateDigitalPowers)

        offss = "background-color: rgb{}".format((0, 247, 255))
        self.blueOffLabel.setStyleSheet(offss)
        readss = "background-color: rgb{}".format((0, 183, 255))
        self.blueReadoutLabel.setStyleSheet(readss)
        vioss = "background-color: rgb{}".format((130, 0, 200))
        self.lilaOnLabel.setStyleSheet(vioss)

        self.DigitalControlButton = QtGui.QPushButton('Digital modulation')
        self.DigitalControlButton.setCheckable(True)
        self.DigitalControlButton.clicked.connect(self.GlobalDigitalMod)
        style = "background-color: rgb{}".format((160, 160, 160))
        self.DigitalControlButton.setStyleSheet(style)

        self.updateDigPowersButton = QtGui.QPushButton('Update powers')
        self.updateDigPowersButton.clicked.connect(self.updateDigitalPowers)

#        self.startGlobalDigitalModButton = QtGui.QPushButton('START')
#        self.startGlobalDigitalModButton.clicked.connect(self.startGlobalDigitalMod)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.blueReadoutLabel, 0, 0)
        grid.addWidget(self.blueReadoutPower, 1, 0)
        grid.addWidget(self.blueOffLabel, 0, 1)
        grid.addWidget(self.blueOffPower, 1, 1)
        grid.addWidget(self.lilaOnLabel, 0, 2)
        grid.addWidget(self.lilaOnPower, 1, 2)
        grid.addWidget(self.DigitalControlButton, 2, 0)
#        grid.addWidget(self.updateDigPowersButton, 2, 1)

#    def GlobalDigitalMod(self):
#        if self.DigitalControlButton.isChecked():
#            for i in np.arange(len(self.lasers)):
#                powMag = float(self.powers[i])
#                self.lasers[i].laser.power_sp = powMag * self.mW
#        else:
#            for i in np.arange(len(self.lasers)):
#                powMag = float(self.lasers[i].laser.setPointEdit)
#                self.lasers[i].laser.power_sp = powMag * self.mW

    def GlobalDigitalMod(self):
        self.digitalPowers = [float(self.blueReadoutPower.text()),
                              float(self.blueOffPower.text()),
                              float(self.lilaOnPower.text())]

        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.lasers)):
                self.lasers[i].laser.digital_mod = True
                self.lasers[i].laser.enter_mod_mode()
                print(self.lasers[i].laser.mod_mode)
                powMag = float(self.digitalPowers[i])
                self.lasers[i].laser.power_sp = powMag * self.mW
        else:
            for i in np.arange(len(self.lasers)):
                self.lasers[i].changeEdit()
                self.lasers[i].laser.query('cp')

#                self.lasers[i].laser.enabled = True
                print('go back to continous')

    def updateDigitalPowers(self):
        self.digitalPowers = [float(self.blueReadoutPower.text()),
                              float(self.blueOffPower.text()),
                              float(self.lilaOnPower.text())]
        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.lasers)):
                powMag = float(self.digitalPowers[i])
                self.lasers[i].laser.power_sp = powMag * self.mW


class DigitalLinkedControl(QtGui.QFrame):

    def __init__(self, lasers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mW = Q_(1, 'mW')
        self.lasers = lasers
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.blueOffLabel = QtGui.QLabel('<h3>488 (OFF pattern)<h3>')
        self.blueOffPower = QtGui.QLineEdit('0')
        self.blueOffPower.textChanged.connect(self.updateDigitalPowers)
        self.blueReadoutLabel = QtGui.QLabel('<h3>488 readout<h3>')
        self.blueReadoutPower = QtGui.QLineEdit('0')
        self.blueReadoutPower.textChanged.connect(self.updateDigitalPowers)
        self.lilaOnLabel = QtGui.QLabel('<h3>405 (ON pattern)<h3>')
        self.lilaOnPower = QtGui.QLineEdit('0')
        self.lilaOnPower.textChanged.connect(self.updateDigitalPowers)

        offss = "background-color: rgb{}".format((0, 247, 255))
        self.blueOffLabel.setStyleSheet(offss)
        readss = "background-color: rgb{}".format((0, 183, 255))
        self.blueReadoutLabel.setStyleSheet(readss)
        vioss = "background-color: rgb{}".format((130, 0, 200))
        self.lilaOnLabel.setStyleSheet(vioss)

        self.DigitalControlButton = QtGui.QPushButton('Digital modulation')
        self.DigitalControlButton.setCheckable(True)
        self.DigitalControlButton.clicked.connect(self.GlobalDigitalMod)
        style = "background-color: rgb{}".format((160, 160, 160))
        self.DigitalControlButton.setStyleSheet(style)

        self.updateDigPowersButton = QtGui.QPushButton('Update powers')
        self.updateDigPowersButton.clicked.connect(self.updateDigitalPowers)

#        self.startGlobalDigitalModButton = QtGui.QPushButton('START')
#        self.startGlobalDigitalModButton.clicked.connect(self.startGlobalDigitalMod)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.blueReadoutLabel, 0, 0)
        grid.addWidget(self.blueReadoutPower, 1, 0)
        grid.addWidget(self.blueOffLabel, 0, 1)
        grid.addWidget(self.blueOffPower, 1, 1)
        grid.addWidget(self.lilaOnLabel, 0, 2)
        grid.addWidget(self.lilaOnPower, 1, 2)
        grid.addWidget(self.DigitalControlButton, 2, 0)
#        grid.addWidget(self.updateDigPowersButton, 2, 1)

#    def GlobalDigitalMod(self):
#        if self.DigitalControlButton.isChecked():
#            for i in np.arange(len(self.lasers)):
#                powMag = float(self.powers[i])
#                self.lasers[i].laser.power_sp = powMag * self.mW
#        else:
#            for i in np.arange(len(self.lasers)):
#                powMag = float(self.lasers[i].laser.setPointEdit)
#                self.lasers[i].laser.power_sp = powMag * self.mW

    def GlobalDigitalMod(self):
        self.digitalPowers = [float(self.blueReadoutPower.text()),
                              float(self.blueOffPower.text()),
                              float(self.lilaOnPower.text())]

        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.lasers)):
                self.lasers[i].laser.digital_mod = True
                self.lasers[i].laser.enter_mod_mode()
                print(self.lasers[i].laser.mod_mode)
                powMag = float(self.digitalPowers[i])
                self.lasers[i].laser.power_sp = powMag * self.mW
        else:
            for i in np.arange(len(self.lasers)):
                self.lasers[i].changeEdit()
                self.lasers[i].laser.query('cp')

#                self.lasers[i].laser.enabled = True
                print('go back to continous')

    def updateDigitalPowers(self):
        self.digitalPowers = [float(self.blueReadoutPower.text()),
                              float(self.blueOffPower.text()),
                              float(self.lilaOnPower.text())]
        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.lasers)):
                powMag = float(self.digitalPowers[i])
                self.lasers[i].laser.power_sp = powMag * self.mW


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, modulable=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port
        self.laser.digital_mod = False
        self.laser.enabled = False

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setStyleSheet("font-size:16px")
        self.name.setFixedHeight(40)

        # Power widget
        self.setPointLabel = QtGui.QLabel('Setpoint')
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(50)
        self.setPointEdit.setAlignment(QtCore.Qt.AlignRight)

        self.powerLabel = QtGui.QLabel('Power')
        powerMag = self.laser.power.magnitude
        self.powerIndicator = QtGui.QLabel(str(powerMag))
        self.powerIndicator.setFixedWidth(50)
        self.powerIndicator.setAlignment(QtCore.Qt.AlignRight)

        # Slider
        self.maxpower = QtGui.QLabel(str(prange[1]))
        self.maxpower.setAlignment(QtCore.Qt.AlignCenter)
        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setMinimum(prange[0])
        self.slider.setMaximum(prange[1])
        self.slider.setTickInterval(tickInterval)
        self.slider.setSingleStep(singleStep)
        self.slider.setValue(self.laser.power.magnitude)
        self.minpower = QtGui.QLabel(str(prange[0]))
        self.minpower.setAlignment(QtCore.Qt.AlignCenter)

        powerFrame = QtGui.QFrame(self)
        self.powerGrid = QtGui.QGridLayout()
        powerFrame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Plain)
        powerFrame.setLayout(self.powerGrid)
        self.powerGrid.addWidget(self.setPointLabel, 1, 0, 1, 2)
        self.powerGrid.addWidget(self.setPointEdit, 2, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 2, 1)
        self.powerGrid.addWidget(self.powerLabel, 3, 0, 1, 2)
        self.powerGrid.addWidget(self.powerIndicator, 4, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 4, 1)
        self.powerGrid.addWidget(self.maxpower, 0, 3)
        self.powerGrid.addWidget(self.slider, 1, 3, 8, 1)
        self.powerGrid.addWidget(self.minpower, 9, 3)

        # ON/OFF button
        self.enableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.name, 0, 0, 1, 2)
        self.grid.addWidget(powerFrame, 1, 0, 1, 2)
        self.grid.addWidget(self.enableButton, 8, 0, 1, 2)

        # Digital modulation
        if modulable:
            self.digimodButton = QtGui.QPushButton('Digital modulation')
            style = "background-color: rgb{}".format((160, 160, 160))
            self.digimodButton.setStyleSheet(style)
            self.digimodButton.setCheckable(True)
#           grid.addWidget(self.digimodButton, 6, 0)
            self.digimodButton.toggled.connect(self.digitalMod)
            # Initial values
#           self.digimodButton.setChecked(False)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

    def toggleLaser(self):
        if self.enableButton.isChecked():
            self.laser.enabled = True
        else:
            self.laser.enabled = False

    def digitalMod(self):
        if self.digimodButton.isChecked():
            self.laser.digital_mod = True
            self.laser.enter_mod_mode()
            print(self.laser.mod_mode)
        else:
            self.laser.query('cp')

    def enableLaser(self):
        self.laser.enabled = True
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW

    def changeSlider(self, value):
        self.laser.power_sp = self.slider.value() * self.mW
        self.setPointEdit.setText(str(self.laser.power_sp.magnitude))

    def changeEdit(self):
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.slider.setValue(self.laser.power_sp.magnitude)


class LaserLinkedControl(QtGui.QFrame):

    def __init__(self, lasers, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, modulable=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.lasers = lasers
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port

        for laser in self.lasers:
            laser.digital_mod = False
            laser.enabled = False

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setStyleSheet("font-size:16px")
        self.name.setFixedHeight(40)

        # Power widget
        self.setPointLabel = QtGui.QLabel('Setpoint')
        initSP = np.mean([laser.power_sp.magnitude for laser in self.lasers])
        self.setPointEdit = QtGui.QLineEdit(str(initSP))
        self.setPointEdit.setFixedWidth(50)
        self.setPointEdit.setAlignment(QtCore.Qt.AlignRight)

        self.powerLabel = QtGui.QLabel('Power')
        initPow = np.mean([laser.power.magnitude for laser in self.lasers])
        self.powerIndicator = QtGui.QLabel(str(initPow))
        self.powerIndicator.setFixedWidth(50)
        self.powerIndicator.setAlignment(QtCore.Qt.AlignRight)

        # Slider
        self.maxpower = QtGui.QLabel(str(prange[1]))
        self.maxpower.setAlignment(QtCore.Qt.AlignCenter)
        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setMinimum(prange[0])
        self.slider.setMaximum(prange[1])
        self.slider.setTickInterval(tickInterval)
        self.slider.setSingleStep(singleStep)
        self.slider.setValue(initPow)
        self.minpower = QtGui.QLabel(str(prange[0]))
        self.minpower.setAlignment(QtCore.Qt.AlignCenter)

        powerFrame = QtGui.QFrame(self)
        self.powerGrid = QtGui.QGridLayout()
        powerFrame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Plain)
        powerFrame.setLayout(self.powerGrid)
        self.powerGrid.addWidget(self.setPointLabel, 1, 0, 1, 2)
        self.powerGrid.addWidget(self.setPointEdit, 2, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 2, 1)
        self.powerGrid.addWidget(self.powerLabel, 3, 0, 1, 2)
        self.powerGrid.addWidget(self.powerIndicator, 4, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 4, 1)
        self.powerGrid.addWidget(self.maxpower, 0, 3)
        self.powerGrid.addWidget(self.slider, 1, 3, 8, 1)
        self.powerGrid.addWidget(self.minpower, 9, 3)

        # ON/OFF button
        self.enableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.name, 0, 0, 1, 2)
        self.grid.addWidget(powerFrame, 1, 0, 1, 2)
        self.grid.addWidget(self.enableButton, 8, 0, 1, 2)

        # Digital modulation
        if modulable:
            self.digimodButton = QtGui.QPushButton('Digital modulation')
            style = "background-color: rgb{}".format((160, 160, 160))
            self.digimodButton.setStyleSheet(style)
            self.digimodButton.setCheckable(True)
#           grid.addWidget(self.digimodButton, 6, 0)
            self.digimodButton.toggled.connect(self.digitalMod)
            # Initial values
#           self.digimodButton.setChecked(False)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

    def toggleLaser(self):
        if self.enableButton.isChecked():
            for laser in self.lasers:
                laser.enabled = True
        else:
            for laser in self.lasers:
                laser.enabled = False

    def digitalMod(self):
        if self.digimodButton.isChecked():
            for laser in self.lasers:
                laser.digital_mod = True
                laser.enter_mod_mode()
                print(laser.mod_mode)
        else:
            for laser in self.lasers:
                laser.query('cp')

    def enableLaser(self):
        for laser in self.lasers:
            laser.enabled = True
            laser.power_sp = float(self.setPointEdit.text()) * self.mW

    def changeSlider(self, value):
        for laser in self.lasers:
            laser.power_sp = self.slider.value() * self.mW
        self.setPointEdit.setText(str(self.lasers[0].power_sp.magnitude))

    def changeEdit(self):
        for laser in self.lasers:
            laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.slider.setValue(self.lasers[0].power_sp.magnitude)
