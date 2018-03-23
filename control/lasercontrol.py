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

        self.actlaser, self.offlaser, self.exclaser = lasers

        self.mW = Q_(1, 'mW')
        self.daq = daq

        self.actControl = LaserControl(
            self.actlaser, '<h3>405<h3>', color=(130, 0, 200), prange=(0, 200),
            tickInterval=5, singleStep=0.1)

        self.offControl = LaserControl(
            self.offlaser, '<h3>488<h3>', color=(0, 247, 255), prange=(0, 200),
            tickInterval=100, singleStep=10, daq=self.daq, port=0)

        self.excControl = LaserControlTTL(
            self.exclaser, '<h3>473<h3>', color=(0, 183, 255))

        self.actlaser.autostart = False
        self.offlaser.autostart = False

        self.controls = (self.actControl, self.offControl, self.excControl)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        self.DigCtrl = DigitalControl(
            controls=(self.actControl, self.offControl, self.excControl))

        grid.addWidget(self.actControl, 0, 0, 4, 1)
        grid.addWidget(self.offControl, 0, 1, 4, 1)
        grid.addWidget(self.excControl, 0, 2, 4, 1)
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


class DigitalControl(QtGui.QFrame):

    def __init__(self, controls, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mW = Q_(1, 'mW')

        title = QtGui.QLabel('<h3>Digital modulation<h3>')
        title.setTextFormat(QtCore.Qt.RichText)
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:12px")
        title.setFixedHeight(20)

        self.controls = controls
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        actPower = str(self.controls[0].laser.power_mod.magnitude)
        self.ActPower = QtGui.QLineEdit(actPower)
        self.ActPower.textChanged.connect(self.updateDigitalPowers)
        offPower = str(self.controls[1].laser.power_mod.magnitude)
        self.OffPower = QtGui.QLineEdit(offPower)
        self.OffPower.textChanged.connect(self.updateDigitalPowers)
        self.ExcPower = QtGui.QLineEdit()
        self.ExcPower.textChanged.connect(self.updateDigitalPowers)
        # Our 473nm laser has a fixed emission power
        self.ExcPower.setEnabled(False)

        self.DigitalControlButton = QtGui.QPushButton('Enable')
        self.DigitalControlButton.setCheckable(True)
        self.DigitalControlButton.clicked.connect(self.GlobalDigitalMod)
        style = "background-color: rgb{}".format((160, 160, 160))
        self.DigitalControlButton.setStyleSheet(style)

        self.updateDigPowersButton = QtGui.QPushButton('Update powers')
        self.updateDigPowersButton.clicked.connect(self.updateDigitalPowers)

        actUnit = QtGui.QLabel('mW')
        actUnit.setFixedWidth(20)
        actModFrame = QtGui.QFrame()
        actModGrid = QtGui.QGridLayout()
        actModFrame.setLayout(actModGrid)
        actModGrid.addWidget(self.ActPower, 0, 0)
        actModGrid.addWidget(actUnit, 0, 1)

        offUnit = QtGui.QLabel('mW')
        offUnit.setFixedWidth(20)
        offModFrame = QtGui.QFrame()
        offModGrid = QtGui.QGridLayout()
        offModFrame.setLayout(offModGrid)
        offModGrid.addWidget(self.OffPower, 0, 0)
        offModGrid.addWidget(offUnit, 0, 1)

        excUnit = QtGui.QLabel('mW')
        excUnit.setFixedWidth(20)
        excModFrame = QtGui.QFrame()
        excModGrid = QtGui.QGridLayout()
        excModFrame.setLayout(excModGrid)
        excModGrid.addWidget(self.ExcPower, 0, 0)
        excModGrid.addWidget(excUnit, 0, 1)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(title, 0, 0)
        grid.addWidget(actModFrame, 1, 0)
        grid.addWidget(offModFrame, 1, 1)
        grid.addWidget(excModFrame, 1, 2)
        grid.addWidget(self.DigitalControlButton, 2, 0, 1, 3)

    def GlobalDigitalMod(self):
        self.digitalPowers = [float(self.ActPower.text()),
                              float(self.OffPower.text())]

        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.controls)):
                try:
                    power = float(self.digitalPowers[i])
                except:
                    power = -1
                if power == -1:
                    self.controls[i].digitalMod(True)
                else:
                    self.controls[i].digitalMod(True, power)
        else:
            for i in np.arange(len(self.controls)):
                self.controls[i].digitalMod(False)

    def updateDigitalPowers(self):
        self.digitalPowers = [float(self.ActPower.text()),
                              float(self.OffPower.text())]
        if self.DigitalControlButton.isChecked():
            for i in np.arange(len(self.digitalPowers)):
                powMag = float(self.digitalPowers[i])
                self.controls[i].laser.power_mod = powMag * self.mW


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
        self.laser.digital_mod = True

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
#        powerFrame.setMinimumHeight(250)
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
#            self.digimodButton.toggled.connect(self.digitalMod)
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

    def digitalMod(self, tof, power=0):
        if tof:
            self.laser.enter_mod_mode()
            self.laser.power_mod = power * self.mW
            print('Entered digital modulation mode with power :', power)
            print('Modulation mode is: ', self.laser.mod_mode)
        else:
            self.laser.digital_mod = False
            self.changeEdit()
            self.laser.query('cp')
            print('Exited digital modulation mode')

    def enableLaser(self):
        self.laser.enabled = True
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW

    def changeSlider(self, value):
        self.laser.power_sp = self.slider.value() * self.mW
        self.setPointEdit.setText(str(self.laser.power_sp.magnitude))

    def changeEdit(self):
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.slider.setValue(self.laser.power_sp.magnitude)


class LaserControlTTL(QtGui.QFrame):

    def __init__(self, laser, name, color, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setStyleSheet("font-size:16px")
        self.name.setFixedHeight(40)

        # ON/OFF button
        self.enableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

        powerFrame = QtGui.QFrame()
        powerFrame.setMinimumHeight(180)
#        powerFrame.setMinimumWidth(100)

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.name, 0, 0, 1, 2)
        self.grid.addWidget(powerFrame, 1, 0, 1, 2)
        self.grid.addWidget(self.enableButton, 8, 0, 1, 2)

#        # Digital modulation
#
#        self.digimodButton = QtGui.QPushButton('Digital modulation')
#        style = "background-color: rgb{}".format((160, 160, 160))
#        self.digimodButton.setStyleSheet(style)
#        self.digimodButton.setCheckable(True)
#           grid.addWidget(self.digimodButton, 6, 0)
#        self.digimodButton.toggled.connect(self.digitalMod)
#        # Initial values
#           self.digimodButton.setChecked(False)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)

    def digitalMod(self, tof):
        if tof:
            self.laser.enabled = False
            self.laser.digital_mod = True
            self.laser.enter_mod_mode()
        else:
            self.laser.digital_mod = False
            if self.enableButton.isChecked():
                self.laser.enabled = True

    def toggleLaser(self):
        if self.enableButton.isChecked():
            try:
                self.laser.enabled = True
            except:
                print('Cannot enable laser when in mod mode, '
                      'will enable when switching back')
        else:
            try:
                self.laser.enabled = False
            except:
                print('Cannot disable laser when in mod mode, '
                      'will disable when switching back')

    def changeEdit(self):
        pass
