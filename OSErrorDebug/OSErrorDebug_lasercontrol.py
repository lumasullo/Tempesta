# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import time
from PyQt4 import QtGui, QtCore
from lantz import Q_
import libnidaqmx
import numpy as np
import instruments


class UpdatePowers(QtCore.QObject):

    def __init__(self, laserwidget, *args, **kwargs):

        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.widget = laserwidget

    def update(self):
        bluepower = '{:~}'.format(self.widget.bluelaser.power)
        violetpower = '{:~}'.format(self.widget.violetlaser.power)
        uvpower = '{:~}'.format(self.widget.uvlaser.power)
        self.widget.blueControl.powerIndicator.setText(bluepower)
        self.widget.violetControl.powerIndicator.setText(violetpower)
        self.widget.uvControl.powerIndicator.setText(uvpower)
        time.sleep(1)
        QtCore.QTimer.singleShot(1, self.update)


class LaserWidget(QtCore.QObject):

    def __init__(self, lasers, daq, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.bluelaser, self.violetlaser, self.uvlaser = lasers
        self.mW = Q_(1, 'mW')
        self.daq = daq

        self.blueControl = LaserControl(self.bluelaser,
                                       '<h3>Cobolt 488nm</h3>',
                                        color=(0, 247, 255), prange=(0, 200),
                                        tickInterval=100, singleStep=10,
                                        daq=self.daq, port=0)

        self.violetControl = LaserControl(self.violetlaser,
                                         '<h3>Cobolt 405nm</h3>',
                                         color=(73, 0, 188), prange=(0, 200),
                                         tickInterval=5, singleStep=0.1)

        self.uvControl = LaserControl(self.uvlaser,
                                         '<h3>Cobolt 355nm</h3>',
                                         color=(97, 0, 97), prange=(0, 20),
                                         tickInterval=10, singleStep=1,
                                         daq=self.daq, port=1, modulable=False)
                                         
        self.tisacontrol = TiSaControl('<h3>TiSa</h3>', color=(200, 0, 0), prange=(0, 10000), modulable = False)

        self.controls = (self.blueControl, self.violetControl, self.uvControl)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.blueControl, 0, 0, 4, 1)
        grid.addWidget(self.violetControl, 0, 1, 4, 1)
        grid.addWidget(self.uvControl, 0, 2, 4, 1)
        grid.addWidget(self.tisacontrol, 0, 3, 1, 1)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)

    def closeEvent(self, *args, **kwargs):
        self.closeShutters()
        self.updateThread.terminate()
        super().closeEvent(*args, **kwargs)


class LaserControl(QtCore.QObject):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, modulable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port
        self.laser.digital_mod = False

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.powerIndicator = QtGui.QLineEdit('{:~}'.format(self.laser.power))
        self.powerIndicator.setReadOnly(True)
        self.powerIndicator.setFixedWidth(100)
        self.powerIndicator.setStyleSheet("background-color: rgb(240,240,240);")
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(100)
        self.enableButton = QtGui.QPushButton('ON')
        self.enableButton.setFixedWidth(100)
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        self.name.setStyleSheet(style)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

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

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 0, 0)
        grid.addWidget(self.powerIndicator, 3, 0)
        grid.addWidget(self.setPointEdit, 4, 0)
        grid.addWidget(self.enableButton, 5, 0)
        grid.addWidget(self.maxpower, 1, 1)
        grid.addWidget(self.slider, 2, 1, 5, 1)
        grid.addWidget(self.minpower, 7, 1)
        grid.setRowMinimumHeight(2, 60)
        grid.setRowMinimumHeight(6, 60)

        # Digital modulation
        if modulable == True:
                self.digimodButton = QtGui.QPushButton('Digital modulation')
                style = "background-color: rgb{}".format((160, 160, 160))
                self.digimodButton.setStyleSheet(style)
                self.digimodButton.setCheckable(True)
                grid.addWidget(self.digimodButton, 6, 0)
                self.digimodButton.toggled.connect(self.digitalMod)
                # Initial values
#                self.digimodButton.setChecked(False)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

    def shutterAction(self, state):
        self.daq.digital_IO[self.port] = self.states[state]

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

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)

class TiSaControl(QtCore.QObject):

    def __init__(self, name, color, prange, invert=True, modulable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
#        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.mV = Q_(1, 'mV')
        init_voltage = 0
#        self.name = QtGui.QLabel(name)
#        self.name.setTextFormat(QtCore.Qt.RichText)
#        self.name.setAlignment(QtCore.Qt.AlignCenter)
#        style = "background-color: rgb{}".format(color)
#        self.powerIndicator = QtGui.QLineEdit(str(init_voltage))
#        self.powerIndicator.setReadOnly(True)
#        self.powerIndicator.setFixedWidth(100)
#        self.setPointEdit = QtGui.QLineEdit(str(init_voltage))
#        self.setPointEdit.setFixedWidth(100)
#        self.powerIndicator = QtGui.QLineEdit()
#        self.powerIndicator.setStyleSheet("background-color: rgb(240,240,240);")
#        self.powerIndicator.setReadOnly(True)
#        self.powerIndicator.setFixedWidth(100)
#        self.name.setStyleSheet(style)

#        grid = QtGui.QGridLayout()
#        self.setLayout(grid)
#        grid.addWidget(self.name, 0, 0)
#        grid.addWidget(self.powerIndicator, 1, 0)
#        grid.addWidget(self.setPointEdit, 2, 0)
#        self.setPointEdit.editingFinished.connect(self.changeEdit)

        self.aotask_tisa = libnidaqmx.AnalogOutputTask('aotask')
        aochannel = 3
        self.aotask_tisa.create_voltage_channel('Dev1/ao%s'%aochannel,
                                                min_val = 0,  max_val = 10)
        self.aotask_tisa.start()

    def changeEdit(self):

        new_value = float(self.setPointEdit.text()) * self.mV
        self.change_voltage(new_value.magnitude)

    def change_voltage(self, new_value):
        for a in range(1,100):
            for i in range(1000, 10000):
                print(a)
                data = (i/1000)*np.ones(10) # new_value in millivolts
                self.aotask_tisa.write(data, layout = 'group_by_channel') 
        #        self.powerIndicator.setText('{}'.format(self.setPointEdit.text()))

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        self.aotask_tisa.stop()
    