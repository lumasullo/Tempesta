# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import time
from PyQt4 import QtGui, QtCore
from lantz import Q_
from control import libnidaqmx
import numpy as np
from multiprocessing import Process

def mWtomV(x):
    
    # p are the coefficients from Ti:Sa calibration at 08/2016
    p = [0.0000000038094, -0.00000444662, 0.0019, -0.36947, 42.6684, 196]
    y = np.polyval(p,x)
    
    return y

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
        QtCore.QTimer.singleShot(0.01, self.update)


class LaserWidget(QtGui.QFrame):

    def __init__(self, lasers, daq, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.bluelaser, self.bluelaser2, self.greenlaser, self.violetlaser, self.uvlaser = lasers
        self.mW = Q_(1, 'mW')
        self.daq = daq

        self.blueControl = LaserControl(self.bluelaser,
                                       '488',
                                        color=(0, 247, 255), prange=(0, 200),
                                        tickInterval=100, singleStep=10,
                                        daq=self.daq, port=0)
                                        
        self.blue2Control = LaserControl(self.bluelaser2,
                                       '488(2)',
                                        color=(0, 247, 255), prange=(0, 200),
                                        tickInterval=100, singleStep=10,
                                        daq=self.daq, port=0)
                                        
        self.greenControl = LaserControl(self.greenlaser,
                                       '561',
                                        color=(198,255, 0), prange=(0, 200),
                                        tickInterval=100, singleStep=10,
                                        daq=self.daq, port=0, modulable = False)

        self.violetControl = LaserControl(self.violetlaser,
                                         '405',
                                         color=(73, 0, 188), prange=(0, 200),
                                         tickInterval=5, singleStep=0.1)

        self.uvControl = LaserControl(self.uvlaser,
                                         '355',
                                         color=(97, 0, 97), prange=(0, 20),
                                         tickInterval=10, singleStep=1,
                                         daq=self.daq, port=1, modulable=False)
                                         
        self.tisacontrol = TiSaControl('<h3>TiSa<h3>',
                                        color=(200, 0, 0), prange=(0, 10000),
                                        tickInterval=5, singleStep=0.01, 
                                        modulable = False)

        self.controls = (self.blueControl, self.blue2Control, self.greenControl, self.violetControl, self.uvControl)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        self.DigCtrl = DigitalControl()
        
        grid.addWidget(self.blueControl, 0, 0, 4, 1)
        grid.addWidget(self.blue2Control, 0, 1, 4, 1)
        grid.addWidget(self.violetControl, 0, 2, 4, 1)
        grid.addWidget(self.greenControl, 0, 3, 4, 1)        
#        grid.addWidget(self.uvControl, 0, 4, 4, 1)
        grid.addWidget(self.tisacontrol, 4, 0, 1, 1)
        grid.addWidget(self.DigCtrl, 4, 1, 2, 3)

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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.blueOffLabel = QtGui.QLabel('488 (OFF pattern)')
        self.blueOffPower = QtGui.QLineEdit('0')
        self.blueReadoutLabel = QtGui.QLabel('488 readout')
        self.blueReadoutPower = QtGui.QLineEdit('0')
        self.violetOnLabel = QtGui.QLabel('405 (ON pattern)')
        self.violetOnPower = QtGui.QLineEdit('0') 
        
        self.blueOffLabel.setStyleSheet("background-color: rgb{}".format((0, 247, 255)))
        self.blueReadoutLabel.setStyleSheet("background-color: rgb{}".format((0, 247, 255)))
        self.violetOnLabel.setStyleSheet("background-color: rgb{}".format((73, 0, 188)))
        
        self.DigitalControlButton = QtGui.QPushButton('Global digital modulation')        
        self.DigitalControlButton.setCheckable(True)
        style = "background-color: rgb{}".format((160, 160, 160))
        self.DigitalControlButton.setStyleSheet(style)
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.blueReadoutLabel, 0, 0)
        grid.addWidget(self.blueReadoutPower, 1, 0)
        grid.addWidget(self.blueOffLabel, 0, 1)
        grid.addWidget(self.blueOffPower, 1, 1)
        grid.addWidget(self.violetOnLabel, 0, 2)
        grid.addWidget(self.violetOnPower, 1, 2)
        grid.addWidget(self.DigitalControlButton,2,0)


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, modulable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port
        self.laser.digital_mod = False

        self.name = QtGui.QLabel('</h3>{}</h3>'.format(name))
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
                self.digimodButton = QtGui.QPushButton('{} digital modulation'.format(name))
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



class TiSaControl(QtGui.QFrame):

    def __init__(self, name, color, prange, tickInterval, singleStep,
                 invert=True, modulable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.mV = Q_(1, 'mV')
        self.mW = Q_(1, 'mW')
        init_voltage = 0
        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        style = "background-color: rgb{}".format(color)
        self.powerIndicator = QtGui.QLineEdit(str(init_voltage))
        self.powerIndicator.setReadOnly(True)
        self.powerIndicator.setFixedWidth(100)
        self.setPointEdit = QtGui.QLineEdit(str(init_voltage))
        self.setPointEdit.setFixedWidth(100)
        self.powerIndicator = QtGui.QLineEdit()
        self.powerIndicator.setStyleSheet("background-color: rgb(240,240,240);")
        self.powerIndicator.setReadOnly(True)
        self.powerIndicator.setFixedWidth(100)
        self.name.setStyleSheet(style)
        self.calibCheck = QtGui.QCheckBox('Calibration mW/mV')

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 0, 0)
        grid.addWidget(self.powerIndicator, 1, 0)
        grid.addWidget(self.setPointEdit, 2, 0)
        grid.addWidget(self.calibCheck, 3, 0)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

#        self.aotask_tisa = libnidaqmx.AnalogOutputTask('aotask')
#        aochannel = 3
#        self.aotask_tisa.create_voltage_channel('Dev1/ao%s'%aochannel, min_val = 0,  max_val = 10)
#        
#        self.aotask_tisa.start()

    def changeEdit(self):
        calibrated = self.calibCheck.isChecked()
        userInput = float(self.setPointEdit.text())
        if calibrated == False:
            new_value = userInput * self.mV
        if calibrated == True:
            new_value = mWtomV(userInput) * self.mV
            print(mWtomV(float(self.setPointEdit.text())))
        aochannel = 3
        self.p = Process(target=change_V_in_process, args=(new_value.magnitude, aochannel))
        self.p.start()
        self.p.join()
        if calibrated == False:
            self.powerIndicator.setText('{:~}'.format(new_value))
        if calibrated == True:
            self.powerIndicator.setText('{:~}'.format(userInput * self.mW))
        
    def change_voltage(self, new_value):

#        data = (new_value/1000)*np.ones(10) # new_value in millivolts
#        self.aotask_tisa.write(data, layout = 'group_by_channel') 
#        self.powerIndicator.setText('{}'.format(self.setPointEdit.text()))
        for i in range(1000, 100000):
            print(i)
            data = (i/1000)*np.ones(2000) # new_value in millivolts
            self.aotask_tisa.write(data, layout = 'group_by_channel') 
    #        self.powerIndicator.setText('{}'.format(self.setPointEdit.text()))

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        
def change_V_in_process(value, aochannel):
    
    aotask_tisa = libnidaqmx.AnalogOutputTask('aotask')
    
    aotask_tisa.create_voltage_channel('Dev1/ao%s'%aochannel, min_val = 0,  max_val = 10)
    
    aotask_tisa.start()
    
    data = (value/1000)*np.ones(2) # new_value in millivolts
    aotask_tisa.write(data, layout = 'group_by_channel')
    
#    for a in range(1,10):        
#        for i in range(1000, 10000):
#            print(i)
#            data = (i/1000)*np.ones(2) # new_value in millivolts
#            aotask_tisa.write(data, layout = 'group_by_channel')
#        
    aotask_tisa.stop()
    
    
    
    
    
    
    
    
    
    
    
    
    