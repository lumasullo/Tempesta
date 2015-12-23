# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import time
from PyQt4 import QtGui, QtCore
from lantz import Q_


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


class LaserWidget(QtGui.QFrame):

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
                                         daq=self.daq, port=1)

        self.controls = (self.blueControl, self.violetControl, self.uvControl)

#        self.findTirfButton = QtGui.QPushButton('Find TIRF (no anda)')
#        self.setEpiButton = QtGui.QPushButton('Set EPI (no anda)')
#        self.tirfButton = QtGui.QPushButton('TIRF (no anda)')
#        self.tirfButton.setCheckable(True)
#        self.epiButton = QtGui.QPushButton('EPI (no anda)')
#        self.epiButton.setCheckable(True)
#        self.stagePosLabel = QtGui.QLabel('0 mm')
#        self.stagePosLabel.setSizePolicy(QtGui.QSizePolicy.Preferred,
#                                         QtGui.QSizePolicy.Expanding)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.blueControl, 0, 0)
        grid.addWidget(self.violetControl, 0, 1)
        grid.addWidget(self.uvControl, 0, 2)
#        grid.addWidget(self.findTirfButton, 1, 0)
#        grid.addWidget(self.setEpiButton, 2, 0)
#        grid.addWidget(self.tirfButton, 1, 1)
#        grid.addWidget(self.epiButton, 2, 1)
#        grid.addWidget(self.stagePosLabel, 1, 2, 2, 1)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)

    def closeShutters(self):
        for control in self.controls:
            if control.port is not None:
                control.shutterBox.setChecked(False)

    def closeEvent(self, *args, **kwargs):
        self.closeShutters()
        self.updateThread.terminate()
        super().closeEvent(*args, **kwargs)


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)

        self.powerIndicator = QtGui.QLineEdit('{:~}'.format(self.laser.power))
        self.powerIndicator.setReadOnly(True)
        self.powerIndicator.setFixedWidth(100)
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(100)
        self.enableButton = QtGui.QPushButton('ON')
        self.enableButton.setFixedWidth(100)
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
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

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

        # Shutter port
        if self.port is not None:
            self.shutterBox = QtGui.QCheckBox('Shutter open')
#            grid.addWidget(self.shutterBox, 6, 0)
            self.shutterBox.stateChanged.connect(self.shutterAction)

            if invert:
                self.daq.digital_IO[self.port] = True
                self.states = {2: False, 0: True}
            else:
                self.daq.digital_IO[self.port] = False
                self.states = {2: True, 0: False}

    def shutterAction(self, state):
        self.daq.digital_IO[self.port] = self.states[state]

    def toggleLaser(self):
        if self.enableButton.isChecked():
            self.laser.enabled = True
        else:
            self.laser.enabled = False

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


if __name__ == '__main__':

    app = QtGui.QApplication([])

    import tormenta.control.instruments as instruments
    CoboltDriver = 'cobolt.cobolt0601.Cobolt0601'
    
#TO DO: check COM ports are right
    
    with instruments.Laser(CoboltDriver, 'COM7') as bluelaser, \
            instruments.Laser(CoboltDriver, 'COM11') as violetlaser, \
            instruments.Laser(CoboltDriver, 'COM5') as uvlaser, \
            instruments.DAQ() as daq:

        print(bluelaser.idn, violetlaser.idn, uvlaser.idn, daq.idn)
        win = LaserWidget((bluelaser, violetlaser, uvlaser), daq)
        win.show()

        app.exec_()
