# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:05:25 2016

@author: testaRES
"""

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


class LaserWidget(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.mW = Q_(1, 'mW')
                                         
        self.tisacontrol = TiSaControl('<h3>TiSa</h3>',
                                        color=(200, 0, 0), prange=(0, 10000),
                                        tickInterval=5, singleStep=0.01, 
                                        modulable = False)


#        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)
        
        grid.addWidget(self.tisacontrol, 0, 3, 1, 1)


    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)


class TiSaControl(QtGui.QFrame):

    def __init__(self, name, color, prange, tickInterval, singleStep,
                 invert=True, modulable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.mV = Q_(1, 'mV')
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

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 0, 0)
        grid.addWidget(self.powerIndicator, 1, 0)
        grid.addWidget(self.setPointEdit, 2, 0)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

        self.aotask_tisa = libnidaqmx.AnalogOutputTask('aotask')
        aochannel = 3
        self.aotask_tisa.create_voltage_channel('Dev1/ao%s'%aochannel,
                                                min_val = 0,  max_val = 10)
        self.aotask_tisa.start()

    def changeEdit(self):

        new_value = float(self.setPointEdit.text()) * self.mV
        self.change_voltage(new_value.magnitude)

    def change_voltage(self, new_value):

        for i in range(1000, 10000):
            print(i)
            data = (i/1000)*np.ones(2) # new_value in millivolts
            self.aotask_tisa.write(data, layout = 'group_by_channel') 

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        print('Closing time')
#        self.aotask_tisa.stop()
    