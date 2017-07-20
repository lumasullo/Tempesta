# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:54:52 2016

@author: testaRES
"""

from pyqtgraph.Qt import QtGui
from control import scanner
import nidaqmx


def main():

    app = QtGui.QApplication([])

    system = nidaqmx.system.System.local()
    widget = scanner.ScanWidget(system.devices['Dev1'], None)
    win = QtGui.QMainWindow()
    win.setCentralWidget(widget)
    win.show()

    app.exec_()

if __name__ == '__main__':
    main()
