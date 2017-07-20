# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:54:52 2016

@author: Barabas, Bodén, Masullo
"""

from pyqtgraph.Qt import QtGui
from control import scanner
import nidaqmx


def main():

    app = QtGui.QApplication([])

    # NI-DAQ channels configuration
    system = nidaqmx.system.System.local()
    DO = {'405': 0, '473': 1, '488': 2, 'CAM': 3}
    AO = {'x': 0, 'y': 1, 'z': 2}

    widget = scanner.ScanWidget(system.devices['Dev1'], [DO, AO])
    win = QtGui.QMainWindow()
    win.setCentralWidget(widget)
    win.show()

    app.exec_()

if __name__ == '__main__':
    main()
