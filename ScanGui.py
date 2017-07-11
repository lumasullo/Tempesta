# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:54:52 2016

@author: testaRES
"""

from pyqtgraph.Qt import QtGui
from scanner_only import Scan_self_GUI
import nidaqmx


def main():

    app = QtGui.QApplication([])

    system = nidaqmx.system.System.local()
    win = Scan_self_GUI.ScanWidget(system.devices['Dev1'])
    win.show()

    app.exec_()

if __name__ == '__main__':
    main()
