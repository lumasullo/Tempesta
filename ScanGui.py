# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:54:52 2016

@author: testaRES
"""
 
from pyqtgraph.Qt import QtGui
from scanner_only import Scan_self_GUI
# from control import control
from control import libnidaqmx


def main():

    app = QtGui.QApplication([])

    win = Scan_self_GUI.ScanWidget(libnidaqmx.Device('Dev1'))
#    win = control.TormentaGUI(bluelaser, violetlaser, uvlaser,
#                              scanZ, daq, orcaflash)
    win.show()

    app.exec_()

if __name__ == '__main__':
    main()
 