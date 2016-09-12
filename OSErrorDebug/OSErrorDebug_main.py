# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:54:52 2016

@author: testaRES
"""

from pyqtgraph.Qt import QtGui
import OSErrorDebug_lasercontrol_mainWindow

def main():
    
    app = QtGui.QApplication([])


    win = OSErrorDebug_lasercontrol_mainWindow.LaserWidget()
    win.show()

    app.exec_()


main()
