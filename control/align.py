# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 15:01:14 2016

@author: testaRES
"""

import numpy as np
import time
import scipy.ndimage as ndi
from matplotlib import pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.ptime as ptime
import pygame

from lantz import Q_

import control.instruments as instruments 
 # , DAQ
import control.guitools as guitools


class AlignWidget(QtGui.QFrame):
    
    def __init__(self, main, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.main = main        
        
        self.ROI = guitools.ROI((50, 50), self.main.vb, (0, 0),
                                handlePos=(1, 0), handleCenter=(0, 1), 
                                color=pg.mkPen(255, 0, 0),
                                scaleSnap=True, translateSnap=True)
                  
        self.ROI.hide()
        self.graph = SumpixelsGraph()
        self.roiButton = QtGui.QPushButton('Show ROI')
        self.roiButton.setCheckable(True)
        self.roiButton.clicked.connect(self.ROItoggle)     
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.graph, 0, 0, 1, 6)
        grid.addWidget(self.roiButton, 1, 0, 1, 1)
        
        self.scansPerS = 10
        self.alignTime = 1000 / self.scansPerS
        self.alignTimer = QtCore.QTimer()
        self.alignTimer.timeout.connect(self.updateValue)
#        self.alignTimer.start(self.alignTime)
    
    def ROItoggle(self):
        if self.roiButton.isChecked() == False:
            self.ROI.hide()
            self.alignTimer.stop()
            self.roiButton.setText('Show ROI')
        else:
            self.ROI.show()
            self.roiButton.setText('Hide ROI')
            self.alignTimer.start(self.alignTime)
            
    def updateValue(self):
        
        if self.main.liveviewButton.isChecked():
            self.selected = self.ROI.getArrayRegion(self.main.latest_image, self.main.img)
            value = np.mean(self.selected)
            self.graph.updateGraph(value)
        else:
            pass
        
    def closeEvent(self, *args, **kwargs):

        self.alignTimer.stop()

        super().closeEvent(*args, **kwargs)
        
        
class SumpixelsGraph(pg.GraphicsWindow):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


        self.setWindowTitle('Sum of area')
        self.setAntialiasing(True)

        self.npoints = 400
        self.data = np.zeros(self.npoints)
        self.ptr = 0


        # Graph without a fixed range
        self.statistics = pg.LabelItem(justify='right')
        self.addItem(self.statistics)
        self.statistics.setText('---')
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabels(bottom=('Time', 's'),
                            left=('Intensity', 'au'))
        self.plot.showGrid(x=True, y=True)
        self.sumCurve = self.plot.plot(pen='y')

        self.time = np.zeros(self.npoints)
        self.startTime = ptime.time()


    def updateGraph(self, value):
        """ Update the data displayed in the graphs
        """

        if self.ptr < self.npoints:
            self.data[self.ptr] = value
            self.time[self.ptr] = ptime.time() - self.startTime
            self.sumCurve.setData(self.time[1:self.ptr + 1],
                                    self.data[1:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = value
            self.time[:-1] = self.time[1:]
            self.time[-1] = ptime.time() - self.startTime

            self.sumCurve.setData(self.time, self.data)

        self.ptr += 1
        
    
        
        