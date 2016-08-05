# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@authors: Federico Barabas, Luciano Masullo
"""

import numpy as np
import os
import time

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget

from lantz import Q_

# tormenta imports
import control.lasercontrol as lasercontrol
#import control.SignalGen as SignalGen
#import control.Scan as Scan
#import control.focus as focus
#import control.molecules_counter as moleculesCounter
#import control.ontime as ontime
#import control.tableWidget as tableWidget
import control.slmWidget as slmWidget
import control.guitools as guitools
import control.camera_image_manager as camera_image_manager
import control.webcamWidget as webcamWidget
import control.oscilloscope as oscilloscope
#from control import libnidaqmx
import nidaqmx
import control.Scan_self_GUI as scanWidget

#Widget to control image or sequence recording. Recording only possible when liveview active.
#StartRecording called when "Rec" presset. Creates recording thread with RecWorker, recording is then 
#done in this seperate thread.

datapath=u"C:\\Users\\aurelien.barbotin\Documents\Data\DefaultDataFolder"



class FileWarning(QtGui.QMessageBox):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   
        
#class PulseWidget(QtGui.QFrame):
        
        
class TemperatureStabilizer(QtCore.QObject):
        pass
#    def __init__(self, main, *args, **kwargs):
#
#        super().__init__(*args, **kwargs)
#        self.main = main
#        self.setPoint = main.tempSetPoint
#        self.main.andor.temperature_setpoint = self.setPoint
#        self.stableText = 'Temperature has stabilized at set point.'
#
#    def start(self):
#        self.main.andor.cooler_on = True
#        self.timer = QtCore.QTimer()
#        self.timer.timeout.connect(self.update)
#        self.timer.start(10000)
#        self.update()
#
#    def stop(self):
#        self.timer.stop()
#
#    def update(self):
#        tempStatus = self.main.andor.temperature_status
#        self.main.tempStatus.setText(tempStatus)
#        temperature = np.round(self.main.andor.temperature, 1)
#        self.main.temp.setText('{} ºC'.format(temperature.magnitude))
#
#        if tempStatus != self.stableText:
#            threshold = Q_(0.8 * self.setPoint.magnitude, 'degC')
#            if temperature <= threshold or self.main.andor.mock:
#                self.main.liveviewButton.setEnabled(True)
#                self.main.liveviewAction.setEnabled(True)
#
#        else:
#            self.timer.stop()


class TempestaGUI(QtGui.QMainWindow):
    """Main GUI class. This class calls other modules in the control folder"""
    
    liveviewStarts = QtCore.pyqtSignal()
    liveviewEnds = QtCore.pyqtSignal()

    def __init__(self, bluelaser, violetlaser, uvlaser, scanZ, daq, orcaflash,slm,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orcaflash = orcaflash
        self.bluelaser = bluelaser
        self.violetlaser = violetlaser
        self.uvlaser = uvlaser
        self.scanZ = scanZ
        self.daq = daq
        self.nidaq = nidaqmx.Device('Dev1')
        self.slm=slm
        self.filewarning = FileWarning()

        self.s = Q_(1, 's')
        self.lastTime = time.clock()
        self.fps = None

        # Actions in menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        self.savePresetAction = QtGui.QAction('Save configuration...', self)
        self.savePresetAction.setShortcut('Ctrl+S')
        self.savePresetAction.setStatusTip('Save camera & recording settings')
        savePresetFunction = lambda: guitools.savePreset(self)
        self.savePresetAction.triggered.connect(savePresetFunction)
        fileMenu.addAction(self.savePresetAction)
        fileMenu.addSeparator()


        self.exportTiffAction = QtGui.QAction('Export HDF5 to Tiff...', self)
        self.exportTiffAction.setShortcut('Ctrl+E')
        self.exportTiffAction.setStatusTip('Export HDF5 file to Tiff format')
        self.exportTiffAction.triggered.connect(guitools.TiffConverterThread)
        fileMenu.addAction(self.exportTiffAction)

        self.exportlastAction = QtGui.QAction('Export last recording to Tiff',
                                              self)
        self.exportlastAction.setEnabled(False)
        self.exportlastAction.setShortcut('Ctrl+L')
        self.exportlastAction.setStatusTip('Export last recording to Tiff ' +
                                           'format')
        fileMenu.addAction(self.exportlastAction)
        fileMenu.addSeparator()

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.closeAllWindows)
        fileMenu.addAction(exitAction)

        self.presetsMenu = QtGui.QComboBox()
        self.presetDir = datapath
        if not(os.path.isdir(self.presetDir)):
            self.presetDir = os.path.join(os.getcwd(), 'control/Presets')
        for preset in os.listdir(self.presetDir):
            self.presetsMenu.addItem(preset)
        self.loadPresetButton = QtGui.QPushButton('Load preset')
        loadPresetFunction = lambda: guitools.loadPreset(self)
        self.loadPresetButton.pressed.connect(loadPresetFunction)        
        
#        # Dock widget
        dockArea = DockArea()

        # Console widget
        consoleDock = Dock("Console", size=(100, 100))
        console = ConsoleWidget(namespace={'pg': pg, 'np': np})
        consoleDock.addWidget(console)
        dockArea.addDock(consoleDock)

        #Oscilloscope
#        oscilloDock = Dock("Oscilloscope")
#        self.oscilloscopeWidget = oscilloscope.Oscilloscope()
#        oscilloDock.addWidget(self.oscilloscopeWidget)
#        dockArea.addDock(oscilloDock,"above",consoleDock)
        
        # Emission filters table widget
#        wheelDock = Dock("Emission filters", size=(20, 20))
#        tabWidget = tableWidget.TableWidget(sortable=False)
#        wheelDock.addWidget(tabWidget)
#        dockArea.addDock(wheelDock, 'top', consoleDock)

#         On time widget
#        ontimeDock = Dock('On time histogram', size=(1, 1))
#        self.ontimeWidget = ontime.OntimeWidget()
#        ontimeDock.addWidget(self.ontimeWidget)
#        dockArea.addDock(ontimeDock, 'above', wheelDock)

 #        Molecule counting widget
#        moleculesDock = Dock('Molecule counting', size=(1, 1))
#        self.moleculeWidget = moleculesCounter.MoleculeWidget()
#        moleculesDock.addWidget(self.moleculeWidget)
#        dockArea.addDock(moleculesDock, 'above', ontimeDock)

#         Focus lock widget
#        focusDock = Dock("Focus Control", size=(1, 1))
#        self.focusWidget = focus.FocusWidget(scanZ, self.recWidget)
#        focusDock.addWidget(self.focusWidget)
#        dockArea.addDock(focusDock, 'top', consoleDock)

        #Laser Widget
        laserDock = Dock("Laser Control", size=(50, 30))
        self.lasers = (bluelaser, violetlaser, uvlaser)
        self.laserWidgets = lasercontrol.LaserWidget(self.lasers, self.daq)
        laserDock.addWidget(self.laserWidgets)
        dockArea.addDock(laserDock,"top",consoleDock)
        
        #SLM widget
        slmDock = Dock("SLM",size=(300,100))
        self.slmWidget = slmWidget.slmWidget(self.slm)
        slmDock.addWidget(self.slmWidget)
        dockArea.addDock(slmDock,'above',laserDock)
        

        # Signal generation widget
#        signalDock = Dock('Signal Generator')
#        self.signalWidget = SignalGen.SigGenWidget(self.nidaq)
#        signalDock.addWidget(self.signalWidget)
#        dockArea.addDock(signalDock, 'above', laserDock)
        
##        Scan Widget
#        scanDock = Dock('Scan')
#        self.scanWidget = Scan.ScanWidget(self.nidaq)
#        scanDock.addWidget(self.scanWidget)
#        dockArea.addDock(scanDock, 'above', signalDock)
        
         #Image Manager
        imgManager=camera_image_manager.ImageManager(self.orcaflash,self)
        self.cameraWidget = imgManager.cameraWidget
        self.viewCtrl = imgManager.viewCtrl
        self.recWidget = imgManager.recWidget
        self.imageWidget = imgManager.imageWidget
        
        self.setWindowTitle('Tempesta 1.01')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)
        
        #Scan Widget
        self.scanxyWidget=scanWidget.ScanWidget(self.nidaq)

        #APD display ma gueule
        self.apdDisplay_splay=self.scanxyWidget.display

        #Webcam
        self.webcamWidget=webcamWidget.WebcamManager(self)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(0, 100)
#        layout.setColumnMinimumWidth(1, 350)
        layout.setColumnMinimumWidth(2, 150)
        layout.setColumnMinimumWidth(3, 200)
        layout.setRowMinimumHeight(0, 350)
        layout.setRowMinimumHeight(1, 350)
        layout.setRowMinimumHeight(2, 350)
        layout.setRowMinimumHeight(3, 30)
#        layout.addWidget(self.presetsMenu, 0, 0)
#        layout.addWidget(self.loadPresetButton, 0, 1)
        
#        layout.addWidget(self.cameraWidget, 1, 0, 2, 2)
#        layout.addWidget(self.viewCtrl, 3, 0, 1, 2)
#        layout.addWidget(self.recWidget, 4, 0, 1, 2)
#        layout.addWidget(self.imageWidget, 0, 2, 5, 1)
        layout.addWidget(dockArea, 0, 3, 5, 1)
        layout.addWidget(self.scanxyWidget,0, 2, 5, 1)
        layout.addWidget(self.apdDisplay_splay,0,0,1,1)
        layout.addWidget(self.webcamWidget,1,0,1,1)
        
        layout.setRowMinimumHeight(2, 40)
#        layout.setColumnMinimumWidth(2, 1000)

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
#        self.viewtimer.stop()
#        self.stabilizer.timer.stop()
#        self.stabilizerThread.terminate()
        try:
            self.lvthread.terminate()
        except:
            pass
            
        # Turn off camera, close shutter and flipper
#        if self.andor.status != 'Camera is idle, waiting for instructions.':
#            self.andor.abort_acquisition()
#        self.andor.shutter(0, 2, 0, 0, 0)
        self.orcaflash.shutdown()
        self.daq.flipper = True
#        if self.signalWidget.running:
#            self.signalWidget.StartStop()

        self.nidaq.reset()        
        self.laserWidgets.closeEvent(*args, **kwargs)
#        self.focusWidget.closeEvent(*args, **kwargs)
        self.slmWidget.closeEvent(*args, **kwargs)
#        self.signalWidget.closeEvent(*args, **kwargs)

        super().closeEvent(*args, **kwargs)
