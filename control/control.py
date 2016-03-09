# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@authors: Federico Barabas, Luciano Masullo
"""

import subprocess
import sys
import numpy as np
import os
import datetime
import time
import re
import ctypes
import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget

from tkinter import Tk, filedialog, messagebox
import h5py as hdf
import tifffile as tiff     # http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd
from lantz import Q_

# tormenta imports
import control.lasercontrol as lasercontrol
import control.SignalGen as SignalGen
import control.focus as focus
import control.molecules_counter as moleculesCounter
import control.ontime as ontime
import control.guitools as guitools

#Widget to control image or sequence recording. Recording only possible when liveview active.
#StartRecording called when "Rec" presset. Creates recording thread with RecWorker, recording is then 
#done in this seperate thread.

class RecordingWidget(QtGui.QFrame):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.dataname = 'data'      # In case I need a QLineEdit for this
        startdir = r'C:\Users\TestaRES\Documents\Data\DefaultDataFolder\%s'
        newfolderpath =  startdir % time.strftime('%Y-%m-%d')
        if not os.path.exists(newfolderpath):
            os.mkdir(newfolderpath)
            
        self.initialDir = newfolderpath
        
        self.filesizewar = QtGui.QMessageBox()
        self.filesizewar.setText("File size is very big!")
        self.filesizewar.setInformativeText("Are you sure you want to continue?")
        self.filesizewar.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        
        # Title
        recTitle = QtGui.QLabel('<h2><strong>Recording</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        
        # Folder and filename fields
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        openFolderButton = QtGui.QPushButton('Open')
        openFolderButton.clicked.connect(self.openFolder)
        loadFolderButton = QtGui.QPushButton('Load...')
        loadFolderButton.clicked.connect(self.loadFolder)
        self.specifyfile = QtGui.QCheckBox('Specify file name')
        self.specifyfile.clicked.connect(self.specFile)
        self.filenameEdit = QtGui.QLineEdit('Current_time')

        # Snap and recording buttons
        self.snapTIFFButton = QtGui.QPushButton('Snap')
        self.snapTIFFButton.setStyleSheet("font-size:16px")
        self.snapTIFFButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.snapTIFFButton.clicked.connect(self.snapTIFF)
        self.recButton = QtGui.QPushButton('REC')
        self.recButton.setStyleSheet("font-size:16px")
        self.recButton.setCheckable(True)
        self.recButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                     QtGui.QSizePolicy.Expanding)
        self.recButton.clicked.connect(self.startRecording)

        # Number of frames and measurement timing
        self.specifyFrames = QtGui.QRadioButton('Nr of frames')
        self.specifyFrames.clicked.connect(self.specFrames)
        self.specifyTime = QtGui.QRadioButton('Time to rec (sec)')
        self.specifyTime.clicked.connect(self.specTime) 
        self.timeToRec = QtGui.QLineEdit('1')
        self.timeToRec.setFixedWidth(45)
        self.timeToRec.textChanged.connect(self.filesizeupdate)
        self.currentTime = QtGui.QLabel('0 /')
        self.currentTime.setAlignment((QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter))
        self.currentFrame = QtGui.QLabel('0 /')
        self.currentFrame.setAlignment((QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter))
        self.currentFrame.setFixedWidth(45)
        self.numExpositionsEdit = QtGui.QLineEdit('100')
        self.numExpositionsEdit.setFixedWidth(45)
        self.tRemaining = QtGui.QLabel()
        self.tRemaining.setAlignment((QtCore.Qt.AlignCenter |
                                      QtCore.Qt.AlignVCenter))
        self.numExpositionsEdit.textChanged.connect(self.filesizeupdate)
#        self.updateRemaining()

        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setTextVisible(False)
        
        self.filesizeBar = QtGui.QProgressBar()
        self.filesizeBar.setTextVisible(False)
        self.filesizeBar.setRange(0, 2000000000)

        # Layout
        buttonWidget = QtGui.QWidget()
        buttonGrid = QtGui.QGridLayout()
        buttonWidget.setLayout(buttonGrid)
        buttonGrid.addWidget(self.snapTIFFButton, 0, 0)
#        buttonGrid.addWidget(self.snapHDFButton, 0, 1)
        buttonWidget.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                   QtGui.QSizePolicy.Expanding)
        buttonGrid.addWidget(self.recButton, 0, 2)

        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)

# Graphically adding the labels and fields etc to the gui. Four numbers specify row, column, rowspan
# and columnspan.
        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Folder'), 2, 0)
        recGrid.addWidget(loadFolderButton, 1, 5)
        recGrid.addWidget(openFolderButton, 1, 4)
        recGrid.addWidget(self.folderEdit, 2, 1, 1, 5)
        recGrid.addWidget(self.specifyfile, 3, 0, 1, 5)
        recGrid.addWidget(self.filenameEdit, 3, 2, 1, 4)
        recGrid.addWidget(self.specifyFrames, 4, 0, 1, 5)
        recGrid.addWidget(self.currentFrame, 4, 1)
        recGrid.addWidget(self.numExpositionsEdit, 4, 2)
        recGrid.addWidget(QtGui.QLabel('File size'), 4, 3, 1, 2)
        recGrid.addWidget(self.filesizeBar, 4, 4, 1, 2)
        recGrid.addWidget(self.specifyTime, 5, 0, 1, 5)
        recGrid.addWidget(self.currentTime, 5, 1)
        recGrid.addWidget(self.timeToRec, 5, 2)
        recGrid.addWidget(self.tRemaining, 5, 4, 1, 2)
        recGrid.addWidget(self.progressBar, 5, 4, 1, 2)
        recGrid.addWidget(self.tRemaining, 4, 3, 1, 2)
        recGrid.addWidget(buttonWidget, 6, 0, 1, 0)

        recGrid.setColumnMinimumWidth(0, 70)
        recGrid.setRowMinimumHeight(6, 40)

# Initial condition of fields and checkboxes.
        self.writable = True
        self.readyToRecord = False
        self.filenameEdit.setEnabled(False)
        self.specTime()
        self.filesizeupdate()

    @property
    def readyToRecord(self):
        return self._readyToRecord

    @readyToRecord.setter
    def readyToRecord(self, value):
        self.snapTIFFButton.setEnabled(value)
#        self.snapHDFButton.setEnabled(value)
        self.recButton.setEnabled(value)
        self._readyToRecord = value

    @property
    def writable(self):
        return self._writable

# Setter for the writable property. If Nr of frame is checked only the frames field is
# set active and vice versa.

    @writable.setter
    def writable(self, value):
        if value:
            if self.specifyFrames.isChecked():
                self.specFrames()
            else:
                self.specTime()
        else:
            self.numExpositionsEdit.setEnabled(False)
            self.timeToRec.setEnabled(False)
#        self.folderEdit.setEnabled(value)
#        self.filenameEdit.setEnabled(value)
        self._writable = value

    def specFile(self):
        
        if self.specifyfile.checkState():
            self.filenameEdit.setEnabled(True)
            self.filenameEdit.setText('Filename')
        else:
            self.filenameEdit.setEnabled(False)
            self.filenameEdit.setText('Current time')

# Functions for changing between choosing frames or time when recording.
            
    def specFrames(self):
        
        self.numExpositionsEdit.setEnabled(True)
        self.timeToRec.setEnabled(False)
        self.filesizeupdate()
    
    def specTime(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(True)
        self.specifyTime.setChecked(True)
        self.filesizeupdate()
            
# For updating the appriximated file size of and eventual recording. Called when frame dimensions
# or frames to record is changed.            
            
    def filesizeupdate(self):
        if self.specifyFrames.isChecked():
            frames = int(self.numExpositionsEdit.text())
        else:
            frames = int(self.timeToRec.text()) / self.main.RealExpPar.value()

        self.filesize = 2 * frames * self.main.shape[0] * self.main.shape[1]
        self.filesizeBar.setValue(min(2000000000, self.filesize)) #Percentage of 2 GB
        self.filesizeBar.setFormat(str(self.filesize/1000))

    def n(self):
        text = self.numExpositionsEdit.text()
        if text == '':
            return 0
        else:
            return int(text)

# Function that returns the time to record in order to record the correct number of frames.
            
    def getRecTime(self):
        
        if self.specifyFrames.isChecked():
            time = int(self.numExpositionsEdit.text()) * self.main.RealExpPar.value()
            return time
        else:
            return int(self.timeToRec.text())

    def openFolder(self, path):
        if sys.platform == 'darwin':
            subprocess.check_call(['open', '', self.folderEdit.text()])
        elif sys.platform == 'linux':
            subprocess.check_call(['gnome-open', '', self.folderEdit.text()])
        elif sys.platform == 'win32':
            os.startfile(self.folderEdit.text())

    def loadFolder(self):
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass

    # Attributes saving
    def getAttrs(self):
        attrs = self.main.tree.attrs()
        attrs.extend([('Date', time.strftime("%Y-%m-%d")),
                      ('Start time', time.strftime("%H:%M:%S")),
                      ('element_size_um', (1, 0.120, 0.120)),
                      ('NA', 1.42),
                      ('lambda_em', 670)])
        for laserControl in self.main.laserWidgets.controls:
            name = re.sub('<[^<]+?>', '', laserControl.name.text())
            attrs.append((name, laserControl.laser.power))
        return attrs

    def snapHDF(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

#            image = self.main.andor.most_recent_image16(self.main.shape)
            image = self.main.image

            name = os.path.join(folder, self.getFileName())
            savename = guitools.getUniqueName(name + '.hdf5')
            store_file = hdf.File(savename)
            store_file.create_dataset(name=self.dataname, data=image)
            for item in self.getAttrs():
                if item[1] is not None:
                    store_file[self.dataname].attrs[item[0]] = item[1]
            store_file.close()

        else:
            self.folderWarning()
            
    def getFileName(self):
        
        if self.specifyfile.checkState():
            filename = self.filenameEdit.text()
            
        else:
            filename = time.strftime('%Hh%Mm%Ss')
            
        return filename
        
    def snapTIFF(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

#            image = self.main.andor.most_recent_image16(self.main.shape)
            time.sleep(0.01)
            savename = (os.path.join(folder, self.getFileName()) +
                        '_snap.tiff')
            savename = guitools.getUniqueName(savename)
#            tiff.imsave(savename, np.flipud(image.astype(np.uint16)),
#                        description=self.dataname, software='Tormenta')
            tiff.imsave(savename, self.main.image.astype(np.uint16),
                        description=self.dataname, software='Tormenta')
            guitools.attrsToTxt(os.path.splitext(savename)[0], self.getAttrs())

        else:
            self.folderWarning()

    def folderWarning(self):
        root = Tk()
        root.withdraw()
        messagebox.showwarning(title='Warning', message="Folder doesn't exist")
        root.destroy()

    def updateGUI(self):

        eSecs = self.worker.timerecorded
        nframe = int(self.worker.timerecorded / self.main.RealExpPar.value())
        rSecs = self.getRecTime() - eSecs
        rText = '{}'.format(datetime.timedelta(seconds=max(0, rSecs)))
        self.tRemaining.setText(rText)
        self.currentFrame.setText(str(nframe) + ' /')
        self.currentTime.setText(str(int(eSecs)) + ' /')
        self.progressBar.setValue(100*(1 - rSecs / (eSecs + rSecs)))
        self.main.img.setImage(self.worker.liveImage, autoLevels=False)

# This funciton is called when "Rec" button is pressed. 

    def startRecording(self):
        if self.recButton.isChecked():  
            ret = QtGui.QMessageBox.Yes
            if self.filesize > 1500000000:  # Checks if estimated file size is dangourusly large, > 1,5GB-.
                ret = self.filesizewar.exec_()
                
            folder = self.folderEdit.text()
            if os.path.exists(folder) and ret == QtGui.QMessageBox.Yes:
                
                self.writable = False # Sets Recording widget to not be writable during recording.
                self.readyToRecord = False
                self.recButton.setEnabled(True)
                self.recButton.setText('STOP')
                self.main.tree.writable = False # Sets camera parameters to not be writable during recording.
                self.main.liveviewButton.setEnabled(False)
                self.main.liveviewPause() # Stops liveview from updating

                self.savename = (os.path.join(folder, self.getFileName()) + '_rec.hdf5') # Sets name for final output file
                self.savename = guitools.getUniqueName(self.savename) # If same  filename exists it is appended by (1) or (2) etc.
                self.startTime = ptime.time() # Saves the time when started to calculate remaining time.

                self.worker = RecWorker(self.main.orcaflash, self.getRecTime(), self.main.shape,  #Creates an instance of RecWorker class.
                                        self.main.RealExpPar, self.savename,
                                        self.dataname, self.getAttrs())
                self.worker.updateSignal.connect(self.updateGUI)    # Connects the updatesignal that is continously emitted from recworker to updateGUI function.
                self.worker.doneSignal.connect(self.endRecording) # Connects the donesignal emitted from recworker to endrecording function.
                self.recordingThread = QtCore.QThread() # Creates a new thread
                self.worker.moveToThread(self.recordingThread) # moves the worker object to this thread. 
                self.recordingThread.started.connect(self.worker.start)
                self.recordingThread.start()

            else:
                self.recButton.setChecked(False)
                self.folderWarning()

        else:
            self.worker.pressed = False

# Function called when recording finishes to reset relevent parameters.

    def endRecording(self):

        self.recordingThread.terminate() 

        converterFunction = lambda: guitools.TiffConverterThread(self.savename)
        self.main.exportlastAction.triggered.connect(converterFunction)
        self.main.exportlastAction.setEnabled(True)

        self.writable = True
        self.readyToRecord = True
        self.recButton.setText('REC')
        self.recButton.setChecked(False)
        self.main.tree.writable = True
        self.main.liveviewButton.setEnabled(True)
        self.main.liveviewRun()
        self.progressBar.setValue(0)
        self.currentTime.setText('0 /')
        self.currentFrame.setText('0 /')


class RecWorker(QtCore.QObject):

    updateSignal = QtCore.pyqtSignal()
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, orcaflash, timetorec, shape, t_exp, savename, dataname, attrs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orcaflash = orcaflash
        self.timetorec = timetorec
        self.shape = shape # Shape of one frame
        self.t_exp = t_exp
        self.savename = savename
        self.dataname = dataname
        self.attrs = attrs
        self.pressed = True

    def start(self):
        self.timerecorded = 0


        self.orcaflash.startAcquisition()
        time.sleep(0.1)

        self.starttime = time.time()
        f_count = 0
        while self.timerecorded < self.timetorec and self.pressed:
            self.timerecorded = time.time() - self.starttime
            f_count = f_count + np.size(self.orcaflash.newFrames())
            self.liveImage = self.orcaflash.hcam_data[f_count-2].getData()
            self.liveImage = np.reshape(self.liveImage, (self.orcaflash.frame_x, self.orcaflash.frame_y), order='F')

            self.updateSignal.emit()
            
        self.orcaflash.stopAcquisition()        
        data = [];
        for i in range(0, f_count):
            data.append(self.orcaflash.hcam_data[i].getData())
        datashape = (f_count, self.shape[1], self.shape[0])     # Adapted for ImageJ data read shape
        self.store_file = hdf.File(self.savename, "w")
        self.store_file.create_dataset(name=self.dataname, shape=datashape, maxshape=datashape, dtype=np.uint16)
        self.dataset = self.store_file[self.dataname]

            
        reshapeddata = np.reshape(data, datashape, order='C')
        self.dataset[...] = reshapeddata
     
        self.store_file.close()
        self.doneSignal.emit()


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




class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    def __init__(self, orcaflash, *args, **kwargs):
        super().__init__(*args, **kwargs)

        BinTip = ("Sets binning mode. Binning mode specifies if and how many \n"
                    "pixels are to be read out and interpreted as a single pixel value.")

        preampTip = ("Andor recommend using the highest value setting for \n"
                     "most low-light applications")

        EMGainTip = ("A gain of x4-5 the read noise (see spec sheet) is \n"
                     "enough to render this noise source negligible. In \n"
                     "practice, this can always be achieved with EM Gain of \n"
                     "less than x300 (often much less). Pushing gain beyond \n"
                     "300 would give little or no extra SNR benefit and \n"
                     "would only reduce dynamic range.")


        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': orcaflash.camera_id},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                      {'name': 'Binning', 'type': 'list', 
                                  'values': [1, 2, 4], 'tip': BinTip},
#{'name': 'Nr of rows', 'type': 'list',
#                       'values': [2048, 1024, 512, 256, 128, 64, 8]}, 
#{'name': 'Nr of columns', 'type': 'list',
#                       'values': [2048, 1024, 512, 256, 128, 64, 8]},
{'name': 'Mode', 'type': 'list', 'values': ['Full Widefield', 'Full chip', 'Custom']},
{'name': 'X0', 'type': 'int', 'value': 0, 'limits': (0, 2044)},
{'name': 'Y0', 'type': 'int', 'value': 0, 'limits': (0, 2044)},
{'name': 'Width', 'type': 'int', 'value': 2048, 'limits': (1, 2048)},
{'name': 'Height', 'type': 'int', 'value': 2048, 'limits': (1, 2048)}, 
                                  {'name': 'Apply', 'type': 'action'},
{'name': 'New ROI', 'type': 'action'}]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Set exposure time', 'type': 'float',
                       'value': 0.03, 'limits': (0,
                                                9999),
                       'siPrefix': True, 'suffix': 's'},
                      {'name': 'Internal frame interval', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': False,
                       'suffix': ' s'},
                      {'name': 'Real accumulation time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Internal frame rate', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': False,
                       'suffix': ' fps'}]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True

    def enableCropMode(self):
        value = self.frameTransferParam.value()
        if value:
            self.cropModeEnableParam.setWritable(True)
        else:
            self.cropModeEnableParam.setValue(False)
            self.cropModeEnableParam.setWritable(False)

    @property
    def writable(self):
        return self._writable

    @writable.setter
    def writable(self, value):
        """
        property to set basically the whole parameters tree as writable
        (value=True) or not writable (value=False)
        useful to set it as not writable during recording
        """
        self._writable = value
        framePar = self.p.param('Image frame')
        framePar.param('Binning').setWritable(value)
        framePar.param('Mode').setWritable(value)
        framePar.param('X0').setWritable(value)
        framePar.param('Y0').setWritable(value)
        framePar.param('Width').setWritable(value)
        framePar.param('Height').setWritable(value)
#       WARNING: If Apply and New ROI button are included here as are they will emit status changed signal
        # and their respective functions will be called... -> problems.
        
        timingPar = self.p.param('Timings')
        timingPar.param('Set exposure time').setWritable(value)

    def attrs(self):
        attrs = []
        for ParName in self.p.getValues():
            Par = self.p.param(str(ParName))
            if not(Par.hasChildren()):
                attrs.append((str(ParName), Par.value()))
            else:
                for sParName in Par.getValues():
                    sPar = Par.param(str(sParName))
                    if sPar.type() != 'action':
                        if not(sPar.hasChildren()):
                            attrs.append((str(sParName), sPar.value()))
                        else:
                            for ssParName in sPar.getValues():
                                ssPar = sPar.param(str(ssParName))
                                attrs.append((str(ssParName), ssPar.value()))
        return attrs


# The main GUI class.

class TormentaGUI(QtGui.QMainWindow):

    liveviewStarts = QtCore.pyqtSignal()
    liveviewEnds = QtCore.pyqtSignal()

    def __init__(self, bluelaser, violetlaser, uvlaser, scanZ, daq, orcaflash,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.orcaflash = orcaflash
        self.orcaflash.setPropertyValue("exposure_time", 0.001)
        self.shape = (self.orcaflash.getPropertyValue('image_height')[0], self.orcaflash.getPropertyValue('image_width')[0])
        self.frameStart = (0, 0)
        self.bluelaser = bluelaser
        self.violetlaser = violetlaser
        self.uvlaser = uvlaser
        self.scanZ = scanZ
        self.daq = daq
        
        self.filewarning = FileWarning()

        self.s = Q_(1, 's')
        self.lastTime = time.clock()
        self.fps = None

        # Actions and menubar
        # Shortcut only
        self.liveviewAction = QtGui.QAction(self)
        self.liveviewAction.setShortcut('Ctrl+Space')
        QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Space'), self,
                        self.liveviewKey)
        self.liveviewAction.triggered.connect(self.liveviewKey)
        self.liveviewAction.setEnabled(False)

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

#        self.shuttersAction = QtGui.QAction(('Close shutters when recording '
#                                             'is over'), self, checkable=True)
#        self.shuttersAction.setChecked(True)
#        self.shuttersAction.setStatusTip(('Close all laser shutters when the '
#                                          'recording session is over'))
#        fileMenu.addAction(self.shuttersAction)
#        fileMenu.addSeparator()

#        snapMenu = fileMenu.addMenu('Snap format')
#        self.snapTiffAction = QtGui.QAction('TIFF', self, checkable=True)
#        snapMenu.addAction(self.snapTiffAction)
#        self.snapHdf5Action = QtGui.QAction('HDF5', self, checkable=True)
#        snapMenu.addAction(self.snapHdf5Action)
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

        self.tree = CamParamTree(self.orcaflash)

        # Indicator for loading frame shape from a preset setting 
        # Currently not used.
        self.customFrameLoaded = False
        self.cropLoaded = False

        # Camera binning signals. Defines seperate variables for each parameter and connects the signal
        # emitted when they've been changed to a function that actually changes the parameters on the camera
        # or other appropriate action.
        self.framePar = self.tree.p.param('Image frame')
        self.binPar = self.framePar.param('Binning')
        self.binPar.sigValueChanged.connect(self.setBinning)
        self.FrameMode = self.framePar.param('Mode')
        self.FrameMode.sigValueChanged.connect(self.testfunction)
        self.X0par= self.framePar.param('X0')
        self.Y0par= self.framePar.param('Y0')
        self.Widthpar= self.framePar.param('Width')
        self.Heightpar= self.framePar.param('Height')
        self.applyParam = self.framePar.param('Apply')
        self.NewROIParam = self.framePar.param('New ROI')
        self.applyParam.sigStateChanged.connect(self.applyfcn)  #WARNING: This signal is emitted whenever anything about the status of the parameter changes eg is set writable or not.
        self.NewROIParam.sigStateChanged.connect(self.updateFrame)


        
        # Exposition signals
        timingsPar = self.tree.p.param('Timings')
        self.expPar = timingsPar.param('Set exposure time')
        self.expPar.sigValueChanged.connect(self.setExposure)
        self.RealExpPar = timingsPar.param('Internal frame interval')
        self.RealExpPar.setOpts(decimals = 5)
        self.setExposure()    # Set default values

        # Gain signals
#        self.PreGainPar = self.tree.p.param('Gain').param('Pre-amp gain')
#        updateGain = lambda: self.setGain
#        self.PreGainPar.sigValueChanged.connect(updateGain)
#        self.GainPar = self.tree.p.param('Gain').param('EM gain')
#        self.GainPar.sigValueChanged.connect(updateGain)
#        updateGain()        # Set default values

        # Camera settings widget
        cameraWidget = QtGui.QFrame()
        cameraWidget.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        cameraTitle = QtGui.QLabel('<h2><strong>Camera settings</strong></h2>')
        cameraTitle.setTextFormat(QtCore.Qt.RichText)
        cameraGrid = QtGui.QGridLayout()
        cameraWidget.setLayout(cameraGrid)
        cameraGrid.addWidget(cameraTitle, 0, 0)
        cameraGrid.addWidget(self.tree, 1, 0)

        self.presetsMenu = QtGui.QComboBox()
        self.presetDir = r'C:\Users\Usuario\Documents\Data\Presets'
        if not(os.path.isdir(self.presetDir)):
            self.presetDir = os.path.join(os.getcwd(), 'control/Presets')
        for preset in os.listdir(self.presetDir):
            self.presetsMenu.addItem(preset)
        self.loadPresetButton = QtGui.QPushButton('Load preset')
        loadPresetFunction = lambda: guitools.loadPreset(self)
        self.loadPresetButton.pressed.connect(loadPresetFunction)

        # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('LIVEVIEW')
        self.liveviewButton.setStyleSheet("font-size:18px")
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.liveviewButton.clicked.connect(self.liveview)      #Link button click to funciton liveview
        self.liveviewButton.setEnabled(True)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)

        # viewBox custom Tools
#        self.gridButton = QtGui.QPushButton('Grid')
#        self.gridButton.setCheckable(True)
#        self.gridButton.setEnabled(False)
#        self.grid2Button = QtGui.QPushButton('Two-color grid')
#        self.grid2Button.setCheckable(True)
#        self.grid2Button.setEnabled(False)
#        self.crosshairButton = QtGui.QPushButton('Crosshair')
#        self.crosshairButton.setCheckable(True)
#        self.crosshairButton.setEnabled(False)

#        self.flipperButton = QtGui.QPushButton('x1000')
#        self.flipperButton.setStyleSheet("font-size:16px")
#        self.flipperButton.setCheckable(True)
#        self.flipperButton.clicked.connect(self.daq.toggleFlipper)

        self.viewCtrl = QtGui.QWidget()
        self.viewCtrlLayout = QtGui.QGridLayout()
        self.viewCtrl.setLayout(self.viewCtrlLayout)
        self.viewCtrlLayout.addWidget(self.liveviewButton, 0, 0, 1, 3)
#        self.viewCtrlLayout.addWidget(self.gridButton, 1, 0)
#        self.viewCtrlLayout.addWidget(self.grid2Button, 1, 1)
#        self.viewCtrlLayout.addWidget(self.crosshairButton, 1, 2)
#        self.viewCtrlLayout.addWidget(self.flipperButton, 2, 0, 1, 3)

        self.fpsBox = QtGui.QLabel()
        self.fpsBox.setText('0 fps')
        self.statusBar().addPermanentWidget(self.fpsBox)
        self.tempStatus = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.tempStatus)
        self.temp = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.temp)
        self.cursorPos = QtGui.QLabel()
        self.cursorPos.setText('0, 0')
        self.statusBar().addPermanentWidget(self.cursorPos)

        # Temperature stabilization functionality
#        self.tempSetPoint = Q_(-50, 'degC')
#        self.stabilizer = TemperatureStabilizer(self)
#        self.stabilizerThread = QtCore.QThread()
#        self.stabilizer.moveToThread(self.stabilizerThread)
#        self.stabilizerThread.started.connect(self.stabilizer.start)
#        self.stabilizerThread.start()
#        self.liveviewStarts.connect(self.stabilizer.stop)
#        self.liveviewEnds.connect(self.stabilizer.start)

        # Recording settings widget
        self.recWidget = RecordingWidget(self)

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.lut = guitools.cubehelix()
        self.img.setLookupTable(self.lut)
        self.img.translate(-0.5, -0.5)
#        self.img.setPxMode(True)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
#        self.hist.vb.setLimits(yMin=0, yMax=2048)
        imageWidget.addItem(self.hist, row=1, col=2)
        self.ROI = guitools.ROI((0, 0), self.vb, (0, 0),
                                        handlePos=(1, 0), handleCenter=(0, 1),
scaleSnap=True, translateSnap=True)
        self.ROI.sigRegionChangeFinished.connect(self.ROIchanged)
        self.ROI.hide()


#        self.grid = guitools.Grid(self.vb, self.shape)
#        self.gridButton.clicked.connect(self.grid.toggle)
#        self.grid2 = guitools.TwoColorGrid(self.vb)
#        self.grid2Button.clicked.connect(self.grid2.toggle)
#        self.crosshair = guitools.Crosshair(self.vb)
#        self.crosshairButton.clicked.connect(self.crosshair.toggle)

        # x and y profiles
        xPlot = imageWidget.addPlot(row=0, col=1)
        xPlot.hideAxis('left')
        xPlot.hideAxis('bottom')
        self.xProfile = xPlot.plot()
        imageWidget.ci.layout.setRowMaximumHeight(0, 40)
        xPlot.setXLink(self.vb)
        yPlot = imageWidget.addPlot(row=1, col=0)
        yPlot.hideAxis('left')
        yPlot.hideAxis('bottom')
        self.yProfile = yPlot.plot()
        self.yProfile.rotate(90)
        imageWidget.ci.layout.setColumnMaximumWidth(0, 40)
        yPlot.setYLink(self.vb)

        # Initial camera configuration taken from the parameter tree
        self.orcaflash.setPropertyValue('exposure_time', self.expPar.value())
        self.adjustFrame()

        # Dock widget
        dockArea = DockArea()

        # Console widget
        consoleDock = Dock("Console", size=(600, 200))
        console = ConsoleWidget(namespace={'pg': pg, 'np': np})
        consoleDock.addWidget(console)
        dockArea.addDock(consoleDock)

        # Emission filters table widget
        wheelDock = Dock("Emission filters", size=(20, 20))
        tableWidget = pg.TableWidget(sortable=False)
        tableWidget.verticalHeader().hide()
        f = [('Rueda 1', 4, 'ZET642NF',    'Notch', ''),
             ('Rueda 2', 5, 'ET700/75m',   'Bandpass', 'Alexa647, Atto655'),
             ('Rueda 3', 6, 'FF01-725/40', 'Bandpass', 'Alexa700 (2 colores)'),
             ('Rueda 4', 1, '',            '', ''),
             ('Rueda 5', 2, 'FF03-525/50', 'Bandpass', 'GFP'),
             ('Rueda 6', 3, '',            'Bandpass', ''),
             ('Tubo', '',   'FF01-582/75', 'Bandpass', 'Alexa532, Alexa568, '
                                                       'Alexa700, \nAtto550, '
                                                       'Atto565, Nile Red')]
        data = np.array(f, dtype=[('Ubicación', object),
                                  ('Antiposición', object),
                                  ('Filtro', object),
                                  ('Tipo', object),
                                  ('Fluoróforos', object)])
        tableWidget.setData(data)
        tableWidget.resizeRowsToContents()
        wheelDock.addWidget(tableWidget)
        dockArea.addDock(wheelDock, 'top', consoleDock)

        # On time widget
        ontimeDock = Dock('On time histogram', size=(1, 1))
        self.ontimeWidget = ontime.OntimeWidget()
        ontimeDock.addWidget(self.ontimeWidget)
        dockArea.addDock(ontimeDock, 'above', wheelDock)

        # Molecule counting widget
        moleculesDock = Dock('Molecule counting', size=(1, 1))
        self.moleculeWidget = moleculesCounter.MoleculeWidget()
        moleculesDock.addWidget(self.moleculeWidget)
        dockArea.addDock(moleculesDock, 'above', ontimeDock)

        # Focus lock widget
        focusDock = Dock("Focus Control", size=(1, 1))
#        self.focusWidget = FocusWidget(DAQ, scanZ, self.recWidget)
        self.focusWidget = focus.FocusWidget(scanZ, self.recWidget)
#        self.focusThread = QtCore.QThread()
#        self.focusWidget.moveToThread(self.focusThread)
#        self.focusThread.started.connect(self.focusWidget)
#        self.focusThread.start()
        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'top', consoleDock)

        laserDock = Dock("Laser Control", size=(1, 1))
        self.lasers = (bluelaser, violetlaser, uvlaser)
        self.laserWidgets = lasercontrol.LaserWidget(self.lasers, self.daq)
        laserDock.addWidget(self.laserWidgets)
        dockArea.addDock(laserDock, 'above', moleculesDock)
        
        signalDock = Dock('Signal Generator')
        self.signalWidget = SignalGen.SigGenWidget()
        signalDock.addWidget(self.signalWidget)
        dockArea.addDock(signalDock, 'above', laserDock)
        

        self.setWindowTitle('Tempesta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(0, 350)
        layout.setColumnMinimumWidth(2, 600)
        layout.setColumnMinimumWidth(3, 200)
        layout.setRowMinimumHeight(1, 720)
        layout.setRowMinimumHeight(2, 910)
        layout.setRowMinimumHeight(3, 30)
        layout.addWidget(self.presetsMenu, 0, 0)
        layout.addWidget(self.loadPresetButton, 0, 1)
        layout.addWidget(cameraWidget, 1, 0, 2, 2)
        layout.addWidget(self.viewCtrl, 3, 0, 1, 2)
        layout.addWidget(self.recWidget, 4, 0, 1, 2)
        layout.addWidget(imageWidget, 0, 2, 5, 1)
        layout.addWidget(dockArea, 0, 3, 5, 1)

        layout.setRowMinimumHeight(2, 40)
        layout.setColumnMinimumWidth(2, 1000)
        
        
    def testfunction(self):
        print('In testfunction ie called from frame mode changed signal')
        self.updateFrame()
        
    def applyfcn(self):
        print('Apply pressed')
        self.resizeFrame()

    def mouseMoved(self, pos):
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(self.shape[1] - mousePoint.y())
            self.cursorPos.setText('{}, {}'.format(x, y))

    def flipperInPath(self, value):
        pass
#        self.flipperButton.setChecked(not(value))
#        self.daq.flipper = value


#    def cropCCD(self):
#
#        if self.cropParam.param('Enable').value():
#
#            # Used when cropmode is loaded from a config file
#            if self.cropLoaded:
#                self.startCropMode()
#
#            else:
#                self.FTMPar.setWritable()
#                if self.shape != self.andor.detector_shape:
#                    self.shape = self.andor.detector_shape
#                    self.frameStart = (1, 1)
#                    self.changeParameter(self.adjustFrame)
#
#                ROIpos = (0, 0)
#                self.cropROI = guitools.ROI(self.shape, self.vb, ROIpos,
#                                            handlePos=(1, 1), movable=False,
#                                            handleCenter=(0, 0),
#                                            scaleSnap=True, translateSnap=True)
#                # Signals
#                applyParam = self.cropParam.param('Apply')
#                applyParam.sigStateChanged.connect(self.startCropMode)
#
#        else:
#            self.cropROI.hide()
#            self.shape = self.andor.detector_shape
#            self.changeParameter(lambda: self.setCropMode(False))
#
#    def startCropMode(self):
#
#        # Used when cropmode is loaded from a config file
#        ROISize = self.cropROI.size()
#        self.shape = (int(ROISize[0]), int(ROISize[1]))
#        self.cropROI.hide()
#
#        self.frameStart = (1, 1)
#        self.andor.crop_mode_shape = self.shape
#        self.changeParameter(lambda: self.setCropMode(True))
#        self.vb.setLimits(xMin=-0.5, xMax=self.shape[0] - 0.5, minXRange=4,
#                          yMin=-0.5, yMax=self.shape[1] - 0.5, minYRange=4)
#        self.updateTimings()
#
#        self.grid.update(self.shape)
#        self.updateLevels()     # not working  # TODO: make this work
#
#    def setCropMode(self, state):
#        self.andor.crop_mode = state
#        if not(state):
#            self.shape = self.andor.detector_shape
#            self.adjustFrame()

    def changeParameter(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        try:
            function()
        except:

            self.liveviewPause()
            function()
            self.liveviewRun()

#        status = self.andor.status
#        if status != ('Camera is idle, waiting for instructions.'):
#            self.viewtimer.stop()
#            self.andor.abort_acquisition()
#
#        function()
#
#        if status != ('Camera is idle, waiting for instructions.'):
#            self.andor.start_acquisition()
#            time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
#            self.viewtimer.start(0)

    def updateLevels(self, image):
        std = np.std(image)
        self.hist.setLevels(np.min(image) - std, np.max(image) + std)

    def setBinning(self):
        
        """Method to change the binning of the captured frame"""

        binning = str(self.binPar.value())

        binstring = binning+'x'+binning
        coded = binstring.encode('ascii')
        
        try:
            self.orcaflash.setPropertyValue('binning', coded)
        
        except:
            self.liveviewPause()
            self.orcaflash.setPropertyValue('binning', coded)
            self.liveviewRun()

            
        
#    def setNrrows(self):
#        
#        """Method to change the number of rows of the captured frame"""
#        self.changeParameter(lambda: self.orcaflash.setPropertyValue('subarray_vsize', 8))
#
#    def setNrcols(self):
#        
#        """Method to change the number of rows of the captured frame"""
#        self.changeParameter(lambda: self.orcaflash.setPropertyValue('subarray_hsize', self.nrcolPar.value()))

    def setGain(self):
        """ Method to change the pre-amp gain and main gain of the EMCCD
        """
        pass
#        PreAmpGain = self.PreGainPar.value()
#        n = np.where(self.andor.PreAmps == PreAmpGain)[0][0]
#        # The (2 - n) accounts for the difference in order between the options
#        # in the GUI and the camera settings
#        self.andor.preamp = 2 - n
#        self.andor.EM_gain = self.GainPar.value()

    def setExposure(self):
        """ Method to change the exposure time setting
        """
        self.orcaflash.setPropertyValue('exposure_time', self.expPar.value())
#        self.andor.frame_transfer_mode = self.FTMPar.value()
#        hhRatesArr = np.array([item.magnitude for item in self.andor.HRRates])
#        n_hrr = np.where(hhRatesArr == self.HRRatePar.value().magnitude)[0][0]
#        # The (3 - n) accounts for the difference in order between the options
#        # in the GUI and the camera settings
#        self.andor.horiz_shift_speed = 3 - n_hrr
#
#        n_vss = np.where(np.array([item.magnitude
#                                  for item in self.andor.vertSpeeds])
#                         == self.vertShiftSpeedPar.value().magnitude)[0][0]
#        self.andor.vert_shift_speed = n_vss
#
#        n_vsa = np.where(np.array(self.andor.vertAmps) ==
#                         self.vertShiftAmpPar.value())[0][0]
#        self.andor.set_vert_clock(n_vsa)
#
        self.updateTimings()
        
    def cropOrca(self, hpos, vpos, hsize, vsize):
        """Method to crop the fram read out by Orcaflash """
#       Round to closest "divisable by 4" value.
        self.orcaflash.setPropertyValue('subarray_vpos', 0)
        self.orcaflash.setPropertyValue('subarray_hpos', 0)
        self.orcaflash.setPropertyValue('subarray_vsize', 2048)
        self.orcaflash.setPropertyValue('subarray_hsize', 2048)

 
        vpos = int(4*np.ceil(vpos/4))
        hpos = int(4*np.ceil(hpos/4))
        vsize = int(min(2048 - vpos, 4*np.ceil(vsize/4)))
        hsize = int(min(2048 - hpos, 4*np.ceil(hsize/4)))

        self.frameStart = (hpos, vpos)
        self.shape = (hsize, vsize)     
        print('orca will be cropped to: ', vpos, hpos, vsize, hsize)
        self.orcaflash.setPropertyValue('subarray_vsize', vsize)
        self.orcaflash.setPropertyValue('subarray_hsize', hsize)
        self.orcaflash.setPropertyValue('subarray_vpos', vpos)
        self.orcaflash.setPropertyValue('subarray_hpos', hpos)
        
        print('orca has been cropped to: ', vpos, hpos, vsize, hsize)

    def adjustFrame(self):
        """ Method to change the area of the sensor to be used and adjust the
        image widget accordingly. It needs a previous change in self.shape
        and self.frameStart)
        """
        binning = self.binPar.value()
#        self.andor.set_image(shape=self.shape, p_0=self.frameStart)
        self.changeParameter(lambda: self.cropOrca(binning*self.frameStart[0], binning*self.frameStart[1], binning*self.shape[0], binning*self.shape[1]))
#        self.vb.setLimits(xMin= -0.5, xMax=self.shape[0] - 0.5, minXRange=4,
#                          yMin= -0.5, yMax=self.shape[1] - 0.5, minYRange=4)

        self.updateTimings()
        self.recWidget.filesizeupdate()

#        self.grid.update(self.shape)
#        self.recWidget.shape = self.shape

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        print('Update frame called')
        frameParam = self.tree.p.param('Image frame')
        if frameParam.param('Mode').value() == 'Custom':

#            if not(self.customFrameLoaded):
            ROIsize = (int(0.2 * self.vb.viewRect().width()), int(0.2 * self.vb.viewRect().height()))
            ROIcenter = (int(self.vb.viewRect().center().x()), int(self.vb.viewRect().center().y()))
            ROIpos = (ROIcenter[0] - 0.5*ROIsize[0], ROIcenter[1] - 0.5*ROIsize[1])
            
#            try:
            self.ROI.setPos(ROIpos)
            self.ROI.setSize(ROIsize)
            self.ROI.show()
#            except:
#                self.ROI = guitools.ROI(ROIsize, self.vb, ROIpos,
#                                        handlePos=(1, 0), handleCenter=(0, 1),
#scaleSnap=True, translateSnap=True)
                
            self.ROIchanged()
            self.applyParam.setWritable(True)
            self.NewROIParam.setWritable(True)
            
        else:
            if frameParam.param('Mode').value() == 'Full Widefield':
                self.X0par.setValue(600)
                self.Y0par.setValue(600)
                self.Widthpar.setValue(900)
                self.Heightpar.setValue(900)
                self.resizeFrame()
                
    #            try:
                self.ROI.hide()
    #            except:
    #                pass

            elif frameParam.param('Mode').value() == 'Full chip':
                print('Full chip')
                self.X0par.setValue(0)
                self.Y0par.setValue(0)
                self.Widthpar.setValue(2048)
                self.Heightpar.setValue(2048)
                self.resizeFrame()
                
    #            try:
                self.ROI.hide()
    #            except:
    #                pass
#            self.applyParam.setWritable(False)
#            self.NewROIParam.setWritable(False)



#        else:
#            pass
#            side = int(frameParam.param('Mode').value().split('x')[0])
#            self.shape = (side, side)
#            start = int(0.5*(self.andor.detector_shape[0] - side) + 1)
#            self.frameStart = (start, start)
#
#            self.changeParameter(self.adjustFrame)
##            self.applyParam.disable()

    def ROIchanged(self):
        print('In ROIchanged..., ROI.pos[0]= ', self.ROI.pos()[0])
        self.X0par.setValue(self.frameStart[0] + int(self.ROI.pos()[0]))
        self.Y0par.setValue(self.frameStart[1] + int(self.ROI.pos()[1]))
        print(self.ROI.size()[0])
        print(self.ROI.size()[1])
        self.Widthpar.setValue(int(self.ROI.size()[0])) # [0] is Width
        self.Heightpar.setValue(int(self.ROI.size()[1])) # [1] is Height
    
    def resizeFrame(self):

#        ROISize = self.ROI.size()
#        self.shape = (int(ROISize[0]), int(ROISize[1]))
        self.shape = (self.Widthpar.value(), self.Heightpar.value())
        print('self.shape in resizeFrame is: ', self.shape)
        print('type of elements in self.shape is: ', type(self.shape[0]))
#        self.frameStart = (int(self.ROI.pos()[0]), int(self.ROI.pos()[1]))
        self.frameStart = (self.X0par.value(), self.Y0par.value())

        self.adjustFrame()
        self.ROI.hide()
#        self.grid.update(self.shape)
#        self.recWidget.shape = self.shape

    def updateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
#        timings = self.orcaflash.getPropertyValue('exposure_time') 
#        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        timingsPar = self.tree.p.param('Timings')
        RealExpPar = timingsPar.param('Internal frame interval')
        RealAccPar = timingsPar.param('Real accumulation time')
        EffFRPar = timingsPar.param('Internal frame rate')
        self.RealExpPar.setValue(self.orcaflash.getPropertyValue('internal_frame_interval')[0])
        RealAccPar.setValue(self.orcaflash.getPropertyValue('timing_readout_time')[0])
        EffFRPar.setValue(self.orcaflash.getPropertyValue('internal_frame_rate')[0])
#        RealExpPar.setValue(self.orcaflash.getPropertyValue('exposure_time')[0])
#        RealAccPar.setValue(self.orcaflash.getPropertyValue('accumulation_time')[0])
#        EffFRPar.setValue(1 / self.orcaflash.getPropertyValue('accumulation_time')[0])

    # This is the function triggered by the liveview shortcut
    def liveviewKey(self):

        if self.liveviewButton.isChecked():
            self.liveviewStop()
            self.liveviewButton.setChecked(False)

        else:
            self.liveviewStart(True)
            self.liveviewButton.setChecked(True)

    # This is the function triggered by pressing the liveview button
    def liveview(self):
        """ Image live view when not recording
        """
        if self.liveviewButton.isChecked():
            self.liveviewStart()

        else:
            self.liveviewStop()
            
        

    def liveviewStart(self):

#        self.orcaflash.startAcquisition()
#        time.sleep(0.3)
#        time.sleep(np.max((5 * self.t_exp_real.magnitude, 1)))
        self.updateFrame()
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        self.recWidget.readyToRecord = True
        self.liveviewRun()
#        self.liveviewStarts.emit()
#
#        idle = 'Camera is idle, waiting for instructions.'
#        if self.andor.status != idle:
#            self.andor.abort_acquisition()
#
#        self.andor.acquisition_mode = 'Run till abort'
#        self.andor.shutter(0, 1, 0, 0, 0)
#
#        self.andor.start_acquisition()
#        time.sleep(np.max((5 * self.t_exp_real.magnitude, 1)))
#        self.recWidget.readyToRecord = True
#        self.recWidget.recButton.setEnabled(True)
#
#        # Initial image
#        rawframes = self.orcaflash.getFrames()
#        firstframe = rawframes[0][-1].getData() #return A numpy array that contains the camera data. "Circular" indexing makes [-1] return the latest frame
#        self.image = np.reshape(firstframe, (self.orcaflash.frame_y, self.orcaflash.frame_x), order='C')
#        print(self.frame)
#        print(type(self.frame))
#        self.img.setImage(self.image, autoLevels=False, lut=self.lut) #Autolevels = True gives a stange numpy (?) warning
#        image = np.transpose(self.andor.most_recent_image16(self.shape))
#        self.img.setImage(image, autoLevels=False, lut=self.lut)
#        if update:
#            self.updateLevels(image)
#        self.viewtimer.start(0)
#        while self.liveviewButton.isChecked():
#            self.updateView()

#        self.moleculeWidget.enableBox.setEnabled(True)
#        self.gridButton.setEnabled(True)
#        self.grid2Button.setEnabled(True)
#        self.crosshairButton.setEnabled(True)

    def liveviewStop(self):
        self.viewtimer.stop()
        self.recWidget.readyToRecord = False

        # Turn off camera, close shutter
        self.orcaflash.stopAcquisition()
        self.img.setImage(np.zeros(self.shape), autoLevels=False)
        

#        self.liveviewEnds.emit()

#    def updateinThread(self):
#        
#        self.recordingThread = QtCore.QThread()
#        self.worker.moveToThread(self.recordingThread)
#        self.recordingThread.started.connect(self.worker.start)
#        self.recordingThread.start()
#        
#        self.updateThread = QtCore.QThread()
#        self.

    def liveviewRun(self):
        
        self.orcaflash.startAcquisition()
        time.sleep(0.3)
        self.viewtimer.start(0)
    
    def liveviewPause(self):
        
        self.viewtimer.stop()
        self.orcaflash.stopAcquisition()

    def updateView(self):
        """ Image update while in Liveview mode
        """

        rawframes = self.orcaflash.getFrames()

        firstframe = rawframes[0][-1].getData() #"Circular indexing" makes [-1] return the latest frame
        self.image = np.reshape(firstframe, (self.orcaflash.frame_x, self.orcaflash.frame_y), order='F')

        self.img.setImage(self.image, autoLevels=False, autoDownsample = False) 

        self.updateTimings()


    def fpsMath(self):
        now = ptime.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0/dt) * s
        self.fpsBox.setText('{} fps'.format(int(self.fps)))

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
#        self.stabilizer.timer.stop()
#        self.stabilizerThread.terminate()

        # Turn off camera, close shutter and flipper
#        if self.andor.status != 'Camera is idle, waiting for instructions.':
#            self.andor.abort_acquisition()
#        self.andor.shutter(0, 2, 0, 0, 0)
        self.orcaflash.shutdown()
        self.daq.flipper = True
        if self.signalWidget.running:
            self.signalWidget.StartStop()

        self.laserWidgets.closeEvent(*args, **kwargs)
        self.focusWidget.closeEvent(*args, **kwargs)
        self.signalWidget.closeEvent(*args, **kwargs)

        super().closeEvent(*args, **kwargs)
