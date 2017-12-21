# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@authors: Federico Barabas, Luciano Masullo, Andreas Bodén
"""

import subprocess
import sys
import numpy as np
import os
import time
import re

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

import control.lasercontrol as lasercontrol
import control.scanner as scanner
import control.align as align
import control.guitools as guitools
import control.focus as focus


# Widget to control image or sequence recording. Recording only possible when
# liveview active. StartRecording called when "Rec" presset. Creates recording
# thread with RecWorker, recording is then done in this seperate thread.
class RecordingWidget(QtGui.QFrame):
    '''Widget to control image or sequence recording.
    Recording only possible when liveview active.
    StartRecording called when "Rec" presset.
    Creates recording thread with RecWorker, recording is then done in this
    seperate thread.'''
    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.dataname = 'data'      # In case I need a QLineEdit for this

        self.recworkers = [None] * len(self.main.cameras)
        self.recthreads = [None] * len(self.main.cameras)
        self.savenames = [None] * len(self.main.cameras)

        self.z_stack = []
        self.recMode = 1

        self.dataDir = r"D:\Data"
        self.initialDir = os.path.join(self.dataDir, time.strftime('%Y-%m-%d'))

        self.filesizewar = QtGui.QMessageBox()
        self.filesizewar.setText("File size is very big!")
        self.filesizewar.setInformativeText(
            "Are you sure you want to continue?")
        self.filesizewar.setStandardButtons(
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        # Title
        recTitle = QtGui.QLabel('<h2><strong>Recording settings</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Folder and filename fields
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        openFolderButton = QtGui.QPushButton('Open')
        openFolderButton.clicked.connect(self.openFolder)
        self.specifyfile = QtGui.QCheckBox('Specify file name')
        self.specifyfile.clicked.connect(self.specFile)
        self.filenameEdit = QtGui.QLineEdit('Current_time')
        self.formatBox = QtGui.QComboBox()
        self.formatBox.addItem('tiff')
        self.formatBox.addItem('hdf5')

        # Snap and recording buttons
        self.showZgraph = QtGui.QCheckBox('Show Z-graph')
        self.showZproj = QtGui.QCheckBox('Show Z-projection')
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
        modeTitle = QtGui.QLabel('<strong>Mode</strong>')
        modeTitle.setTextFormat(QtCore.Qt.RichText)
        self.specifyFrames = QtGui.QRadioButton('Number of frames')
        self.specifyFrames.clicked.connect(self.specFrames)
        self.specifyTime = QtGui.QRadioButton('Time (s)')
        self.specifyTime.clicked.connect(self.specTime)
        self.recScanOnceBtn = QtGui.QRadioButton('Scan once')
        self.recScanOnceBtn.clicked.connect(self.recScanOnce)
        self.recScanLapseBtn = QtGui.QRadioButton('Time-lapse scan')
        self.recScanLapseBtn.clicked.connect(self.recScanLapse)
        self.timeLapseEdit = QtGui.QLineEdit('5')
        self.timeLapseLabel = QtGui.QLabel('Each/Total [s]')
        self.timeLapseLabel.setAlignment(QtCore.Qt.AlignRight)
        self.timeLapseTotalEdit = QtGui.QLineEdit('60')
        self.timeLapseScan = 0
        self.untilSTOPbtn = QtGui.QRadioButton('Run until STOP')
        self.untilSTOPbtn.clicked.connect(self.untilStop)
        self.timeToRec = QtGui.QLineEdit('1')
        self.timeToRec.setFixedWidth(45)
        self.timeToRec.textChanged.connect(self.filesizeupdate)
        self.currentTime = QtGui.QLabel('0 / ')
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
        buttonWidget.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                   QtGui.QSizePolicy.Expanding)
        buttonGrid.addWidget(self.recButton, 0, 2)

        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)

        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Folder'), 2, 0)
        recGrid.addWidget(self.showZgraph, 1, 0)
        recGrid.addWidget(self.showZproj, 1, 1)

        if len(self.main.cameras) > 1:
            self.DualCam = QtGui.QCheckBox('Two-cam rec')
            recGrid.addWidget(self.DualCam, 1, 3)
            recGrid.addWidget(self.folderEdit, 2, 1, 1, 2)
            recGrid.addWidget(openFolderButton, 2, 3)
            recGrid.addWidget(self.filenameEdit, 3, 1, 1, 2)
            recGrid.addWidget(self.formatBox, 3, 3)
        else:
            recGrid.addWidget(self.folderEdit, 2, 1)
            recGrid.addWidget(openFolderButton, 2, 2)
            recGrid.addWidget(self.filenameEdit, 3, 1)
            recGrid.addWidget(self.formatBox, 3, 2)

        recGrid.addWidget(self.specifyfile, 3, 0)

        recGrid.addWidget(modeTitle, 4, 0)
        recGrid.addWidget(self.specifyFrames, 5, 0, 1, 5)
        recGrid.addWidget(self.currentFrame, 5, 1)
        recGrid.addWidget(self.numExpositionsEdit, 5, 2)
        recGrid.addWidget(self.specifyTime, 6, 0, 1, 5)
        recGrid.addWidget(self.currentTime, 6, 1)
        recGrid.addWidget(self.timeToRec, 6, 2)
        recGrid.addWidget(self.tRemaining, 6, 3, 1, 2)
#        recGrid.addWidget(self.progressBar, 5, 4, 1, 2)
        recGrid.addWidget(self.recScanOnceBtn, 7, 0, 1, 5)
        recGrid.addWidget(self.recScanLapseBtn, 8, 0, 1, 5)
        recGrid.addWidget(self.timeLapseLabel, 8, 1)
        recGrid.addWidget(self.timeLapseEdit, 8, 2)
        recGrid.addWidget(self.timeLapseTotalEdit, 8, 3)
        recGrid.addWidget(self.untilSTOPbtn, 9, 0, 1, 5)
        recGrid.addWidget(buttonWidget, 10, 0, 1, 0)

        recGrid.setColumnMinimumWidth(0, 70)

        # Initial condition of fields and checkboxes.
        self.writable = True
        self.readyToRecord = False
        self.filenameEdit.setEnabled(False)
        self.specifyTime.setChecked(True)
        self.specTime()
        self.filesizeupdate()

    @property
    def readyToRecord(self):
        return self._readyToRecord

    @readyToRecord.setter
    def readyToRecord(self, value):
        self.snapTIFFButton.setEnabled(value)
        self.recButton.setEnabled(value)
        self._readyToRecord = value

    @property
    def writable(self):
        return self._writable

    # Setter for the writable property. If Nr of frame is checked only the
    # frames field is set active and viceversa.
    @writable.setter
    def writable(self, value):
        if value:
            if self.specifyFrames.isChecked():
                self.specFrames()
            elif self.specifyTime.isChecked():
                self.specTime()
            elif self.recScanOnceBtn.isChecked():
                self.recScanOnce()
            elif self.recScanLapseBtn.isChecked():
                self.recScanLapse()
            else:
                self.untilStop()
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

    # Functions for changing between choosing frames or time or "Run until
    # stop" when recording.
    def specFrames(self):
        self.numExpositionsEdit.setEnabled(True)
        self.timeToRec.setEnabled(False)
        self.timeLapseEdit.setEnabled(False)
        self.timeLapseTotalEdit.setEnabled(False)
        self.filesizeBar.setEnabled(True)
        self.progressBar.setEnabled(True)
        self.recMode = 1
        self.filesizeupdate()

    def specTime(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(True)
        self.timeLapseEdit.setEnabled(False)
        self.timeLapseTotalEdit.setEnabled(False)
        self.filesizeBar.setEnabled(True)
        self.progressBar.setEnabled(True)
        self.recMode = 2
        self.filesizeupdate()

    def recScanOnce(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(False)
        self.timeLapseEdit.setEnabled(False)
        self.timeLapseTotalEdit.setEnabled(False)
        self.filesizeBar.setEnabled(False)
        self.progressBar.setEnabled(False)
        self.recMode = 3

    def recScanLapse(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(False)
        self.timeLapseEdit.setEnabled(True)
        self.timeLapseTotalEdit.setEnabled(True)
        self.filesizeBar.setEnabled(False)
        self.progressBar.setEnabled(False)
        self.recMode = 4

    def untilStop(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(False)
        self.timeLapseEdit.setEnabled(False)
        self.timeLapseTotalEdit.setEnabled(False)
        self.filesizeBar.setEnabled(False)
        self.progressBar.setEnabled(False)
        self.recMode = 5

    def filesizeupdate(self):
        ''' For updating the approximated file size of and eventual recording.
        Called when frame dimensions or frames to record is changed.'''
        if self.specifyFrames.isChecked():
            frames = int(self.numExpositionsEdit.text())
        else:
            frames = int(self.timeToRec.text()) / self.main.RealExpPar.value()

        self.filesize = 2*frames*max([np.prod(s) for s in self.main.shapes])
        # Percentage of 2 GB
        self.filesizeBar.setValue(min(2000000000, self.filesize))
        self.filesizeBar.setFormat(str(self.filesize / 1000))

    def n(self):
        text = self.numExpositionsEdit.text()
        if text == '':
            return 0
        else:
            return int(text)

    def getTimeOrFrames(self):
        ''' Returns the time to record in order to record the correct number
        of frames.'''
        if self.specifyFrames.isChecked():
            return int(self.numExpositionsEdit.text())
        else:
            return int(self.timeToRec.text())

    def openFolder(self):
        try:
            if sys.platform == 'darwin':
                subprocess.check_call(['open', '', self.folderEdit.text()])
            elif sys.platform == 'linux':
                subprocess.check_call(
                    ['gnome-open', '', self.folderEdit.text()])
            elif sys.platform == 'win32':
                os.startfile(self.folderEdit.text())

        except FileNotFoundError:
            if sys.platform == 'darwin':
                subprocess.check_call(['open', '', self.dataDir])
            elif sys.platform == 'linux':
                subprocess.check_call(['gnome-open', '', self.dataDir])
            elif sys.platform == 'win32':
                os.startfile(self.dataDir)

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
        self.main.AbortROI()
        attrs = self.main.tree.attrs()

        for laserControl in self.main.laserWidgets.controls:
            name = re.sub('<[^<]+?>', '', laserControl.name.text())
            attrs.append((name, laserControl.laser.power))

        for key in self.main.scanWidget.scanParValues:
            attrs.append((key, self.main.scanWidget.scanParValues[key]))

        attrs.append(('Scan mode',
                      self.main.scanWidget.scanMode.currentText()))
        attrs.append(('True_if_scanning',
                      self.main.scanWidget.scanRadio.isChecked()))

        for key in self.main.scanWidget.pxParValues:
            attrs.append((key, self.main.scanWidget.pxParValues[key]))

        attrs.extend([('element_size_um', [1, 0.066, 0.066]),
                      ('Date', time.strftime("%Y-%m-%d")),
                      ('Saved at', time.strftime("%H:%M:%S")),
                      ('NA', 1.42)])

        return attrs

    def snapHDF(self):
        folder = self.folderEdit.text()

        if not os.path.exists(folder):
            os.mkdir(folder)

        image = self.main.image

        name = os.path.join(folder, self.getFileName())
        savename = guitools.getUniqueName(name + '.hdf5')
        store_file = hdf.File(savename)
        store_file.create_dataset(name=self.dataname, data=image)
        for item in self.getAttrs():
            if item[1] is not None:
                store_file[self.dataname].attrs[item[0]] = item[1]
        store_file.close()

    def getFileName(self):
        if self.specifyfile.checkState():
            filename = self.filenameEdit.text()

        else:
            filename = time.strftime('%Hh%Mm%Ss')

        return filename

    def snapTIFF(self):
        folder = self.folderEdit.text()
        if not os.path.exists(folder):
            os.mkdir(folder)

        time.sleep(0.01)
        savename = (os.path.join(folder, self.getFileName()) +
                    '_snap.tiff')
        savename = guitools.getUniqueName(savename)
        image = self.main.latest_images[self.main.currCamIdx].astype(np.uint16)
        tiff.imsave(
            savename, image, description=self.dataname, software='Tormenta',
            imagej=True, resolution=(1/self.main.umxpx, 1/self.main.umxpx),
            metadata={'spacing': 1, 'unit': 'um'})

        guitools.attrsToTxt(os.path.splitext(savename)[0], self.getAttrs())

    def folderWarning(self):
        root = Tk()
        root.withdraw()
        messagebox.showwarning(title='Warning', message="Folder doesn't exist")
        root.destroy()

    def updateGUI(self):
        pass
#        eSecs = self.worker.timerecorded
#        nframe = self.worker.frames_recorded
#        rSecs = self.getTimeOrFrames() - eSecs
#        rText = '{}'.format(datetime.timedelta(seconds=max(0, rSecs)))
#        self.tRemaining.setText(rText)
#        self.currentFrame.setText(str(nframe) + ' /')
#        self.currentTime.setText(str(int(eSecs)) + ' /')
#        self.progressBar.setValue(100*(1 - rSecs / (eSecs + rSecs)))

    def startRecording(self):
        ''' Called when "Rec" button is pressed.'''
        if self.recButton.isChecked():
            ret = QtGui.QMessageBox.Yes
            # Checks if estimated file size is dangourusly large, > 1,5GB-.
            if self.filesize > 1500000000:
                ret = self.filesizewar.exec_()

            if ret == QtGui.QMessageBox.Yes:

                # Sets Recording widget to not be writable during recording.
                self.writable = False
                self.readyToRecord = False
                self.recButton.setEnabled(True)
                self.recButton.setText('STOP')

                # Sets camera parameters to not be writable during recording.
                self.main.tree.writable = False
                self.main.liveviewButton.setEnabled(False)
#                self.main.liveviewStop() # Stops liveview from updating

                # Saves the time when started to calculate remaining time.
                self.startTime = ptime.time()

                if self.recMode == 4:
                    total = float(self.timeLapseTotalEdit.text())
                    each = float(self.timeLapseEdit.text())
                    self.timeLapseScan = int(np.ceil(total/each))
                    self.timer = QtCore.QTimer()
                    self.timer.timeout.connect(self.doRecording)
                    self.timer.start(float(self.timeLapseEdit.text())*1000)
                self.doRecording()

            else:
                self.recButton.setChecked(False)
                self.folderWarning()

        else:
            if self.recMode in [3, 4]:
                self.timeLapseScan = 0
                self.recButton.setEnabled(False)
                if not self.main.scanWidget.scanning:
                    self.endRecording()
            for i in range(0, self.nr_cameras):
                ind = np.mod(self.main.currCamIdx + i, 2)
                print('Terminating recording on camera index ' + str(ind))
                self.recworkers[ind].pressed = False

    def doRecording(self):
        if not self.main.scanWidget.scanning:
            self.makeSavenames()
            for i in range(0, self.nr_cameras):
                ind = np.mod(self.main.currCamIdx + i, 2)
                print('Starting recording on camera ' + str(ind + 1))

                # Creates an instance of RecWorker class.
                self.recworkers[ind] = RecWorker(
                    self, self.main.cameras[ind], self.recMode,
                    self.getTimeOrFrames(), self.main.shapes[ind],
                    self.main.lvworkers[ind], self.main.RealExpPar,
                    self.savenames[ind], self.dataname, self.getAttrs())
                # Connects the updatesignal that is continously emitted
                # from recworker to updateGUI function.
                self.recworkers[ind].updateSignal.connect(self.updateGUI)
                # Connects the donesignal emitted from recworker to
                # endrecording function.
                self.recworkers[ind].doneSignal.connect(self.endRecording)
                # Creates a new thread
                self.recthreads[ind] = QtCore.QThread()
                # moves the worker object to this thread.
                self.recworkers[ind].moveToThread(self.recthreads[ind])
                self.recthreads[ind].started.connect(
                    self.recworkers[ind].start)

            for i in range(0, self.nr_cameras):
                print('Starting recordings')
                ind = np.mod(self.main.currCamIdx + i, 2)
                self.recthreads[ind].start()

    def endRecording(self):
        """ Function called when recording finishes to reset relevent
        parameters."""
        if self.nr_cameras == 2 and (
                not self.recworkers[0].done or not self.recworkers[1].done):
            print('In first endRecordig "skip me" if')
        else:
            print('In endRecording')
            ind = self.main.currCamIdx
            if self.showZgraph.checkState():
                plt.figure()
                plt.plot(self.recworkers[ind].z_stack)
            if self.showZproj.checkState():
                plt.imshow(self.recworkers[ind].Z_projection, cmap='gray')

            for i in range(0, self.nr_cameras):
                ind = np.mod(self.main.currCamIdx + i, 2)
                self.recthreads[ind].terminate()
                # Same as done in Liveviewrun()

            print('After terminating recording thread')
            if self.recMode != 4:
                self.writable = True
                self.readyToRecord = True
                self.recButton.setText('REC')
                self.recButton.setChecked(False)
                self.main.tree.writable = True
                self.main.liveviewButton.setEnabled(True)
                self.progressBar.setValue(0)
                self.currentTime.setText('0 /')
                self.currentFrame.setText('0 /')
            else:
                self.timeLapseScan -= 1
                if self.timeLapseScan <= 0:
                    self.timer.stop()
                    self.writable = True
                    self.readyToRecord = True
                    self.recButton.setEnabled(True)
                    self.recButton.setText('REC')
                    self.recButton.setChecked(False)
                    self.main.tree.writable = True
                    self.main.liveviewButton.setEnabled(True)
                    self.progressBar.setValue(0)
                    self.currentTime.setText('0 /')
                    self.currentFrame.setText('0 /')

    def makeSavenames(self):
        try:
            print('Dual camera mode is ' + str(self.DualCam.checkState()))
            self.nr_cameras = 2
        except AttributeError:
            self.nr_cameras = 1
        folder = self.folderEdit.text()
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i in range(0, self.nr_cameras):
            ind = np.mod(self.main.currCamIdx + i, 2)
            # Sets name for final output file
            self.savenames[ind] = (os.path.join(folder, self.getFileName(
            )) + '_rec_cam_' + str(ind + 1))
            # If same  filename exists it is appended by (1) or (2)
            # etc.
            self.savenames[ind] = guitools.getUniqueName(self.savenames[ind])

    def saveData(self, data, savename):
        saveMode = self.formatBox.currentText()
        savename = savename + '.' + saveMode
        if saveMode == 'tiff':
            print('Savename = ', savename)
            t = time.time()
            para = dict(imagej=True)
            tiff.imsave(savename, data, **para)
        elif saveMode == 'hdf5':
            print('Savename = ', savename)
            store_file = hdf.File(savename, "w")
            datashape = data.shape
            store_file.create_dataset(name=self.dataname, shape=datashape,
                                      maxshape=datashape, dtype=np.uint16)
            dataset = store_file[self.dataname]
            t = time.time()
            dataset[...] = data
            # Saving parameters
            for item in self.attrs:
                if item[1] is not None:
                    dataset.attrs[item[0]] = item[1]
            store_file.close()
        elapsed = time.time() - t
        print('Data written, time to write: ', elapsed)


class RecWorker(QtCore.QObject):

    updateSignal = QtCore.pyqtSignal()
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, main, camera, recMode, timeorframes, shape, lvworker,
                 t_exp, savename, dataname, attrs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.camera = camera
        self.recMode = recMode  # 1=frames, 2=time, 3=until stop
        # Nr of seconds or frames to record depending on bool_ToF.
        self.timeorframes = timeorframes
        self.shape = shape  # Shape of one frame
        self.lvworker = lvworker
        self.t_exp = t_exp
        self.savename = savename
        self.dataname = dataname
        self.attrs = attrs
        self.pressed = True
        self.done = False
        self.scanWidget = self.main.main.scanWidget

    def start(self):
        # Set initial values
        self.timerecorded = 0

        self.lvworker.startRecording()
        time.sleep(0.1)

        self.starttime = time.time()
        numStored = 0  # number of frames stored
        saveMode = self.main.formatBox.currentText()

        # Main loop for waiting until recording is finished and sending update
        # signal
        if self.recMode == 1:
            if saveMode == 'tiff':
                with tiff.TiffWriter(self.savename + '.tiff',
                                     software='Tormenta') as storeFile:
                    while len(self.lvworker.framesRecorded) < self.timeorframes\
                            and self.pressed:
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        numStored += len(newFrames)
                        for frame in newFrames:
                            storeFile.save(frame)
                        self.updateSignal.emit()
            elif saveMode == 'hdf5':
                with hdf.File(self.savename + '.hdf5', "w") as storeFile:
                    storeFile.create_dataset('Images',
                                             (1, self.shape[0], self.shape[1]),
                                             maxshape=(None,
                                                       self.shape[0],
                                                       self.shape[1]))
                    dataset = storeFile['Images']
                    while len(self.lvworker.framesRecorded) < self.timeorframes\
                            and self.pressed:
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        dataset.resize((numStored+len(newFrames)), axis=0)
                        dataset[numStored:] = newFrames
                        numStored += len(newFrames)
                        self.updateSignal.emit()

        elif self.recMode == 2:
            if saveMode == 'tiff':
                with tiff.TiffWriter(self.savename + '.tiff',
                                     software='Tormenta') as storeFile:
                    while self.timerecorded < self.timeorframes and self.pressed:
                        self.timerecorded = time.time() - self.starttime
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        numStored += len(newFrames)
                        for frame in newFrames:
                            storeFile.save(frame)
                        self.updateSignal.emit()

            elif saveMode == 'hdf5':
                with hdf.File(self.savename + '.hdf5', "w") as storeFile:
                    storeFile.create_dataset('Images',
                                             (1, self.shape[0], self.shape[1]),
                                             maxshape=(None,
                                                       self.shape[0],
                                                       self.shape[1]))
                    dataset = storeFile['Images']
                    while self.timerecorded < self.timeorframes and self.pressed:
                        self.timerecorded = time.time() - self.starttime
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        dataset.resize((numStored+len(newFrames)), axis=0)
                        dataset[numStored:] = newFrames
                        numStored += len(newFrames)
                        self.updateSignal.emit()

        elif self.recMode in [3, 4]:
            # Change setting for scanning
            self.main.main.trigsourceparam.setValue('External "frame-trigger"')
            laserWidget = self.main.main.laserWidgets
            laserWidget.DigCtrl.DigitalControlButton.setChecked(True)

            # Getting Z steps
            if self.scanWidget.scanMode.currentText() == 'VOL scan':
                sizeZ = self.scanWidget.scanParValues['sizeZ']
                stepSizeZ = self.scanWidget.scanParValues['stepSizeZ']
                stepsZ = int(np.ceil(sizeZ / stepSizeZ))
            else:
                stepsZ = 1
            framesExpected = int(self.scanWidget.stageScan.frames / stepsZ)

            # start scanning
            self.scanWidget.scanButton.click()
            if saveMode == 'tiff':
                for i in range(stepsZ):
                    with tiff.TiffWriter(self.savename+'_z'+str(i)+'.tiff',
                                         software='Tormenta') as storeFile:
                        while numStored != framesExpected*(i+1)\
                                and self.pressed:
                            time.sleep(0.01)
                            newFrames = self.lvworker.framesRecorded[numStored:]
                            if numStored+len(newFrames) > framesExpected*(i+1):
                                maxF = framesExpected*(i+1)-numStored
                                newFrames = newFrames[:maxF]
                            numStored += len(newFrames)
                            for frame in newFrames:
                                storeFile.save(frame)
                            self.updateSignal.emit()

            elif saveMode == 'hdf5':
                with hdf.File(self.savename + '.hdf5', "w") as storeFile:
                    for i in range(stepsZ):
                        zPlane = storeFile.create_group('z' + str(i))
                        dataset = zPlane.create_dataset('Images',
                                              (1, self.shape[0],self.shape[1]),
                                              maxshape=(None, self.shape[0],
                                                        self.shape[1]))
                        while numStored != framesExpected*(i+1)\
                                and self.pressed:
                            time.sleep(0.01)
                            newFrames = self.lvworker.framesRecorded[numStored:]
                            if numStored+len(newFrames) > framesExpected*(i+1):
                                maxF = framesExpected*(i+1)-numStored
                                newFrames = newFrames[:maxF]
                            size = (numStored-framesExpected*i+len(newFrames))
                            dataset.resize(size, axis=0)
                            dataset[numStored:] = newFrames
                            numStored += len(newFrames)
                            self.updateSignal.emit()
        else:
            if saveMode == 'tiff':
                with tiff.TiffWriter(self.savename + '.tiff',
                                 software='Tormenta') as storeFile:
                    while self.pressed:
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        numStored += len(newFrames)
                        for frame in newFrames:
                            storeFile.save(frame)
                        self.updateSignal.emit()

            elif saveMode == 'hdf5':
                with hdf.File(self.savename + '.hdf5', "w") as storeFile:
                    storeFile.create_dataset('Images',
                                             (1, self.shape[0],
                                              self.shape[1]),
                                             maxshape=(None, self.shape[0],
                                                       self.shape[1]))
                    dataset = storeFile['Images']
                    while self.pressed:
                        time.sleep(0.01)
                        newFrames = self.lvworker.framesRecorded[numStored:]
                        dataset.resize((numStored + len(newFrames)), axis=0)
                        dataset[numStored:] = newFrames
                        numStored += len(newFrames)
                        self.updateSignal.emit()

        print('Number of recorded frame : '
              + str(len(self.lvworker.framesRecorded)))
        self.lvworker.stopRecording()

        self.done = True
        self.doneSignal.emit()


class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """
    def __init__(self, orcaflash, *args, **kwargs):
        super().__init__(*args, **kwargs)

        BinTip = ("Sets binning mode. Binning mode specifies if and how \n"
                  "many pixels are to be read out and interpreted as a \n"
                  "single pixel value.")

        # Parameter tree for the camera configuration
        params = [{'name': 'Model', 'type': 'str',
                   'value': orcaflash.camera_model.decode("utf-8")},
                  {'name': 'Pixel size', 'type': 'float',
                   'value': 0.238, 'readonly': True, 'suffix': ' µm'},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                      {'name': 'Binning', 'type': 'list',
                       'values': [1, 2, 4], 'tip': BinTip},
                      {'name': 'Mode', 'type': 'list', 'values':
                          ['Full Widefield', 'Full chip', 'Minimal line',
                           'Microlenses', 'Fast ROI', 'Fast ROI only v2',
                           'Custom']},
                      {'name': 'X0', 'type': 'int', 'value': 0,
                       'limits': (0, 2044)},
                      {'name': 'Y0', 'type': 'int', 'value': 0,
                       'limits': (0, 2044)},
                      {'name': 'Width', 'type': 'int', 'value': 2048,
                       'limits': (1, 2048)},
                      {'name': 'Height', 'type': 'int', 'value': 2048,
                       'limits': (1, 2048)},
                      {'name': 'Apply', 'type': 'action'},
                      {'name': 'New ROI', 'type': 'action'},
                      {'name': 'Abort ROI', 'type': 'action',
                       'align': 'right'}]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Set exposure time', 'type': 'float',
                       'value': 0.03, 'limits': (0, 9999),
                       'siPrefix': True, 'suffix': 's'},
                      {'name': 'Real exposure time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': ' s'},
                      {'name': 'Internal frame interval', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': ' s'},
                      {'name': 'Readout time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Internal frame rate', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': False,
                       'suffix': ' fps'}]},
                  {'name': 'Acquisition mode', 'type': 'group', 'children': [
                      {'name': 'Trigger source', 'type': 'list',
                       'values': ['Internal trigger',
                                  'External "Start-trigger"',
                                  'External "frame-trigger"'],
                       'siPrefix': True, 'suffix': 's'}]}]

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

        # WARNING: If Apply and New ROI button are included here they will
        # emit status changed signal and their respective functions will be
        # called... -> problems.
        timingPar = self.p.param('Timings')
        timingPar.param('Set exposure time').setWritable(value)

    def attrs(self):
        attrs = []
        for ParName in self.p.getValues():
            print(ParName)
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


class LVWorker(QtCore.QObject):

    def __init__(self, main, ind, orcaflash, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.ind = ind
        self.orcaflash = orcaflash
        self.running = False
        self.recording = False
        self.framesRecorded = []

        # Memory variable to keep track of if update has been run many times in
        # a row with camera trigger source as internal trigger
        self.mem = 0
        # If so the GUI trigger mode should also be set to internal trigger.
        # Happens when using external start tigger.

    def run(self):
        self.vtimer = QtCore.QTimer()
        self.vtimer.timeout.connect(self.update)
        self.running = True
        self.vtimer.start(30)
        time.sleep(0.03)

        # Grab first frame only to set suitable histogram limits
        hcData = self.orcaflash.getFrames()[0]
        frame = hcData[0].getData()
        self.image = np.reshape(
            frame, (self.orcaflash.frame_x, self.orcaflash.frame_y), 'F')
        self.main.latest_images[self.ind] = self.image
        self.main.hist.setLevels(*guitools.bestLimits(self.image))
        self.main.hist.vb.autoRange()

    def update(self):
        if self.running:
            try:
                hcData = self.orcaflash.getFrames()[0]
                frame = hcData[0].getData()
                self.image = np.reshape(
                    frame, (self.orcaflash.frame_x, self.orcaflash.frame_y),
                    'F')
                self.main.latest_images[self.ind] = self.image

                # stock frames while recording
                if self.recording:
                    for hcDatum in hcData:
                        reshapedFrame = np.reshape(hcDatum.getData(),
                                                   (self.orcaflash.frame_x,
                                                    self.orcaflash.frame_y),
                                                   'F')
                        self.framesRecorded.append(reshapedFrame)

                """Following is causing problems with two cameras..."""
    #            trigSource = self.orcaflash.getPropertyValue('trigSource')[0]
    #            print('Trigger source = ', trigSource)
    #            if trigSource == 1:
    #                if self.mem == 3:
    #                    self.main.trigsourceparam.setValue('Internal trigger')
    #                    self.mem = 0
    #                else:
    #                    self.mem = self.mem + 1
            except IndexError:
                pass

    def stop(self):
        if self.running:
            self.running = False
            print('Acquisition stopped')
        else:
            print('Cannot stop when not running (from LVThread)')

    def startRecording(self):
        self.recording = True
        self.framesRecorded.clear()

    def stopRecording(self):
        self.recording = False


class TormentaGUI(QtGui.QMainWindow):

    liveviewStarts = QtCore.pyqtSignal()
    liveviewEnds = QtCore.pyqtSignal()

    def __init__(self, actlaser, offlaser, exclaser, cameras, nidaq, pzt,
                 webcam, *args, **kwargs):
        super().__init__(*args, **kwargs)

#        self.resize(2400, 1300)

        self.lasers = [actlaser, offlaser, exclaser]
        self.cameras = cameras
        self.nidaq = nidaq
        self.orcaflash = self.cameras[0]

        for c in self.cameras:
            self.changeParameter(
                lambda: c.setPropertyValue('trigger_polarity', 2))
            # 3:DELAYED, 5:GLOBAL RESET
            self.changeParameter(
                lambda: c.setPropertyValue('trigger_global_exposure', 5))
            # 1: EGDE, 2: LEVEL, 3:SYNCHREADOUT
            self.changeParameter(lambda: c.setPropertyValue(
                'trigger_active', 2))

        self.shapes = [(c.getPropertyValue('image_height')[0],
                        c.getPropertyValue('image_width')[0])
                       for c in self.cameras]
        self.frameStart = (0, 0)

        self.currCamIdx = 0
        noImage = np.zeros(self.shapes[self.currCamIdx])
        self.latest_images = [noImage] * len(self.cameras)

        self.s = Q_(1, 's')
        self.lastTime = time.clock()
        self.fps = None

        # Actions and menubar
        # Shortcut only
        self.liveviewAction = QtGui.QAction(self)
        self.liveviewAction.setShortcut('Ctrl+Space')
        QtGui.QShortcut(
            QtGui.QKeySequence('Ctrl+Space'), self, self.liveviewKey)
        self.liveviewAction.triggered.connect(self.liveviewKey)
        self.liveviewAction.setEnabled(False)

        # Actions in menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        self.savePresetAction = QtGui.QAction('Save configuration...', self)
        self.savePresetAction.setShortcut('Ctrl+S')
        self.savePresetAction.setStatusTip('Save camera & recording settings')

        def savePresetFunction(): return guitools.savePreset(self)
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

        self.tree = CamParamTree(self.orcaflash)
        self.umxpx = self.tree.p.param('Pixel size').value()

        # Indicator for loading frame shape from a preset setting
        # Currently not used.
        self.customFrameLoaded = False
        self.cropLoaded = False

        # Camera binning signals. Defines seperate variables for each parameter
        # and connects the signal emitted when they've been changed to a
        # function that actually changes the parameters on the camera or other
        # appropriate action.
        self.framePar = self.tree.p.param('Image frame')
        self.binPar = self.framePar.param('Binning')
        self.binPar.sigValueChanged.connect(self.setBinning)
        self.FrameMode = self.framePar.param('Mode')
        self.FrameMode.sigValueChanged.connect(self.updateFrame)
        self.X0par = self.framePar.param('X0')
        self.Y0par = self.framePar.param('Y0')
        self.widthPar = self.framePar.param('Width')
        self.heightPar = self.framePar.param('Height')
        self.applyParam = self.framePar.param('Apply')
        self.NewROIParam = self.framePar.param('New ROI')
        self.AbortROIParam = self.framePar.param('Abort ROI')

        # WARNING: This signal is emitted whenever anything about the status of
        # the parameter changes eg is set writable or not.
        self.applyParam.sigStateChanged.connect(self.adjustFrame)
        self.NewROIParam.sigStateChanged.connect(self.updateFrame)
        self.AbortROIParam.sigStateChanged.connect(self.AbortROI)

        # Exposition signals
        timingsPar = self.tree.p.param('Timings')
        self.EffFRPar = timingsPar.param('Internal frame rate')
        self.expPar = timingsPar.param('Set exposure time')
        self.expPar.sigValueChanged.connect(self.setExposure)
        self.ReadoutPar = timingsPar.param('Readout time')
        self.RealExpPar = timingsPar.param('Real exposure time')
        self.FrameInt = timingsPar.param('Internal frame interval')
        self.RealExpPar.setOpts(decimals=5)
        self.setExposure()    # Set default values

        # Acquisition signals
        acquisParam = self.tree.p.param('Acquisition mode')
        self.trigsourceparam = acquisParam.param('Trigger source')
        self.trigsourceparam.sigValueChanged.connect(self.changeTriggerSource)

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

        def loadPresetFunction(): return guitools.loadPreset(self)
        self.loadPresetButton.pressed.connect(loadPresetFunction)

        # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('LIVEVIEW')
        self.liveviewButton.setStyleSheet("font-size:20px")
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        # Link button click to funciton liveview
        self.liveviewButton.clicked.connect(self.liveview)
        self.liveviewButton.setEnabled(True)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)

        self.alignmentON = False

        # Liveview control buttons
        self.viewCtrl = QtGui.QWidget()
        self.viewCtrlLayout = QtGui.QGridLayout()
        self.viewCtrl.setLayout(self.viewCtrlLayout)
        self.viewCtrlLayout.addWidget(self.liveviewButton, 0, 0, 1, 2)

        if len(self.cameras) > 1:
            self.toggleCamButton = QtGui.QPushButton('Toggle camera')
            self.toggleCamButton.setStyleSheet("font-size:18px")
            self.toggleCamButton.clicked.connect(self.toggleCamera)
            self.camLabel = QtGui.QLabel('Hamamatsu0')
            self.camLabel.setStyleSheet("font-size:18px")
            self.viewCtrlLayout.addWidget(self.toggleCamButton, 2, 0)
            self.viewCtrlLayout.addWidget(self.camLabel, 2, 1)

        # Status bar info
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
        self.cursorPosInt = QtGui.QLabel('0 counts', self)
        self.statusBar().addPermanentWidget(self.cursorPosInt)

        # Recording settings widget
        self.recWidget = RecordingWidget(self)

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.vb.setLimits(yMin=0, yMax=66000)
        self.cubehelixCM = pg.ColorMap(np.arange(0, 1, 1/256),
                                       guitools.cubehelix().astype(int))
        self.hist.gradient.setColorMap(self.cubehelixCM)
        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=1, col=2)
        self.ROI = guitools.ROI((0, 0), self.vb, (0, 0), handlePos=(1, 0),
                                handleCenter=(0, 1), color='y', scaleSnap=True,
                                translateSnap=True)
        self.ROI.sigRegionChangeFinished.connect(self.ROIchanged)
        self.ROI.hide()

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

        # viewBox custom Tools
        self.grid = guitools.Grid(self.vb)
        self.gridButton = QtGui.QPushButton('Grid')
        self.gridButton.setCheckable(True)
        self.gridButton.setEnabled(False)
        self.gridButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.gridButton.clicked.connect(self.grid.toggle)
        self.viewCtrlLayout.addWidget(self.gridButton, 1, 0)

        self.crosshair = guitools.Crosshair(self.vb)
        self.crosshairButton = QtGui.QPushButton('Crosshair')
        self.crosshairButton.setCheckable(True)
        self.crosshairButton.setEnabled(False)
        self.crosshairButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                           QtGui.QSizePolicy.Expanding)
        self.crosshairButton.pressed.connect(self.crosshair.toggle)
        self.viewCtrlLayout.addWidget(self.crosshairButton, 1, 1)

        self.levelsButton = QtGui.QPushButton('Update Levels')
        self.levelsButton.setEnabled(False)
        self.levelsButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                        QtGui.QSizePolicy.Expanding)
        self.levelsButton.pressed.connect(self.autoLevels)

        proxy = QtGui.QGraphicsProxyWidget()
        proxy.setWidget(self.levelsButton)
        imageWidget.addItem(proxy, row=0, col=2)

        # Initial camera configuration taken from the parameter tree
        self.orcaflash.setPropertyValue('exposure_time', self.expPar.value())
        self.adjustFrame()
        self.updateFrame()

        # Illumination dock area
        illumDockArea = DockArea()

        # Laser dock
        laserDock = Dock("Laser Control", size=(300, 1))
        self.laserWidgets = lasercontrol.LaserWidget(self.lasers, self.nidaq)
        laserDock.addWidget(self.laserWidgets)
        illumDockArea.addDock(laserDock)

        # Line Alignment Tool
        self.alignmentWidget = QtGui.QWidget()
        alignmentLayout = QtGui.QGridLayout()
        self.alignmentWidget.setLayout(alignmentLayout)
        self.angleEdit = QtGui.QLineEdit('30')
        self.alignmentLineMakerButton = QtGui.QPushButton('Alignment Line')
        self.angle = np.float(self.angleEdit.text())
        self.alignmentLineMakerButton.clicked.connect(self.alignmentToolAux)
        self.alignmentCheck = QtGui.QCheckBox('Show Alignment Tool')
        alignmentLayout.addWidget(QtGui.QLabel('Line Angle'), 0, 0)
        alignmentLayout.addWidget(self.angleEdit, 0, 1)
        alignmentLayout.addWidget(self.alignmentLineMakerButton, 1, 0)
        alignmentLayout.addWidget(self.alignmentCheck, 1, 1)
        alignmentDock = Dock("Alignment Tool", size=(1, 1))
        alignmentDock.addWidget(self.alignmentWidget)
        illumDockArea.addDock(alignmentDock, 'right')

        # Z align widget
        ZalignDock = Dock("Axial Alignment Tool", size=(1, 1))
        self.ZalignWidget = align.AlignWidgetAverage(self)
        ZalignDock.addWidget(self.ZalignWidget)
        illumDockArea.addDock(ZalignDock, 'above', alignmentDock)

        # Rotational align widget
        RotalignDock = Dock("Rotational Alignment Tool", size=(1, 1))
        self.RotalignWidget = align.AlignWidgetXYProject(self)
        RotalignDock.addWidget(self.RotalignWidget)
        illumDockArea.addDock(RotalignDock, 'above', alignmentDock)

        # Dock widget
        dockArea = DockArea()

        # Focus Lock widget
        FocusLockDock = Dock("Focus Lock", size=(400, 400))
        self.FocusLockWidget = focus.FocusWidget(pzt, webcam)
        FocusLockDock.addWidget(self.FocusLockWidget)
        dockArea.addDock(FocusLockDock)

        # Scanner
        scanDock = Dock('Scan', size=(1, 1))
        self.scanWidget = scanner.ScanWidget(self.nidaq, self)
        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock, 'below', FocusLockDock)

        # Piezo positioner
        piezoDock = Dock('Piezo positioner', size=(1, 1))
        self.piezoWidget = scanner.Positionner(self.scanWidget)
        piezoDock.addWidget(self.piezoWidget)
        dockArea.addDock(piezoDock, 'bottom', alignmentDock)

        console = ConsoleWidget(namespace={'pg': pg, 'np': np})

        self.setWindowTitle('TempestaDev')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.addWidget(self.presetsMenu, 0, 0)
        layout.addWidget(self.loadPresetButton, 0, 1)
        layout.addWidget(cameraWidget, 1, 0, 2, 2)
        layout.addWidget(self.viewCtrl, 3, 0, 1, 2)
        layout.addWidget(self.recWidget, 4, 0, 1, 2)
        layout.addWidget(console, 5, 0, 1, 2)
        layout.addWidget(imageWidget, 0, 2, 6, 1)
        layout.addWidget(illumDockArea, 0, 3, 2, 1)
        layout.addWidget(dockArea, 2, 3, 4, 1)

        layout.setRowMinimumHeight(2, 175)
        layout.setRowMinimumHeight(3, 100)
        layout.setRowMinimumHeight(5, 175)
        layout.setColumnMinimumWidth(0, 275)
        imageWidget.ci.layout.setColumnFixedWidth(1, 1150)
        imageWidget.ci.layout.setRowFixedHeight(1, 1150)
        layout.setColumnMinimumWidth(2, 1350)

    def autoLevels(self):
        self.hist.setLevels(*guitools.bestLimits(self.img.image))
        self.hist.vb.autoRange()

    def toggleCamera(self):
        self.currCamIdx = (self.currCamIdx + 1) % len(self.cameras)
        self.orcaflash = self.cameras[self.currCamIdx]
        self.camLabel.setText('Hamamatsu {}'.format(self.currCamIdx))
        self.updateTimings()
        self.expPar.setValue(self.RealExpPar.value())

    def mouseMoved(self, pos):
        if self.vb.sceneBoundingRect().contains(pos):
            # Get pointer position
            mousePoint = self.vb.mapSceneToView(pos)
            x = int(mousePoint.x())
            y = int(self.shapes[self.currCamIdx][1] - mousePoint.y())

            try:
                # Outputs
                self.cursorPos.setText('{}, {}'.format(x, y))
                currCamWorker = self.lvworkers[self.currCamIdx]
                cs = currCamWorker.image[x, int(mousePoint.y())]
                countsStr = '{} counts'.format(cs)
                self.cursorPosInt.setText(countsStr)
            except AttributeError:
                pass

    def changeParameter(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        try:
            function()
        except BaseException:
            self.liveviewPause()
            function()
            self.liveviewRun()

    def changeTriggerSource(self):
        if self.trigsourceparam.value() == 'Internal trigger':
            self.changeParameter(
                lambda: self.cameras[self.currCamIdx].setPropertyValue(
                    'trigger_source', 1))

        elif self.trigsourceparam.value() == 'External "Start-trigger"':
            self.changeParameter(
                lambda: self.cameras[self.currCamIdx].setPropertyValue(
                    'trigger_source', 2))
            self.changeParameter(
                lambda: self.cameras[self.currCamIdx].setPropertyValue(
                    'trigger_mode', 6))

        elif self.trigsourceparam.value() == 'External "frame-trigger"':
            self.changeParameter(
                lambda: self.cameras[self.currCamIdx].setPropertyValue(
                    'trigger_source', 2))
            self.changeParameter(
                lambda: self.cameras[self.currCamIdx].setPropertyValue(
                    'trigger_mode', 1))

    def updateLevels(self, image):
        std = np.std(image)
        self.hist.setLevels(np.min(image) - std, np.max(image) + std)

    def setBinning(self):
        """Method to change the binning of the captured frame."""
        binning = str(self.binPar.value())
        binstring = binning + 'x' + binning
        coded = binstring.encode('ascii')

        self.changeParameter(
           lambda: self.orcaflash.setPropertyValue('binning', coded))

#    def setNrrows(self):
#        """Method to change the number of rows of the captured frame"""
#        self.changeParameter(
#            lambda: self.orcaflash.setPropertyValue('subarray_vsize', 8))
#
#    def setNrcols(self):
#        """Method to change the number of rows of the captured frame"""
#        self.changeParameter(
#            lambda: self.orcaflash.setPropertyValue('subarray_hsize',
#                                                    self.nrcolPar.value()))

    def setExposure(self):
        """ Method to change the exposure time setting."""
        self.orcaflash.setPropertyValue('exposure_time', self.expPar.value())
        self.updateTimings()

    def cropOrca(self, hpos, vpos, hsize, vsize):
        """Method to crop the frame read out by Orcaflash. """
        print('cropping camera' + str(self.currCamIdx))
        self.cameras[self.currCamIdx].setPropertyValue('subarray_vpos', 0)
        self.cameras[self.currCamIdx].setPropertyValue('subarray_hpos', 0)
        self.cameras[self.currCamIdx].setPropertyValue(
            'subarray_vsize', 2048)
        self.cameras[self.currCamIdx].setPropertyValue(
            'subarray_hsize', 2048)

        # Round to closest "divisable by 4" value.
        vpos = int(4 * np.ceil(vpos / 4))
        hpos = int(4 * np.ceil(hpos / 4))
        # V3 camera seems to only be able to take multiples of 128.
        minroi = 64
        vsize = int(min(2048 - vpos, minroi * np.ceil(vsize / minroi)))
        hsize = int(min(2048 - hpos, minroi * np.ceil(hsize / minroi)))

        self.cameras[self.currCamIdx].setPropertyValue('subarray_vsize', vsize)
        self.cameras[self.currCamIdx].setPropertyValue('subarray_hsize', hsize)
        self.cameras[self.currCamIdx].setPropertyValue('subarray_vpos', vpos)
        self.cameras[self.currCamIdx].setPropertyValue('subarray_hpos', hpos)

        # This should be the only place where self.frameStart is changed
        self.frameStart = (hpos, vpos)
        # Only place self.shapes is changed
        self.shapes[self.currCamIdx] = (hsize, vsize)
        print('orca has been cropped to: ', vpos, hpos, vsize, hsize)

    def adjustFrame(self):
        """ Method to change the area of the sensor to be used and adjust the
        image widget accordingly. It needs a previous change in self.shape
        and self.frameStart)
        """

        binning = self.binPar.value()
        width = self.widthPar.value()
        height = self.heightPar.value()
        self.changeParameter(lambda: self.cropOrca(binning*self.X0par.value(),
                                                   binning*self.Y0par.value(),
                                                   binning*width, height))

        # Final shape values might differ from the user-specified one because
        # of camera limitation x128
        width, height = self.shapes[self.currCamIdx]
        self.X0par.setValue(self.frameStart[0])
        self.Y0par.setValue(self.frameStart[1])
        self.widthPar.setValue(width)
        self.heightPar.setValue(height)

        self.vb.setLimits(xMin=-0.5, xMax=width - 0.5, minXRange=4,
                          yMin=-0.5, yMax=height - 0.5, minYRange=4)
        self.vb.setAspectLocked()
        self.grid.update([width, height])
        self.updateTimings()
        self.recWidget.filesizeupdate()
        self.ROI.hide()

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor.
        """
        frameParam = self.tree.p.param('Image frame')
        if frameParam.param('Mode').value() == 'Custom':
            self.X0par.setWritable(True)
            self.Y0par.setWritable(True)
            self.widthPar.setWritable(True)
            self.heightPar.setWritable(True)

            ROIsize = (64, 64)
            ROIcenter = (int(self.vb.viewRect().center().x()),
                         int(self.vb.viewRect().center().y()))
            ROIpos = (ROIcenter[0] - 0.5 * ROIsize[0],
                      ROIcenter[1] - 0.5 * ROIsize[1])

            self.ROI.setPos(ROIpos)
            self.ROI.setSize(ROIsize)
            self.ROI.show()
            self.ROIchanged()

        else:
            self.X0par.setWritable(False)
            self.Y0par.setWritable(False)
            self.widthPar.setWritable(False)
            self.heightPar.setWritable(False)

            if frameParam.param('Mode').value() == 'Full Widefield':
                self.X0par.setValue(630)
                self.Y0par.setValue(610)
                self.widthPar.setValue(800)
                self.heightPar.setValue(800)
                self.adjustFrame()
                self.ROI.hide()

            elif frameParam.param('Mode').value() == 'Full chip':
                self.X0par.setValue(0)
                self.Y0par.setValue(0)
                self.widthPar.setValue(2048)
                self.heightPar.setValue(2048)
                self.adjustFrame()

                self.ROI.hide()

            elif frameParam.param('Mode').value() == 'Microlenses':
                self.X0par.setValue(595)
                self.Y0par.setValue(685)
                self.widthPar.setValue(600)
                self.heightPar.setValue(600)
                self.adjustFrame()
                self.ROI.hide()

            elif frameParam.param('Mode').value() == 'Fast ROI':
                self.X0par.setValue(595)
                self.Y0par.setValue(960)
                self.widthPar.setValue(600)
                self.heightPar.setValue(128)
                self.adjustFrame()
                self.ROI.hide()

            elif frameParam.param('Mode').value() == 'Fast ROI only v2':
                self.X0par.setValue(595)
                self.Y0par.setValue(1000)
                self.widthPar.setValue(600)
                self.heightPar.setValue(50)
                self.adjustFrame()
                self.ROI.hide()

            elif frameParam.param('Mode').value() == 'Minimal line':
                self.X0par.setValue(0)
                self.Y0par.setValue(1020)
                self.widthPar.setValue(2048)
                self.heightPar.setValue(8)
                self.adjustFrame()
                self.ROI.hide()

    def ROIchanged(self):

        self.X0par.setValue(self.frameStart[0] + int(self.ROI.pos()[0]))
        self.Y0par.setValue(self.frameStart[1] + int(self.ROI.pos()[1]))

        self.widthPar.setValue(int(self.ROI.size()[0]))   # [0] is Width
        self.heightPar.setValue(int(self.ROI.size()[1]))  # [1] is Height

    def AbortROI(self):

        self.ROI.hide()

        self.X0par.setValue(self.frameStart[0])
        self.Y0par.setValue(self.frameStart[1])

        self.widthPar.setValue(self.shapes[self.currCamIdx][0])
        self.heightPar.setValue(self.shapes[self.currCamIdx][1])

    def updateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree."""
        self.RealExpPar.setValue(
            self.orcaflash.getPropertyValue('exposure_time')[0])
        self.FrameInt.setValue(
            self.orcaflash.getPropertyValue('internal_frame_interval')[0])
        self.ReadoutPar.setValue(
            self.orcaflash.getPropertyValue('timing_readout_time')[0])
        self.EffFRPar.setValue(
            self.orcaflash.getPropertyValue('internal_frame_rate')[0])

    def liveviewKey(self):
        '''Triggered by the liveview shortcut.'''

        if self.liveviewButton.isChecked():
            self.liveviewStop()
            self.liveviewButton.setChecked(False)

        else:
            self.liveviewStart(True)
            self.liveviewButton.setChecked(True)

    def liveview(self):
        """ Triggered by pressing the liveview button. Image live view when
        not recording. """
        if self.liveviewButton.isChecked():
            self.liveviewStart()
        else:
            self.liveviewStop()

    # Threading below is done in this way since making LVThread a QThread
    # resulted in QTimer not functioning in the thread. Image is now also
    # saved as latest_image in TormentaGUI class since setting image in GUI
    # from thread resultet in issues when interacting with the viewbox from
    # GUI. Maybe due to simultaious manipulation of viewbox from GUI and
    # thread.
    def liveviewStart(self):
        ''' Threading below  is done in this way since making LVThread a
        QThread resulted in QTimer not functioning in the thread. Image is now
        also saved as latest_image in TormentaGUI class since setting image in
        GUI from thread results in issues when interacting with the viewbox
        from GUI. Maybe due to simultaneous manipulation of viewbox from GUI
        and thread.'''

        self.crosshairButton.setEnabled(True)
        self.gridButton.setEnabled(True)
        self.levelsButton.setEnabled(True)
        self.recWidget.readyToRecord = True

        self.lvworkers = [None] * len(self.cameras)
        self.lvthreads = [None] * len(self.cameras)

        for i in np.arange(len(self.cameras)):
            self.lvworkers[i] = LVWorker(self, i, self.cameras[i])
            self.lvthreads[i] = QtCore.QThread()
            self.lvworkers[i].moveToThread(self.lvthreads[i])
            self.lvthreads[i].started.connect(self.lvworkers[i].run)
            self.lvthreads[i].start()
            self.viewtimer.start(30)

        self.liveviewRun()

    def liveviewStop(self):

        for i in np.arange(len(self.cameras)):
            self.lvworkers[i].stop()
            self.lvthreads[i].terminate()
            # Turn off camera, close shutter
            self.cameras[i].stopAcquisition()

        self.viewtimer.stop()
        self.crosshairButton.setEnabled(False)
        self.gridButton.setEnabled(False)
        self.levelsButton.setEnabled(False)
        self.recWidget.readyToRecord = False

        self.vb.scene().sigMouseMoved.disconnect()
        self.img.setImage(
            np.zeros(self.shapes[self.currCamIdx]), autoLevels=False)

    def liveviewRun(self):
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        for i in np.arange(len(self.cameras)):
            # Needed if parameter is changed during liveview since that causes
            # camera to start writing to buffer place zero again.
            self.cameras[i].startAcquisition()

    def liveviewPause(self):
        for c in self.cameras:
            c.stopAcquisition()

    def updateView(self):
        """ Image update while in Liveview mode
        """
        self.img.setImage(self.latest_images[self.currCamIdx],
                          autoLevels=False, autoDownsample=False)
        if self.alignmentON:
            if self.alignmentCheck.isChecked():
                self.vb.addItem(self.alignmentLine)
            else:
                self.vb.removeItem(self.alignmentLine)

    def alignmentToolAux(self):
        self.angle = np.float(self.angleEdit.text())
        return self.alignmentToolMaker(self.angle)

    def alignmentToolMaker(self, angle):

        # alignmentLine
        try:
            self.vb.removeItem(self.alignmentLine)
        except BaseException:
            pass

        pen = pg.mkPen(color=(255, 255, 0), width=0.5,
                       style=QtCore.Qt.SolidLine, antialias=True)
        self.alignmentLine = pg.InfiniteLine(
            pen=pen, angle=angle, movable=True)
        self.alignmentON = True

    def fpsMath(self):
        now = ptime.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0 / dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0 / dt) * s
        self.fpsBox.setText('{} fps'.format(int(self.fps)))

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
        try:
            for thread in self.lvthreads:
                thread.terminate()
        except BaseException:
            pass

        # Turn off camera, close shutter and flipper
        for c in self.cameras:
            c.shutdown()

        self.nidaq.reset_device()

        self.laserWidgets.closeEvent(*args, **kwargs)
        self.ZalignWidget.closeEvent(*args, **kwargs)
        self.RotalignWidget.closeEvent(*args, **kwargs)
        self.scanWidget.closeEvent(*args, **kwargs)
        self.FocusLockWidget.closeEvent(*args, **kwargs)

        super().closeEvent(*args, **kwargs)
