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
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Pipe
import queue # Queue.Empty exception is not available in the multiprocessing Queue namespace

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
import control.Scan as Scan
import control.focus as focus
import control.align as align
import control.molecules_counter as moleculesCounter
import control.ontime as ontime
import control.guitools as guitools
from control import libnidaqmx

def write_data(datashape, dataname, savename, attrs, p):
    """Function meant to be run in seperate process that gets frames to save from a pipe and saves them
    to a hdf5 file.
    NOTE: since every call to write to the dataset appearently includes some overhead it, the speed of the writing
    is optimized by gathering a bunch of frames at a time and writing a whole bunch at a time."""
    frame_shape = [datashape[1], datashape[2]]   
        
    running = True
    # Initiate file to save to
    with hdf.File(savename, "w") as store_file:
        store_file.create_dataset(name=dataname, shape=datashape, maxshape=datashape, dtype=np.uint16)
        dataset = store_file[dataname]
        
        bn = 0
        f_ind = 0
        pkg = None
        while running:
            p.send('Ready')
            f_bunch = []
            while p.poll(0.01):
                pkg = p.recv() #Package should be either the string 'Finish' or a frame to be saved
                p.send('Ready')
                if pkg == 'Finish':
                    running = False
                    print('Running set to False, exit loop')
                else:
                    f_bunch.append(pkg)
                    bn = bn + 1
                    
            if bn > 0:
                bunch_shape = [bn, frame_shape[0], frame_shape[1]]
                frames = np.reshape(f_bunch, bunch_shape, order='C')
                dataset[f_ind:f_ind+bn:1, :, :] = frames
                f_ind = f_ind + bn
                print('Written to file, bn = ', bn)
                bn = 0
        
        # Saving parameters
        for item in attrs:
            if item[1] is not None:
                dataset.attrs[item[0]] = item[1]        
        
        dataset.resize((f_ind, frame_shape[0], frame_shape[1]))
        store_file.close()
        print('File closed (from "write_data")')

#Widget to control image or sequence recording. Recording only possible when liveview active.
#StartRecording called when "Rec" presset. Creates recording thread with RecWorker, recording is then 
#done in this seperate thread.

class RecordingWidget(QtGui.QFrame):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.dataname = 'data'      # In case I need a QLineEdit for this

        startdir = r'E:\Tempesta\DefaultDataFolder\%s'
        newfolderpath =  startdir % time.strftime('%Y-%m-%d')
        if not os.path.exists(newfolderpath):
            os.mkdir(newfolderpath)

        self.z_stack  = []
        self.rec_mode = 1;
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
#        openFolderButton = QtGui.QPushButton('Open')
#        openFolderButton.clicked.connect(self.openFolder)
#        loadFolderButton = QtGui.QPushButton('Load...')
#        loadFolderButton.clicked.connect(self.loadFolder)
        self.specifyfile = QtGui.QCheckBox('Specify file name')
        self.specifyfile.clicked.connect(self.specFile)
        self.filenameEdit = QtGui.QLineEdit('Current_time')

        # Snap and recording buttons
        self.showZgraph = QtGui.QCheckBox('Show Z-graph after rec')
        self.showZproj = QtGui.QCheckBox('Show Z-projection after rec')
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
        self.untilSTOPbtn = QtGui.QRadioButton('Run until STOP')
        self.untilSTOPbtn.clicked.connect(self.untilStop)
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
        recGrid.addWidget(self.showZgraph, 1, 0)
        recGrid.addWidget(self.showZproj, 1, 4)
#        recGrid.addWidget(loadFolderButton, 1, 5)
#        recGrid.addWidget(openFolderButton, 1, 4)
        recGrid.addWidget(self.folderEdit, 2, 1, 1, 5)
        recGrid.addWidget(self.specifyfile, 3, 0, 1, 5)
        recGrid.addWidget(self.filenameEdit, 3, 2, 1, 4)
        recGrid.addWidget(self.specifyFrames, 4, 0, 1, 5)
        recGrid.addWidget(self.currentFrame, 4, 1)
        recGrid.addWidget(self.numExpositionsEdit, 4, 2)
#        recGrid.addWidget(QtGui.QLabel('File size'), 4, 3, 1, 2)
#        recGrid.addWidget(self.filesizeBar, 4, 4, 1, 2)
        recGrid.addWidget(self.specifyTime, 5, 0, 1, 5)
        recGrid.addWidget(self.currentTime, 5, 1)
        recGrid.addWidget(self.timeToRec, 5, 2)
        recGrid.addWidget(self.tRemaining, 5, 3, 1, 2)
#        recGrid.addWidget(self.progressBar, 5, 4, 1, 2)
        recGrid.addWidget(self.untilSTOPbtn, 6, 0, 1, 5)
        recGrid.addWidget(buttonWidget, 7, 0, 1, 0)

        recGrid.setColumnMinimumWidth(0, 70)
        recGrid.setRowMinimumHeight(6, 40)

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
            elif self.specifyTime.isChecked():
                self.specTime()
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

# Functions for changing between choosing frames or time or "Run until stop" when recording.
            
    def specFrames(self):
        
        self.numExpositionsEdit.setEnabled(True)
        self.timeToRec.setEnabled(False)
        self.filesizeBar.setEnabled(True)
        self.progressBar.setEnabled(True)
        self.rec_mode = 1
        self.filesizeupdate()
    
    def specTime(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(True)
        self.filesizeBar.setEnabled(True)
        self.progressBar.setEnabled(True)
        self.rec_mode = 2
        self.filesizeupdate()
        
    def untilStop(self):
        self.numExpositionsEdit.setEnabled(False)
        self.timeToRec.setEnabled(False)
        self.filesizeBar.setEnabled(False)
        self.progressBar.setEnabled(False)
        self.rec_mode = 3
            
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
            
    def getTimeOrFrames(self):
        
        if self.specifyFrames.isChecked():
            return int(self.numExpositionsEdit.text())
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
        self.main.AbortROI()
        attrs = self.main.tree.attrs()

        for laserControl in self.main.laserWidgets.controls:
            name = re.sub('<[^<]+?>', '', laserControl.name.text())
            attrs.append((name, laserControl.laser.power))

        for key in self.main.scanWidget.scan_par_values:
            attrs.append((key, self.main.scanWidget.scan_par_values[key]))    
            
        attrs.append(('Scan mode', self.main.scanWidget.Scan_Mode.currentText()))
        attrs.append(('True_if_scanning', self.main.scanWidget.scanRadio.isChecked())) 
        
        for key in self.main.scanWidget.pixel_par_values:
            attrs.append((key, self.main.scanWidget.pixel_par_values[key]))
            
        attrs.extend([('element_size_um', [1, 0.066, 0.066]),
                      ('Date', time.strftime("%Y-%m-%d")),
                      ('Saved at', time.strftime("%H:%M:%S")),
                      ('NA', 1.42)])
            
        return attrs

    def snapHDF(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

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
            tiff.imsave(savename, self.main.latest_image.astype(np.uint16),
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
        nframe = self.worker.frames_recorded
        rSecs = self.getTimeOrFrames() - eSecs
        rText = '{}'.format(datetime.timedelta(seconds=max(0, rSecs)))
        self.tRemaining.setText(rText)
        self.currentFrame.setText(str(nframe) + ' /')
        self.currentTime.setText(str(int(eSecs)) + ' /')
        self.progressBar.setValue(100*(1 - rSecs / (eSecs + rSecs)))
#        self.main.img.setImage(self.worker.liveImage, autoLevels=False)

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
#                self.main.liveviewStop() # Stops liveview from updating

                self.savename = (os.path.join(folder, self.getFileName()) + '_rec.hdf5') # Sets name for final output file
                self.savename = guitools.getUniqueName(self.savename) # If same  filename exists it is appended by (1) or (2) etc.
                self.startTime = ptime.time() # Saves the time when started to calculate remaining time.

                self.worker = RecWorker(self.main.orcaflash, self.rec_mode, self.getTimeOrFrames(), self.main.shape, self.main.lvworker,  #Creates an instance of RecWorker class.
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
            self.main.orcaflash.stopAcquisition()   # To avoid camera overwriting buffer while saving recording

# Function called when recording finishes to reset relevent parameters.

    def endRecording(self):
        if self.showZgraph.checkState():
            plt.plot(self.worker.z_stack)
        if self.showZproj.checkState():
            plt.imshow(self.worker.Z_projection, cmap='gray')
        self.recordingThread.terminate()
        
        self.writable = True
        self.readyToRecord = True
        self.recButton.setText('REC')
        self.recButton.setChecked(False)
        self.main.tree.writable = True
        self.main.lvworker.reset() # Same as done in Liveviewrun()
        self.main.orcaflash.startAcquisition()
        self.main.liveviewButton.setEnabled(True)
        self.progressBar.setValue(0)
        self.currentTime.setText('0 /')
        self.currentFrame.setText('0 /')


class RecWorker(QtCore.QObject):

    updateSignal = QtCore.pyqtSignal()
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, orcaflash, rec_mode, timeorframes, shape, lvworker, t_exp, savename, dataname, attrs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orcaflash = orcaflash
        self.rec_mode = rec_mode  # 1=frames, 2=time, 3=until stop
        print(self.rec_mode)
        self.timeorframes = timeorframes #Nr of seconds or frames to record depending on bool_ToF.
        self.shape = shape # Shape of one frame
        self.max_frames = np.floor(6000000000 / self.orcaflash.frame_bytes) #Max frames in 6 GB memory
        self.lvworker = lvworker
        self.t_exp = t_exp
        self.savename = savename
        self.dataname = dataname
        self.attrs = attrs
        self.pressed = True


    def start(self):
        #Set initial values
        self.timerecorded = 0
        self.frames_recorded = 0
        self.buffer_size = self.orcaflash.number_image_buffers

        # Initiate data-writing process
        datashape = (self.max_frames, self.shape[1], self.shape[0])
        self.queue = Queue()
        pipe_recv, self.pipe_send = Pipe()
        self.write_process = Process(target=write_data, args=(datashape, self.dataname, self.savename, self.attrs, pipe_recv))
        self.write_process.start()
        
        #Find what the index of the first recorded frame will be
        last_aq = self.lvworker.f_ind
        start_f = last_aq + 1 # index of first frame is one more then provious frame.
            
        self.starttime = time.time()
     
        """ Main loop for waiting until recording is finished and sending update signal
        self.rec_mode determins how length of recording is set."""
        self.last_aq = last_aq
        self.last_saved = start_f - 1
        if self.rec_mode == 1:
            self.pkgs_sent = 0
            while self.frames_recorded < self.timeorframes and self.pressed:
                self.frames_recorded = self.lvworker.f_ind - start_f
                self.save_frames()  
                
        elif self.rec_mode == 2:
            self.pkgs_sent = 0
            while self.timerecorded < self.timeorframes and self.pressed:
                self.timerecorded = time.time() - self.starttime
                self.save_frames()  
        else:
            self.pkgs_sent = 0
            while self.pressed or (not self.last_saved == self.last_saved):       
                self.timerecorded = time.time() - self.starttime
                self.save_frames()
                
                
        t = time.time()
        self.pipe_send.send('Finish')        
        self.pipe_send.recv()
        self.write_process.join()
        print('Process joined, first frame index was: ', start_f, 'Last frame index was: ', self.next_f - 1, 'Packages sent was (excl. "Finish"): ', self.pkgs_sent)
        self.pipe_send.close()
        self.doneSignal.emit()
        
    def save_frames(self):
        self.last_aq = self.lvworker.f_ind
        while (self.last_saved == self.last_aq and self.pressed): #True if next_f is one "ahead" of camera f_ind.
            time.sleep(0.001) #Gives time for liveview thread to access memory and keep liveview responsive (somehow...?)
        
        if not self.last_saved == self.last_aq:
            if self.last_aq > self.last_saved:
                f_range = range(self.last_saved + 1, self.last_aq + 1)
            else:
                f_range = np.append(range(self.last_saved + 1, buffer_size), range(self.last_aq + 1))
                
            
                        
            self.orcaflash.hcam_data[self.next_f].getData()
            
            self.pipe_send.recv()
            print('After recieve')
            self.pipe_send.send(f)
            print('After send')
            self.pkgs_sent = self.pkgs_sent + 1
            self.next_f = np.mod(self.next_f + 1, self.buffer_size) # Mod to make it work if image buffer is circled
            self.updateSignal.emit() 
        


class FileWarning(QtGui.QMessageBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    def __init__(self, orcaflash, *args, **kwargs):
        super().__init__(*args, **kwargs)

        BinTip = ("Sets binning mode. Binning mode specifies if and how many \n"
                    "pixels are to be read out and interpreted as a single pixel value.")
                    

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': orcaflash.camera_id},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                      {'name': 'Binning', 'type': 'list', 
                                  'values': [1, 2, 4], 'tip': BinTip},
{'name': 'Mode', 'type': 'list', 'values': ['Full Widefield', 'Full chip', 'Minimal line', 'Custom']},
{'name': 'X0', 'type': 'int', 'value': 0, 'limits': (0, 2044)},
{'name': 'Y0', 'type': 'int', 'value': 0, 'limits': (0, 2044)},
{'name': 'Width', 'type': 'int', 'value': 2048, 'limits': (1, 2048)},
{'name': 'Height', 'type': 'int', 'value': 2048, 'limits': (1, 2048)}, 
                                  {'name': 'Apply', 'type': 'action'},
{'name': 'New ROI', 'type': 'action'}, {'name': 'Abort ROI', 'type': 'action', 'align': 'right'}]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Set exposure time', 'type': 'float',
                       'value': 0.03, 'limits': (0,
                                                9999),
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
                       'values': ['Internal trigger', 'External "Start-trigger"', 'External "frame-trigger"'],
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
#       WARNING: If Apply and New ROI button are included here they will emit status changed signal
        # and their respective functions will be called... -> problems.
        
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
    
    def __init__(self, main, orcaflash, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = main
        self.orcaflash = orcaflash
        self.running = False
        self.f_ind = -1
        self.mem = 0  # Memory variable to keep track of if update has been run twice in a row with camera trigger source as internal trigger
                        # If so the GUI trigger mode should also be set to internal trigger. Happens when using external start tigger.
    def run(self):
        
        self.vtimer = QtCore.QTimer()
        self.vtimer.timeout.connect(self.update)
        self.running = True
        self.f_ind = -1
        self.vtimer.start(30)
        print('f_ind when started = ',self.f_ind)
        
    def update(self):

        if self.running:
            self.f_ind = self.orcaflash.newFrames()[-1]
#            print('f_ind = :', self.f_ind)
            frame = self.orcaflash.hcam_data[self.f_ind].getData()
            self.image = np.reshape(frame, (self.orcaflash.frame_x, self.orcaflash.frame_y), 'F')
            self.main.latest_image = self.image
            trigger_source = self.orcaflash.getPropertyValue('trigger_source')[0]
            if trigger_source == 1:
                if self.mem == 1:
                    self.main.trigsourceparam.setValue('Internal trigger')
                    self.mem = 0
                else:
                    self.mem = 1

        
    def stop(self):
        if self.running:
            self.running = False
            print('Acquisition stopped')
        else:
            print('Cannot stop when not running (from LVThread)')
            
    def reset(self):
        self.f_ind = -1
        print('LVworker reset, f_ind = ', self.f_ind)

# The main GUI class.



class TormentaGUI(QtGui.QMainWindow):

    liveviewStarts = QtCore.pyqtSignal()
    liveviewEnds = QtCore.pyqtSignal()

    def __init__(self, bluelaser, bluelaser2, greenlaser, violetlaser, uvlaser, scanZ, daq, orcaflash,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.orcaflash = orcaflash
        self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_polarity', 2))
        self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_global_exposure', 5)) # 3:DELAYED, 5:GLOBAL RESET
        self.changeParameter(lambda: self.orcaflash.setPropertyValue('defect_correct_mode', 2)) # 1:OFF, 2:ON
#        self.orcaflash.setPropertyValue('readout_speed', 1)
#        self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_mode', 6))
        self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_active', 2)) # 1: EGDE, 2: LEVEL, 3:SYNCHREADOUT
        self.shape = (self.orcaflash.getPropertyValue('image_height')[0], self.orcaflash.getPropertyValue('image_width')[0])
        self.frameStart = (0, 0)
        self.scanZ = scanZ
        self.daq = daq
        self.nidaq = libnidaqmx.Device('Dev1')
        self.latest_image = np.zeros(self.shape)
        
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
        self.AbortROIParam = self.framePar.param('Abort ROI')
        self.applyParam.sigStateChanged.connect(self.applyfcn)  #WARNING: This signal is emitted whenever anything about the status of the parameter changes eg is set writable or not.
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
        self.RealExpPar.setOpts(decimals = 5)
        self.setExposure()    # Set default values
        
        #Acquisition signals
        acquisParam = self.tree.p.param('Acquisition mode')
        self.trigsourceparam = acquisParam.param('Trigger source')
        self.trigsourceparam.sigValueChanged.connect(self.ChangeTriggerSource)

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
        
        self.alignmentON = False
        
        # RESOLFT rec
        
        self.resolftRecButton = QtGui.QPushButton('RESOLFT REC')
        self.resolftRecButton.setStyleSheet("font-size:18px")
        self.resolftRecButton.clicked.connect(self.resolftRec)
        
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
        self.viewCtrlLayout.addWidget( self.resolftRecButton, 1, 0, 1, 3)
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
                                color='y', scaleSnap=True, translateSnap=True)
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

        # Initial camera configuration taken from the parameter tree
        self.orcaflash.setPropertyValue('exposure_time', self.expPar.value())
        self.adjustFrame()

        # Dock widget
        dockArea = DockArea()

        laserDock = Dock("Laser Control", size=(1, 1))
        self.lasers = (bluelaser, bluelaser2, greenlaser, violetlaser, uvlaser)
        self.laserWidgets = lasercontrol.LaserWidget(self.lasers, self.daq)
        laserDock.addWidget(self.laserWidgets)
        dockArea.addDock(laserDock)

        scanDock = Dock('Scan')
        self.scanWidget = Scan.ScanWidget(self.nidaq)
        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock)
        
#        # Console widget
#        consoleDock = Dock("Console", size=(600, 200))
#        console = ConsoleWidget(namespace={'pg': pg, 'np': np})
#        consoleDock.addWidget(console)
#        dockArea.addDock(consoleDock, 'above', scanDock)
#        
        # Line Alignment Tool
        alignmentDock = Dock("Alignment Tool", size=(50,50))
        self.alignmentWidget = QtGui.QWidget()
        alignmentDock.addWidget(self.alignmentWidget)
        dockArea.addDock(alignmentDock)
        
        alignmentLayout = QtGui.QGridLayout()
        self.alignmentWidget.setLayout(alignmentLayout)
        
        self.angleLabel = QtGui.QLabel('Line Angle')
        self.angleEdit = QtGui.QLineEdit('30')
        self.alignmentLineMakerButton = QtGui.QPushButton('Make Alignment Line')
        self.angle = np.float(self.angleEdit.text())
            
        self.alignmentLineMakerButton.clicked.connect(self.alignmentToolAux)
        
        self.alignmentCheck = QtGui.QCheckBox('Show Alignment Tool')
        
        alignmentLayout.addWidget(self.angleLabel, 0, 0, 1, 1)
        alignmentLayout.addWidget(self.angleEdit, 0, 1, 1, 1)
        alignmentLayout.addWidget(self.alignmentLineMakerButton, 1, 0, 1, 1)
        alignmentLayout.addWidget(self.alignmentCheck, 1, 1, 1, 1)
        
##         Z Align widget
        ZalignDock = Dock("Axial Alignment Tool", size=(1, 1))
        self.ZalignWidget = align.AlignWidgetAverage(self)
        ZalignDock.addWidget(self.ZalignWidget)
        dockArea.addDock(ZalignDock, 'above', scanDock)
        
        ##         Z Align widget
        RotalignDock = Dock("Rotational Alignment Tool", size=(1, 1))
        self.RotalignWidget = align.AlignWidgetXYProject(self)
        RotalignDock.addWidget(self.RotalignWidget)
        dockArea.addDock(RotalignDock, 'above', ZalignDock)



##         Focus lock widget
#        focusDock = Dock("Focus Control", size=(1, 1))
##        self.focusWidget = FocusWidget(DAQ, scanZ, self.recWidget)
#        self.focusWidget = focus.FocusWidget(scanZ, self.recWidget)
##        self.focusThread = QtCore.QThread()
##        self.focusWidget.moveToThread(self.focusThread)
##        self.focusThread.started.connect(self.focusWidget)
##        self.focusThread.start()
#        focusDock.addWidget(self.focusWidget)
#        dockArea.addDock(focusDock)
#        
#        
#        # Signal generation widget
#        signalDock = Dock('Signal Generator')
#        self.signalWidget = SignalGen.SigGenWidget(self.nidaq)
#        signalDock.addWidget(self.signalWidget)
#        dockArea.addDock(signalDock, 'above', laserDock)
        
        

        self.setWindowTitle('Tempesta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(0, 350)
        layout.setColumnMinimumWidth(2, 600)
        layout.setColumnMinimumWidth(3, 200)
        layout.setRowMinimumHeight(1, 550)
        layout.setRowMinimumHeight(2, 100)
        layout.setRowMinimumHeight(3, 300)
        layout.addWidget(self.presetsMenu, 0, 0)
        layout.addWidget(self.loadPresetButton, 0, 1)
        layout.addWidget(cameraWidget, 1, 0, 1, 2)
        layout.addWidget(self.viewCtrl, 2, 0, 1, 2)
        layout.addWidget(self.recWidget, 3, 0, 2, 2)
        layout.addWidget(imageWidget, 0, 2, 5, 1)
        layout.addWidget(dockArea, 0, 3, 5, 1)

        layout.setRowMinimumHeight(2, 40)
        layout.setColumnMinimumWidth(2, 1000)
        
        
    def testfunction(self):
        print('In testfunction ie called from frame mode changed signal')
        self.updateFrame()
        
    def applyfcn(self):
        print('Apply pressed')
        self.adjustFrame()

    def mouseMoved(self, pos):
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            x, y = int(mousePoint.x()), int(self.shape[1] - mousePoint.y())
            self.cursorPos.setText('{}, {}'.format(x, y))

    def flipperInPath(self, value):
        pass
#        self.flipperButton.setChecked(not(value))
#        self.daq.flipper = value


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


    def ChangeTriggerSource(self):
        
        if self.trigsourceparam.value() == 'Internal trigger':
            print('Changing to internal trigger')
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_source', 1))
#            self.RealExpPar.Enable(True)
#            self.EffFRPar.Enable(True)
            
        elif self.trigsourceparam.value() == 'External "Start-trigger"':
            print('Changing to external start trigger')
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_source', 2))
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_mode', 6))
            print(self.orcaflash.getPropertyValue('trigger_mode'))
#            self.RealExpPar.Enable(False)
#            self.EffFRPar.Enable(False)
        
        elif self.trigsourceparam.value() == 'External "frame-trigger"':
            print('Changing to external trigger')
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_source', 2))
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_mode', 1))
            
        else:
            pass
                

    def updateLevels(self, image):
        std = np.std(image)
        self.hist.setLevels(np.min(image) - std, np.max(image) + std)

    def setBinning(self):
        
        """Method to change the binning of the captured frame"""

        binning = str(self.binPar.value())

        binstring = binning+'x'+binning
        coded = binstring.encode('ascii')
        

        self.changeParameter(lambda: self.orcaflash.setPropertyValue('binning', coded))


            
        
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
        print('Exp time set to:', self.orcaflash.getPropertyValue('exposure_time'))
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
        """Method to crop the frame read out by Orcaflash """
#       Round to closest "divisable by 4" value.
        t1 = time.time()
        print('time in beginning of cropOrca = ', t1)
        self.orcaflash.setPropertyValue('subarray_vpos', 0)
        self.orcaflash.setPropertyValue('subarray_hpos', 0)
        self.orcaflash.setPropertyValue('subarray_vsize', 2048)
        self.orcaflash.setPropertyValue('subarray_hsize', 2048)

 
        vpos = int(4*np.ceil(vpos/4))
        hpos = int(4*np.ceil(hpos/4))
        vsize = int(min(2048 - vpos, 4*np.ceil(vsize/4)))
        hsize = int(min(2048 - hpos, 4*np.ceil(hsize/4)))

        self.orcaflash.setPropertyValue('subarray_vsize', vsize)
        self.orcaflash.setPropertyValue('subarray_hsize', hsize)
        self.orcaflash.setPropertyValue('subarray_vpos', vpos)
        self.orcaflash.setPropertyValue('subarray_hpos', hpos)
        
        self.frameStart = (hpos, vpos) # This should be the only place where self.frameStart is changed
        self.shape = (hsize, vsize)     # Only place self.shape is changed
        print('time after beginning of cropOrca= ', time.time())
        print('orca has been cropped to: ', vpos, hpos, vsize, hsize)

    def adjustFrame(self):
        """ Method to change the area of the sensor to be used and adjust the
        image widget accordingly. It needs a previous change in self.shape
        and self.frameStart)
        """

        binning = self.binPar.value()

        self.changeParameter(lambda: self.cropOrca(binning*self.X0par.value(), binning*self.Y0par.value(), binning*self.Widthpar.value(), self.Heightpar.value()))
        
        self.updateTimings()
        self.recWidget.filesizeupdate()
        self.ROI.hide()

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        print('Update frame called')
        frameParam = self.tree.p.param('Image frame')
        if frameParam.param('Mode').value() == 'Custom':
            self.X0par.setWritable(True)
            self.Y0par.setWritable(True)
            self.Widthpar.setWritable(True)
            self.Heightpar.setWritable(True)

#            if not(self.customFrameLoaded):
#            ROIsize = (int(0.2 * self.vb.viewRect().width()), int(0.2 * self.vb.viewRect().height()))
            ROIsize = (int(0.2 * self.vb.viewRect().height()), int(0.2 * self.vb.viewRect().height()))
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
            
        else:
            self.X0par.setWritable(False)
            self.Y0par.setWritable(False)
            self.Widthpar.setWritable(False)
            self.Heightpar.setWritable(False)

            
            if frameParam.param('Mode').value() == 'Full Widefield':
                self.X0par.setValue(678)
                self.Y0par.setValue(662)  # 7 november 2016
                self.Widthpar.setValue(800)
                self.Heightpar.setValue(800)
                self.adjustFrame()

                self.ROI.hide()


            elif frameParam.param('Mode').value() == 'Full chip':
                print('Full chip')
                self.X0par.setValue(0)
                self.Y0par.setValue(0)
                self.Widthpar.setValue(2048)
                self.Heightpar.setValue(2048)
                self.adjustFrame()

                self.ROI.hide()
                
            elif frameParam.param('Mode').value() == 'Minimal line':
                print('Full chip')
                self.X0par.setValue(0)
                self.Y0par.setValue(1020)
                self.Widthpar.setValue(2048)
                self.Heightpar.setValue(8)
                self.adjustFrame()

                self.ROI.hide()




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

        self.X0par.setValue(self.frameStart[0] + int(self.ROI.pos()[0]))
        self.Y0par.setValue(self.frameStart[1] + int(self.ROI.pos()[1]))

        self.Widthpar.setValue(int(self.ROI.size()[0])) # [0] is Width
        self.Heightpar.setValue(int(self.ROI.size()[1])) # [1] is Height
        
        
    def AbortROI(self):
        
        self.ROI.hide()
        
        self.X0par.setValue(self.frameStart[0])
        self.Y0par.setValue(self.frameStart[1])

        self.Widthpar.setValue(self.shape[0]) # [0] is Width
        self.Heightpar.setValue(self.shape[1]) # [1] is Height    

    def updateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
#        timings = self.orcaflash.getPropertyValue('exposure_time') 
#        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        self.RealExpPar.setValue(self.orcaflash.getPropertyValue('exposure_time')[0])
        self.FrameInt.setValue(self.orcaflash.getPropertyValue('internal_frame_interval')[0])
        self.ReadoutPar.setValue(self.orcaflash.getPropertyValue('timing_readout_time')[0])
        self.EffFRPar.setValue(self.orcaflash.getPropertyValue('internal_frame_rate')[0])
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
            
# Threading below  is done in this way since making LVThread a QThread resulted in QTimer
# not functioning in the thread. Image is now also saved as latest_image in 
# TormentaGUI class since setting image in GUI from thread resultet in 
# issues when interacting with the viewbox from GUI. Maybe due to 
# simultaious manipulation of viewbox from GUI and thread. 

    def liveviewStart(self):

#        self.orcaflash.startAcquisition()
#        time.sleep(0.3)
#        time.sleep(np.max((5 * self.t_exp_real.magnitude, 1)))
        self.updateFrame()
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        self.recWidget.readyToRecord = True
        self.lvworker = LVWorker(self, self.orcaflash)
        self.lvthread = QtCore.QThread()
        self.lvworker.moveToThread(self.lvthread)
        self.lvthread.started.connect(self.lvworker.run)
        self.lvthread.start()
        self.viewtimer.start(30)
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
        self.lvworker.stop()
        self.lvthread.terminate()
        self.viewtimer.stop()
        self.recWidget.readyToRecord = False

        # Turn off camera, close shutter
        self.orcaflash.stopAcquisition()
        self.img.setImage(np.zeros(self.shape), autoLevels=False)
        del self.lvthread

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
#       
        self.lvworker.reset() # Needed if parameter is changed during liveview since that causes camera to start writing to buffer place zero again.      
        self.orcaflash.startAcquisition()
#        time.sleep(0.3)
#        self.viewtimer.start(0)
#        self.lvthread.run()
#        self.lvthread.start()
    
    def liveviewPause(self):
        
#        self.lvworker.stop()
#        self.viewtimer.stop()
        self.orcaflash.stopAcquisition()

    def updateView(self):
        """ Image update while in Liveview mode
        """

        self.img.setImage(self.latest_image, autoLevels=False, autoDownsample = False) 
        
        if self.alignmentON == True:
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
        except:
            pass
        
        pen = pg.mkPen(color=(255, 255, 0), width=0.5,
                             style=QtCore.Qt.SolidLine, antialias=True)
        self.alignmentLine = pg.InfiniteLine(pen=pen, angle=angle, movable=True)
        self.alignmentON = True
        
    def resolftRec(self):
        self.trigsourceparam.setValue('External "frame-trigger"')
        self.recWidget.untilSTOPbtn.setChecked(True)
        self.recWidget.untilStop()
        self.recWidget.recButton.setChecked(True)
        self.recWidget.startRecording()
        self.laserWidgets.DigCtrl.DigitalControlButton.setChecked(True)
        self.laserWidgets.DigCtrl.GlobalDigitalMod()


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
        self.ZalignWidget.closeEvent(*args, **kwargs)
        self.RotalignWidget.closeEvent(*args, **kwargs)        
        self.scanWidget.closeEvent(*args, **kwargs)
#        self.focusWidget.closeEvent(*args, **kwargs)
#        self.signalWidget.closeEvent(*args, **kwargs)

        super().closeEvent(*args, **kwargs)
