# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:00:43 2016

@author: aurelien.barbotin
"""

import numpy as np


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import cv2

class WebcamManager(QtGui.QWidget):
    def __init__(self,main):
        super().__init__()
        self.webcam=cv2.VideoCapture(0)
        
        self.main=main
        self.frameStart = (0, 0)
        self.title=QtGui.QLabel()
        self.title.setText("Webcam")
        self.title.setStyleSheet("font-size:18px")

#        These attributes are the widgets which will be used in the main script
        self.imageWidget = pg.GraphicsLayoutWidget()
        
      # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('LIVEVIEW')
        self.liveviewButton.setStyleSheet("font-size:18px")
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.liveviewButton.clicked.connect(self.liveview)      #Link button click to funciton liveview
        self.liveviewButton.setEnabled(True)
        
        self.exposureButton = QtGui.QLineEdit('-15')        
        self.exposureLabel=QtGui.QLabel()
        self.exposureLabel.setText("exposure time:")
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)
        
        self.viewCtrlLayout = QtGui.QGridLayout()
        self.viewCtrlLayout.addWidget(self.title,0,0)
        self.viewCtrlLayout.addWidget(self.imageWidget,1,0,3,3)
        self.viewCtrlLayout.addWidget(self.liveviewButton, 5, 0, 1, 1)
        self.viewCtrlLayout.addWidget(self.exposureLabel,5,1,1,1)
        self.viewCtrlLayout.addWidget(self.exposureButton,5,2,1,1)
        self.setLayout(self.viewCtrlLayout)
        
        self.lvworker = LVWorker(self, self.webcam)
        self.exposureButton.textChanged.connect(self.lvworker.setExposure)
        
        self.connect(self.lvworker, QtCore.SIGNAL("newFrameToDisplay(PyQt_PyObject)"),self.setImage )

        # Image Widget
        self.vb = self.imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)        


    def setImage(self,array):
        self.img.setImage(array.astype(np.float))


    def liveviewKey(self):

        if self.liveviewButton.isChecked():
            self.liveviewStop()
            self.liveviewButton.setChecked(False)

        else:
            self.liveviewStart(True)
            self.liveviewButton.setChecked(True)

    # This is the function triggered by pressing the liveview button
    def liveview(self):
        """ Image live view
        """
        if self.liveviewButton.isChecked():
            self.liveviewStart()

        else:
            self.liveviewStop()

    def liveviewStart(self):
        self.lvworker.start()


    def liveviewStop(self):
        self.lvworker.stop()


    def updateView(self):
        """ Image update while in Liveview mode
        Not used anymore. Kept for archeologic purpose.
        """
        osef,frame=self.webcam.read()
        self.img.setImage(np.rot90(frame), autoLevels=False, autoDownsample = False) 

    def closeEvent(self, *args, **kwargs):
        self.webcam.release()
        
class LVWorker(QtCore.QThread):
    """Thread acquiring images from the webcam and sending it to the display widget"""
    def __init__(self, main, webcam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = main
        self.webcam = webcam
        self.running = False
    def run(self):
        
        self.vtimer = QtCore.QTimer()

        self.running = True
        import time
        while(self.running):
            osef,frame = self.webcam.read()
            frame = np.rot90(frame)
            self.emit(QtCore.SIGNAL("newFrameToDisplay(PyQt_PyObject)"),frame )
            time.sleep(0.01)

    def setExposure(self,value):
        """ Method to change the exposure time setting
        """
        try:
            exp_time=float(value)
        except:
            return
        self.webcam.set(15,exp_time)
        
    def stop(self):
        if self.running:
#            self.vtimer.stop()
            self.running = False
            print('Acquisition stopped')
        else:
            print('Cannot stop when not running (from LVThread)')
            
        
if __name__ == '__main__':
    app = QtGui.QApplication([])
    webm=WebcamManager("train")
    webm.show()
    app.exec_()