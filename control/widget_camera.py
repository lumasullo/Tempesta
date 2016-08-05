# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:48:15 2016

@author: aurelien.barbotin
"""
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui

class CameraWidget(QtGui.QFrame):
    """class used to handle the different methods and properties of a CMOS-like 
    camera used for imaging. It is typically used with an imageWidget"""
    
    def __init__(self,cam_instruments,imgWidget):
        super().__init__()
        self.camera=cam_instruments
        self.imageWidget=imgWidget
        #Camera and image Widget share the same ROI
        self.ROI= imgWidget.ROI
        self.ROI.sigRegionChangeFinished.connect(self.ROIchanged)
        
        self.tree = CamParamTree(self.camera)

        # Indicator for loading frame shape from a preset setting 
        # Currently not used.
        self.customFrameLoaded = False
        self.cropLoaded = False

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        cameraTitle = QtGui.QLabel('<h2><strong>Camera settings</strong></h2>')
        cameraTitle.setTextFormat(QtCore.Qt.RichText)
        cameraGrid = QtGui.QGridLayout()
        self.setLayout(cameraGrid)
        cameraGrid.addWidget(cameraTitle, 0, 0)
        cameraGrid.addWidget(self.tree, 1, 0)

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
        self.NewROIParam.sigStateChanged.connect(self.imageWidget.updateFrame)
        self.AbortROIParam.sigStateChanged.connect(self.imageWidget.AbortROI)


        
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
        
    def ChangeTriggerSource(self):
        
        if self.trigsourceparam.value() == 'Internal trigger':
            print('Changing to internal trigger')
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_source', 1))
#            self.RealExpPar.Enable(True)
#            self.EffFRPar.Enable(True)
            
        elif self.trigsourceparam.value() == 'External trigger':
            print('Changing to external trigger')
            self.changeParameter(lambda: self.orcaflash.setPropertyValue('trigger_source', 2))
#            self.RealExpPar.Enable(False)
#            self.EffFRPar.Enable(False)
            
        else:
            pass
        
    def setBinning(self):
        
        """Method to change the binning of the captured frame"""

        binning = str(self.binPar.value())

        binstring = binning+'x'+binning
        coded = binstring.encode('ascii')
        

        self.changeParameter(lambda: self.orcaflash.setPropertyValue('binning', coded))

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
        self.updateTimings()
        
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
            ROIsize = (int(0.2 * self.vb.viewRect().width()), int(0.2 * self.vb.viewRect().height()))
            ROIcenter = (int(self.vb.viewRect().center().x()), int(self.vb.viewRect().center().y()))
            ROIpos = (ROIcenter[0] - 0.5*ROIsize[0], ROIcenter[1] - 0.5*ROIsize[1])
            
#            try:
            self.ROI.setPos(ROIpos)
            self.ROI.setSize(ROIsize)
            self.ROI.show()

                
            self.ROIchanged()
            
        else:
            self.X0par.setWritable(False)
            self.Y0par.setWritable(False)
            self.Widthpar.setWritable(False)
            self.Heightpar.setWritable(False)

            
            if frameParam.param('Mode').value() == 'Full Widefield':
                self.X0par.setValue(600)
                self.Y0par.setValue(600)
                self.Widthpar.setValue(900)
                self.Heightpar.setValue(900)
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
        
    def updateLevels(self, image):
        std = np.std(image)
        self.hist.setLevels(np.min(image) - std, np.max(image) + std)
        
        
        
        
        
class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    def __init__(self, camera, *args, **kwargs):
        super().__init__(*args, **kwargs)

        BinTip = ("Sets binning mode. Binning mode specifies if and how many \n"
                    "pixels are to be read out and interpreted as a single pixel value.")
                    

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': camera.camera_id},
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
                       'values': ['Internal trigger', 'External trigger'],
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