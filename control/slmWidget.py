# -*- coding: utf-8 -*-

import numpy as np
#import control.slmpy as slmpy
import control.instruments as instruments
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import control.SLM.Mask as Mask
from pyqtgraph.parametertree import Parameter, ParameterTree
import pickle

#Width and height of the SLM which can change from one device to another:
m=600
n=792

class slmWidget(QtGui.QFrame):
    """Class creating a GUI to control the phase pattern displayed by the SLM. In this version, it is optimized to 
    display and address 2 masks independently. The whole image is separated in two: one left part and right part. 
    One part is selected at a time and can be controlled with the arrows from *ArrowsControl*.
    :param SLMdisplay slm: instance of a second monitor generatd via slmpy. 
    Communication with the SLM is initiated when Tempesta is started along with all other instruments
        """
    def __init__(self, slm, *args, **kwargs):

        super().__init__(*args, **kwargs)
#        self.setMinimumSize(2, 350)
        
        #Get the parameters from the tree
        self.tree=SLMParamTree()
        self.RPar=self.tree.p.child("R")
        self.sigmaPar=self.tree.p.param("sigma")
        self.anglePar=self.tree.p.param("angle")
        self.lbdPar=self.tree.p.param('lambda depletion: (nm)')
        self.helix_rotPar=self.tree.p.param("helix clock rotation")

        self.slm=slm
        self.mask=Mask.Helix_Hat(m,n,self.lbdPar.value(),self.RPar.value(),self.sigmaPar.value())
        self.mask.tilt(self.anglePar.value())
        self.left_center=(0,0)
        self.right_center=(0,0)
        
        #Indicates wether each side of the mask is actually displaying a mask or not
        self.black_left=False
        self.black_right=False
        
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.updateButton = QtGui.QPushButton('Update')
        self.updateButton.setCheckable(True)
        self.updateButton.clicked.connect(self.update)
        self.updateButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)

        self.applyPar = self.tree.p.param('Apply')
        self.applyPar.sigStateChanged.connect(self.apply)
        
        #Widget displaying the phase pattern displayed on the SLM                        
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)     
        self.img=pg.ImageItem()
        image=self.mask.img
        image=np.fliplr(image.transpose())
        self.img.setImage(image, autoLevels=False, autoDownsample = True,autoRange=True) 
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.slm.updateArray(self.mask)

        self.arrowsModule=ArrowsControl()
        
        #Link between the buttons in the arrow module and the functions to control the SLM
        self.arrowsModule.upButton.clicked.connect(self.upClicked)
        self.arrowsModule.downButton.clicked.connect(self.downClicked)
        self.arrowsModule.leftButton.clicked.connect(self.leftClicked)
        self.arrowsModule.rightButton.clicked.connect(self.rightClicked)
        self.arrowsModule.saveButton.clicked.connect(self.save)
        self.arrowsModule.loadButton.clicked.connect(self.loadParam)
        self.arrowsModule.blackButton.clicked.connect(self.setBlack)
        
        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(imageWidget, 0, 0, 4, 3)    
        grid.addWidget(self.tree, 4, 0,1,1)
        grid.addWidget(self.arrowsModule,4,1,1,1)

        grid.setColumnMinimumWidth(1, 100)
        grid.setColumnMinimumWidth(2, 40)
        grid.setColumnMinimumWidth(0, 100)
        grid.setRowMinimumHeight(0,75)

    def upClicked(self):
        """Moves the current Mask up"""
        self.moveMask(-1*self.arrowsModule.increment.value(),0)
        
        self.left_center=self.mask.left_center
        self.right_center=self.mask.right_center
        
    def downClicked(self):
        """Moves the current Mask down"""
        self.moveMask(self.arrowsModule.increment.value(),0)
        
        self.left_center=self.mask.left_center
        self.right_center=self.mask.right_center
        
    def leftClicked(self):     
        """Moves the current Mask left"""
        self.moveMask(0,-1*self.arrowsModule.increment.value())
        
        self.left_center=self.mask.left_center
        self.right_center=self.mask.right_center

        
    def rightClicked(self):
        """Moves the current Mask right"""
        self.moveMask(0,self.arrowsModule.increment.value()) 
        
        self.left_center=self.mask.left_center
        self.right_center=self.mask.right_center

    def moveMask(self,x,y):
        """Sends instruction to both the SLM and the display to move the corresponding mask
        when one arrow is pressed.
        
        :param int x: new x position of the center of the Mask
        :param int y: new y position of the center of the Mask"""
        
        if(str(self.arrowsModule.maskMenu.currentText())=="Left"):
            self.mask.moveLeft(x,y)
        elif(self.arrowsModule.maskMenu.currentText()=="Right"):
            self.mask.moveRight(x,y)
        else:
            print("échec Barbotin, c-est mal problematise, mal introduit, mal construit")
        self.mask.update()
        self.update()
    
    def setBlack(self):
        """Sets the current mask to a black (null phase) Mask. Useful to check the masks one by one"""
        if(str(self.arrowsModule.maskMenu.currentText())=="Left"):
            self.mask.left.setBlack()
            self.black_left = True
        elif(self.arrowsModule.maskMenu.currentText()=="Right"):
            self.mask.right.setBlack()
            self.black_right = True
        else:
            print("échec Barbotin, c-est mal problematise, mal introduit, mal construit")
        self.mask.update()
        self.update()
        
    def apply(self):
        """Applies a configuration to the SLM and changes the mask displayed"""
        self.mask=Mask.Helix_Hat(m,n,self.lbdPar.value(),self.RPar.value(),\
            self.sigmaPar.value(),self.left_center,self.right_center,self.helix_rotPar.value())
        self.mask.tilt(self.anglePar.value())
#        if(self.black_left):
#            self.mask.left.setBlack()
#        if(self.black_right):
#            self.mask.right.setBlack()
#        self.mask.update()
        self.update()
        
    def update(self):
        """When any parameter changes, sends the new image to the SLM and the display"""
        #Changing the orientation of image so they have the same orientation on the slm and on the screen
        image=self.mask.img.transpose()
        image=np.fliplr(image)
        self.img.setImage(image, autoLevels=False, autoDownsample = False)
        self.slm.updateArray(self.mask)
                
    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)

    def save(self):
        """saves the current SLM configuration, in particular the position of the Masks.
        The informations are stored in a file 'informations.bbn' (arbitrary extension) with the module pickle. """
        state=self.tree.p.saveState()
        mask_state= {"left_center":self.mask.left_center,"right_center":self.mask.right_center}
        with open("informations.bbn","wb") as f:
            pickler=pickle.Pickler(f)
            pickler.dump(state)
            pickler.dump(mask_state)
        return
        
    def loadParam(self):
        """loads the parameters from a previous configuration"""
        with open('informations.bbn', 'rb') as f:
            depickler = pickle.Unpickler(f)
            state = depickler.load()
            mask_state=depickler.load()
            
        self.tree.p.restoreState(state)
        print("load: centers",mask_state)
        self.left_center = mask_state["left_center"]
        self.right_center = mask_state["right_center"]
        self.mask=Mask.Helix_Hat(m,n,self.lbdPar.value(),self.RPar.value(),\
            self.sigmaPar.value(),left_pos=self.left_center,right_pos=self.right_center)
        self.mask.tilt(self.anglePar.value())
        self.update()
        
        
class SLMParamTree(ParameterTree):
    """ Parameter Tree containing the different parameters for the SLM's phase masks. These parameters are:
    Radius (int): of circular phase masks, in pixels
    sigma (float): std of the incident gaussian beam, to determine the inner radius of a top-hat phase mask, in pixels.
    angle (float): in an off-axis configuration.
    lambda depletion (nm): the wavelength incident on the SLM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params = [
{'name': 'R', 'type': 'int', 'value': 70, 'limits': (0, 600)},
{'name': 'sigma', 'type': 'float', 'value': 100, 'limits': (0.001, 10**6 )},
{'name': 'angle', 'type': 'float', 'value': 0.1, 'limits': (0, 0.3)},
{'name': 'lambda depletion: (nm)', 'type': 'int', 'value': 561, 'limits': (0 , 1200 )},
{'name': 'helix clock rotation', 'type': 'bool', 'value': True},
{'name': 'Apply', 'type': 'action'}
]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True

class ArrowsControl(QtGui.QFrame):
    """This widget creates four buttons able to move a circular phase mask with a tunable number of pixels. 
    Useful to align the phase mask with the incident beam without touching any optics."""
    def __init__(self,*args,**kwargs):
        #Definition of the Widget to choose left or right part of the Mask
        super().__init__(*args,**kwargs)
        self.chooseInterface=QtGui.QWidget()
        self.chooseInterface_layout=QtGui.QGridLayout()
        self.chooseInterface.setLayout(self.chooseInterface_layout)
        
        self.maskMenu=QtGui.QComboBox()
        self.maskMenu.addItem("Left")
        self.maskMenu.addItem("Right")
        
        self.chooseInterface_layout.addWidget(QtGui.QLabel('Select part of the mask:'),0,0)
        self.chooseInterface_layout.addWidget(self.maskMenu,0,1)
        
        #Defining the part with only the arrows themselves
        self.arrows=QtGui.QFrame()
        self.arrow_layout=QtGui.QGridLayout()
        self.arrows.setLayout(self.arrow_layout)
        
        self.upButton = QtGui.QPushButton('up')
        self.upButton.setCheckable(False)
        self.upButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.upButton.setFixedSize(self.upButton.sizeHint())
        
        self.downButton = QtGui.QPushButton('down')
        self.downButton.setCheckable(False)
        self.downButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)   
        self.downButton.setFixedSize(self.upButton.sizeHint())     

        self.leftButton = QtGui.QPushButton('left')
        self.leftButton.setCheckable(False)
        self.leftButton.setFixedSize(self.upButton.sizeHint())

        self.rightButton = QtGui.QPushButton('right')
        self.rightButton.setCheckable(False)
        self.rightButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.rightButton.setFixedSize(self.upButton.sizeHint())

        #Widget to change manually the amount of deplacement induced by the arrows
        self.incrementWidget=QtGui.QWidget()
        self.incrementLayout=QtGui.QVBoxLayout()
        self.incrementWidget.setLayout(self.incrementLayout)
        
        label=QtGui.QLabel()
        label.setText("move(pix)")
        self.increment=QtGui.QSpinBox()        
        self.increment.setRange(1,500)
        self.increment.setValue(10)
        self.incrementLayout.addWidget(label)
        self.incrementLayout.addWidget(self.increment)
        
        self.saveButton=QtGui.QPushButton("Save")
        self.loadButton=QtGui.QPushButton("load")
        
        self.blackButton = QtGui.QPushButton("Black frame")        
        
        self.arrow_layout.addWidget(self.upButton, 1, 1)
        self.arrow_layout.addWidget(self.downButton,3,1)
        self.arrow_layout.addWidget(self.leftButton,2,0)
        self.arrow_layout.addWidget(self.rightButton,2,3)
        self.arrow_layout.addWidget(self.incrementWidget,2,1)
        
        self.arrow_layout.addWidget(self.saveButton,4,0)
        self.arrow_layout.addWidget(self.loadButton,4,1)
        self.arrow_layout.addWidget(self.blackButton,4,3)
        #Definition of the global layout:
        self.layout=QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        
        self.layout.addWidget(self.chooseInterface)
        self.layout.addWidget(self.arrows)
      
        

if __name__ == '__main__':
    app = QtGui.QApplication([])
    slm=1
    
    win = slmWidget(slm)
    win.show()
    app.exec_()
#    slm.close()
    