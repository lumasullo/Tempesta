# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:39:37 2016

@author: aurelien.barbotin
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import control.instruments as instruments
from control.SLM import Mask
from pyqtgraph.parametertree import Parameter, ParameterTree

#Width and height of the SLM which can change from one device to another:
m=600
n=792

class slmWidget(QtGui.QFrame):
    """Class creating the window for the control of the SLM???"""

    def __init__(self, slm, *args, **kwargs):

        super().__init__(*args, **kwargs)
#        self.setMinimumSize(2, 350)
        
        #Get the parameters from the tree
        self.tree=SLMParamTree()
        self.RPar=self.tree.p.child("R")
        self.sigmaPar=self.tree.p.param("sigma")
        self.anglePar=self.tree.p.param("angle")
        self.lbdPar=self.tree.p.param('lambda depletion: (nm)')

        self.slm=slm
        self.mask=Mask.Helix_Hat(m,n,self.lbdPar.value(),self.RPar.value(),self.sigmaPar.value())
        self.mask.tilt(self.anglePar.value())
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
        image=image.transpose()
        self.img.setImage(image, autoLevels=False, autoDownsample = False) 
        self.vb.addItem(self.img)
        self.slm.updateArray(self.mask)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(imageWidget, 0, 0, 3, 4)
        grid.addWidget(self.updateButton, 4, 2,1,1)        
        grid.addWidget(self.tree, 3, 0,1,1)

        grid.setColumnMinimumWidth(1, 100)
        grid.setColumnMinimumWidth(2, 40)
        grid.setColumnMinimumWidth(0, 100)

    def update(self):
        print("update in SLMWidget called and R vaults now:",self.RPar.value())

    def apply(self):
        """Applies a configuration to the SLM by changing the mask displayed"""
        self.mask = Mask.Helix_Hat(m,n,self.lbdPar.value(),self.RPar.value(),self.sigmaPar.value())
        self.mask.tilt(self.anglePar.value())
        #Get the image of the new mask and displays it
        image=self.mask.img.transpose()
        self.img.setImage(image, autoLevels=False, autoDownsample = False)
        self.slm.updateArray(self.mask)
        print("Parameters applied")
        
    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)

    def refresh(self):
        """closes and reopens the SLM in case of problem: sleep mode..."""
        return

class SLMParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the SLM during imaging
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params = [
{'name': 'R', 'type': 'int', 'value': 70, 'limits': (0, 600)},
{'name': 'sigma', 'type': 'float', 'value': 100, 'limits': (0.001, 10**6 )},
{'name': 'angle', 'type': 'float', 'value': 0.1, 'limits': (0, 0.3 )},
{'name': 'lambda depletion: (nm)', 'type': 'int', 'value': 561, 'limits': (0, 1200 )},
{'name': 'Apply', 'type': 'action'}
]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True
        

if __name__ == '__main__':

    app = QtGui.QApplication([])
#instruments.ScanZ(12)
    
    with instruments.SLM() as z:
        win = slmWidget(z)
        win.show()
        app.exec_()
    