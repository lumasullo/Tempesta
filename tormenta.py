# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: federico
"""
from pyqtgraph.Qt import QtGui

def main():

    from control import control
    import control.instruments as instruments    
    
    app = QtGui.QApplication([])

#TO DO: create an instruments.Camera(hamamatsu) or something similar

#    with instruments.Camera('hamamatsu.hamamatsu_camera.HamamatsuCameraMR') as orcaflash, \
    with instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM12') as bluelaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM13') as bluelaser2, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM5') as greenlaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM7') as violetlaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM10') as uvlaser, \
         instruments.DAQ() as daq, instruments.ScanZ(12) as scanZ:
        #instruments.Camera('andor.ccd.CCD') as andor, \

         
#for now, bluelaser is the 488nm laser, greenlaser is the 405nm laser and redlaser is the 355nm laser
        orcaflash = instruments.Camera()
        print(bluelaser.idn)
        print(bluelaser2.idn)
        print(greenlaser.idn)        
        print(violetlaser.idn)
        print(uvlaser.idn)
        print(daq.idn)
        print('Prior Z stage')

        win = control.TormentaGUI(bluelaser, bluelaser2, greenlaser, violetlaser, uvlaser,
                                  scanZ, daq, orcaflash)
        win.show()

        app.exec_()


def analysisApp():

    from analysis import analysis

    app = QtGui.QApplication([])

    win = analysis.AnalysisWidget()
    win.show()

    app.exec_()

    
