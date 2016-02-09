# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: federico
"""
from pyqtgraph.Qt import QtGui


def main():

    from control import control
    import control.instruments as instruments
    import hamamatsu.hamamatsu_camera as hm    
    
    app = QtGui.QApplication([])

#TO DO: create an instruments.Camera(hamamatsu) or something similar

    with instruments.Camera('andor.ccd.CCD') as andor, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM12') as bluelaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM11') as violetlaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM5') as uvlaser, \
         instruments.DAQ() as daq, instruments.ScanZ(12) as scanZ:

        orcaflash = hm.HamamatsuCameraMR(0)
        print(orcaflash.camera_model)
        print(andor.idn)
        print(bluelaser.idn)
        print(violetlaser.idn)
        print(uvlaser.idn)
        print(daq.idn)
        print('Prior Z stage')

        win = control.TormentaGUI(andor, bluelaser, violetlaser, uvlaser,
                                  scanZ, daq, orcaflash)
        win.show()

        app.exec_()


def analysisApp():

    from analysis import analysis

    app = QtGui.QApplication([])

    win = analysis.AnalysisWidget()
    win.show()

    app.exec_()

    
    
