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

#    with instruments.Camera('hamamatsu.hamamatsu_camera.HamamatsuCameraMR') as orcaflash, \
    with instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM12') as violetlaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM13') as exclaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM5') as offlaser, \
         instruments.DAQ() as daq, instruments.ScanZ(12) as scanZ:

        orcaflash = instruments.Camera()
        print(violetlaser.idn)
        print(exclaser.idn)
        print(offlaser.idn)
        print(daq.idn)
        print('Prior Z stage')

        win = control.TormentaGUI(violetlaser, exclaser, offlaser, scanZ, daq,
                                  orcaflash)
        win.show()

        app.exec_()
