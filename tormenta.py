# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: Federico Barabas
"""
from pyqtgraph.Qt import QtGui


def main():

    from control import control
    import control.instruments as instruments

    app = QtGui.QApplication([])

# TODO: create an instruments.Camera(hamamatsu) or something similar
#    with instruments.Camera('hamamatsu.hamamatsu_camera.HamamatsuCameraMR') as orcaflash, \
    with instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM4') as violetlaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM13') as exclaser, \
         instruments.Laser('cobolt.cobolt0601.Cobolt0601', 'COM5') as offlaser:

<<<<<<< HEAD
        orcaflash = instruments.Camera()
        print(violetlaser.idn)
        print(exclaser.idn)
        print(offlaser.idn)
        print('Prior Z stage')

        win = control.TormentaGUI(violetlaser, exclaser, offlaser, orcaflash)
=======
        orcaflashV3 = instruments.Camera(0)
        orcaflashV2 = instruments.Camera(1)
        print(violetlaser.idn)
        print(exclaser.idn)
        print(offlaser.idn)

        win = control.TormentaGUI(violetlaser, exclaser, offlaser,
                                  orcaflashV2, orcaflashV3)
>>>>>>> DualCam
        win.show()

        app.exec_()
