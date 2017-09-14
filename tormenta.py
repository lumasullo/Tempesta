# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: Barabas, Bodén, Masullo
"""
from pyqtgraph.Qt import QtGui
import nidaqmx

from control import control
import control.instruments as instruments


def main():

    app = QtGui.QApplication([])

    cobolt = 'cobolt.cobolt0601.Cobolt0601'

    nidaq = nidaqmx.system.System.local().devices['Dev1']

    with instruments.Laser(cobolt, 'COM5') as violetlaser, \
        instruments.PZT('nv401', 8) as pzt:

      offlaser = instruments.LinkedLaserCheck(cobolt, ['COM6', 'COM4'])
        exclaser = instruments.LaserTTL(0)
        orcaflashV3 = instruments.Camera(0)
        orcaflashV2 = instruments.Camera(1)
        print(violetlaser.idn)
        print(exclaser.line)
        print(offlaser.idn)

        win = control.TormentaGUI(violetlaser, offlaser, exclaser, orcaflashV2,
                                  orcaflashV3, nidaq, pzt)
        win.show()

        app.exec_()