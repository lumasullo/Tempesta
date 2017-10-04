# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: Barabas, Bodén, Masullo
"""
from pyqtgraph.Qt import QtGui
import nidaqmx
import sys

from control import control
import control.instruments as instruments


def main():

    app = QtGui.QApplication([])

    cobolt = 'cobolt.cobolt0601.Cobolt0601'

    nidaq = nidaqmx.system.System.local().devices['Dev1']

    with instruments.Laser(cobolt, 'COM4') as actlaser, \
            instruments.PZT(8) as pzt, instruments.Webcam() as webcam:

        offlaser = instruments.LinkedLaserCheck(cobolt, ['COM5', 'COM10'])
        exclaser = instruments.LaserTTL(0)
        orcaflashV3 = instruments.Camera(0)
        orcaflashV2 = instruments.Camera(1)
        print(actlaser.idn)
        print(exclaser.line)
        print(offlaser.idn)

        win = control.TormentaGUI(actlaser, offlaser, exclaser, orcaflashV2,
                                  orcaflashV3, nidaq, pzt, webcam)
        win.show()

        sys.exit(app.exec_())
