# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:25:27 2014

@author: federico
"""

import numpy as np
import importlib

from lantz.drivers.andor.ccd import CCD
#   from lantz.drivers.labjack.t7 import T7
from lantz import Q_


import pygame
import pygame.camera

import control.mockers as mockers


class Webcam(object):

    def __new__(cls, *args):
        try:
            pygame.camera.init()
            webcam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
            webcam.start()
            return webcam

        except:
            return mockers.MockWebcam()


class Laser(object):

    def __new__(cls, iName, *args):
        try:
            pName, driverName = iName.rsplit('.', 1)
            package = importlib.import_module('lantz.drivers.' + pName)
            driver = getattr(package, driverName)
            laser = driver(*args)
            laser.initialize()
        
            return driver(*args)

        except:
            return mockers.MockLaser()


class DAQ(object):
    def __new__(cls, *args):

        try:
            from labjack import ljm
            handle = ljm.openS("ANY", "ANY", "ANY")
            ljm.close(handle)
            return STORMDAQ(*args)

        except:
            return mockers.MockDAQ()


#class STORMDAQ(T7):
#    """ Subclass of the Labjack lantz driver. """
#    def __init__(self, *args):
#
#        super().__init__(*args)
#        super().initialize(*args)
#
#        # Clock configuration for the flipper
#        self.writeName("DIO_EF_CLOCK0_ENABLE", 0)
#        self.writeName("DIO_EF_CLOCK0_DIVISOR", 1)
#        self.writeName("DIO_EF_CLOCK0_ROLL_VALUE", 1600000)
#        self.writeName("DIO_EF_CLOCK0_ENABLE", 1)
#        self.writeName("DIO2_EF_ENABLE", 0)
#        self.writeName("DIO2_EF_INDEX", 0)
#        self.writeName("DIO2_EF_OPTIONS", 0)
#        self.flipperState = True
#        self.flipper = self.flipperState
#        self.writeName("DIO2_EF_ENABLE", 1)
#
#    @property
#    def flipper(self):
#        """ Flipper True means the ND filter is in the light path."""
#        return self.flipperState
#
#    @flipper.setter
#    def flipper(self, value):
#        if value:
#            self.writeName("DIO2_EF_CONFIG_A", 150000)
#        else:
#            self.writeName("DIO2_EF_CONFIG_A", 72000)
#
#        self.flipperState = value
#
#    def toggleFlipper(self):
#        self.flipper = not(self.flipper)


class ScanZ(object):
    def __new__(cls, *args):

        try:
            from lantz.drivers.prior.nanoscanz import NanoScanZ
            scan = NanoScanZ(*args)
            scan.initialize()
            return scan

        except:
            return mockers.MockScanZ()


class Camera(object):
    """ Buffer class for testing whether the camera is connected. If it's not,
    it returns a dummy class for program testing. """
#TODO:
    """This was originally (by federica) called from tormenta.py using a "with" call, as with the Lasers. But
    accoring to litterature, "with" should be used with classes having __enter__ and __exit functions defined. 
    For some reason this particular class gives "Error: class is missing __exit__ fcn" (or similar)
    Maybe it could be rewritten using __enter__  __exit__. 
    http://effbot.org/zone/python-with-statement.htm
    Although I believe that design is more suitable for funcions that a 
    called alot or environments that are used alot."""


    def __new__(cls, *args):

        try:     
            import lantz.drivers.hamamatsu.hamamatsu_camera as hm
            orcaflash = hm.HamamatsuCameraMR(0)
            print('Initializing Hamamatsu Camera Object, model: ', orcaflash.camera_model)
            return orcaflash

        except:
            import control.MockHamamatsu as MockHamamatsu
            print('Initializing Mock Hamamatsu')
            return MockHamamatsu.MockHamamatsu()


#class OrcaflashCamera(hm.HamamatsuCameraMR):
#
#
#    def __init__(self, camera_id):
#        hm.HamamatsuCameraMR.__init__(self, camera_id)
#
##            pName, driverName = iName.rsplit('.', 1)
##            package = importlib.import_module('lantz.drivers.' + pName)
##            driver = getattr(package, driverName)
##            camera = driver(*args)
##            camera.lib.Initialize()


class STORMCamera(CCD):
    """ Subclass of the Andor's lantz driver. It adapts to our needs the whole
    functionality of the camera. """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        super().initialize(*args, **kwargs)

        self.s = Q_(1, 's')
        self.mock = False

        # Default imaging parameters
        self.readout_mode = 'Image'
        self.trigger_mode = 'Internal'
        self.EM_advanced_enabled = False
        self.EM_gain_mode = 'RealGain'
        self.amp_typ = 0
        self.set_accum_time(0 * self.s)          # Minimum accumulation and
        self.set_kinetic_cycle_time(0 * self.s)  # kinetic times

        # Lists needed for the ParameterTree
        self.PreAmps = np.around([self.true_preamp(n)
                                  for n in np.arange(self.n_preamps)],
                                 decimals=1)[::-1]
        self.HRRates = [self.true_horiz_shift_speed(n)
                        for n in np.arange(self.n_horiz_shift_speeds())][::-1]
        self.vertSpeeds = [np.round(self.true_vert_shift_speed(n), 1)
                           for n in np.arange(self.n_vert_shift_speeds)]
        self.vertAmps = ['+' + str(self.true_vert_amp(n))
                         for n in np.arange(self.n_vert_clock_amps)]
        self.vertAmps[0] = 'Normal'
