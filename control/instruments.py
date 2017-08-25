# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:25:27 2014

@author: federico
"""

import importlib
import control.mockers as mockers


class Laser(object):

    def __new__(cls, iName, *args):
        try:
            pName, driverName = iName.rsplit('.', 1)
            package = importlib.import_module('lantz.drivers.legacy.' + pName)
            driver = getattr(package, driverName)
            laser = driver(*args)
            laser.initialize()
            return driver(*args)

        except:
            return mockers.MockLaser()


class LinkedLaserCheck(object):

    def __new__(cls, iName, ports):
        try:
            pName, driverName = iName.rsplit('.', 1)
            package = importlib.import_module('lantz.drivers.legacy.' + pName)
            driver = getattr(package, driverName)
            laser0 = driver(ports[0])
            laser0.initialize()
            laser1 = driver(ports[1])
            laser1.initialize()
            return LinkedLaser([laser0, laser1])

        except:
            return mockers.MockLaser()


class LinkedLaser(object):

    def __init__(self, lasers):
        self.lasers = lasers

    @property
    def idn(self):
        return 'Linked Lasers' + self.lasers[0].idn + self.lasers[1].idn

    @property
    def autostart(self):
        return self.lasers[0].autostart

    @autostart.setter
    def autostart(self, value):
        self.lasers[0].autostart = self.lasers[1].autostart = value

    @property
    def power(self):
        return self.lasers[0].power + self.lasers[1].power

    @property
    def power_sp(self):
        return self.lasers[0].power_sp + self.lasers[1].power_sp

    @power_sp.setter
    def power_sp(self, value):
        self.lasers[0].power_sp = self.lasers[1].power_sp = 0.5*value

    @property
    def digital_mod(self):
        return self.lasers[0].digital_mode

    @digital_mod.setter
    def digital_mod(self, value):
        self.lasers[0].digital_mod = self.lasers[1].digital_mod = value

    def finalize(self):
        self.lasers[0].finalize()
        self.lasers[1].finalize()


class Camera(object):
    """ Buffer class for testing whether the camera is connected. If it's not,
    it returns a dummy class for program testing. """
# TODO:
    """This was originally (by federico) called from tormenta.py using a "with"
    call, as with the Lasers. But accoring to literature, "with" should be
    used with classes having __enter__ and __exit functions defined.
    For some reason this particular class gives "Error: class is missing
    __exit__ fcn" (or similar)
    Maybe it could be rewritten using __enter__  __exit__.
    http://effbot.org/zone/python-with-statement.htm
    Although I believe that design is more suitable for funcions that are
    called alot or environments that are used alot."""
    def __new__(cls, id, *args, **kwargs):
        try:
            import lantz.drivers.hamamatsu.hamamatsu_camera as hm
            orcaflash = hm.HamamatsuCameraMR(id)
            print('Initializing Hamamatsu Camera Object, model: ',
                  orcaflash.camera_model)
            return orcaflash

        except:
            print('Initializing Mock Hamamatsu')
            return mockers.MockHamamatsu()
