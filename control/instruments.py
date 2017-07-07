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


class Camera(object):
    """ Buffer class for testing whether the camera is connected. If it's not,
    it returns a dummy class for program testing. """
# TODO:
    """This was originally (by federico) called from tormenta.py using a
    "with" call, as with the Lasers. But accoring to litterature, "with"
    should be used with classes having __enter__ and __exit functions defined.
    For some reason this particular class gives "Error: class is missing
    __exit__ fcn" (or similar). Maybe it could be rewritten using __enter__
    __exit__.
    http://effbot.org/zone/python-with-statement.htm
    Although I believe that design is more suitable for funcions that are
    called alot or environments that are used alot."""

    def __new__(cls, *args):

        try:
            import lantz.drivers.hamamatsu.hamamatsu_camera as hm
            orcaflash = hm.HamamatsuCameraMR(0)
            print('Initializing Hamamatsu Camera Object, model: ',
                  orcaflash.camera_model)
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
