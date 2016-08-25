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
import time

import pygame
import pygame.camera

import control.mockers as mockers
from control.SLM import slmpy

class Webcam(object):
    """Initiates communication with a webcam with pygame"""
    def __new__(cls, *args):
        try:
            pygame.camera.init()
            webcam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
            webcam.start()
            return webcam

        except:
            return mockers.MockWebcam()


class Laser(object):
    """An object to communicate with a laser with a Lantz driver"""
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
    """Drives a stage for Z scanning"""
    def __new__(cls, *args):

        try:
            from lantz.drivers.prior.nanoscanz import NanoScanZ
            scan = NanoScanZ(*args)
            scan.initialize()
            return scan

        except:
            return mockers.MockScanZ()


class OneFiveLaser(object):
    """class used to control the 775 nm laser via a RS 232 interface. The commands are defined
    in the laser manual, must be binary and end with an LF statement 
    
    :param string port: specifies the name of the port to which the laser is connected (usually COM10).
    :param float intensity_max: specifies the maximum intensity (in our case in W) the user can ask the laser to emit. It is used to protect sensitive hardware from high intensities
    """
    
    def __init__(self,port="COM10",intensity_max=0.8):
        self.serial_port = None
        self.info = None      #Contains informations about the different methods of the laser. Can be used that the communication works
        self.power_setting = 0    #To change the power with python
        self.intensity_max = intensity_max
        self.mode = 0     # Constant current or Power
        self.triggerMode = 2        #Trigger=TTL input
        self.enabled_state = False    #laser initially off
        self.mW = Q_(1, 'mW')
        
        try:
            import serial
            self.serial_port = serial.Serial(
            	port=port,
            	baudrate=38400,
            	stopbits=serial.STOPBITS_ONE,
            	bytesize=serial.EIGHTBITS
                 )
#            self.getInfo()
            self.setPowerSetting()
#            self.mode=self.getMode()
            self.setPowerSetting(self.power_setting)
            self.power_setpoint=0
            self.power_sp = 0
            self.setTriggerSource(self.triggerMode)
        except:
            print("Channel Busy")
            self=mockers.MockLaser()
            
    @property
    def idn(self):
        return 'OneFive 775nm'

    @property
    def status(self):
        """Current device status
        """
        return 'OneFive laser status'

    @property
    def enabled(self):
        """Method for turning on the laser
        """
        return self.enabled_state

    @enabled.setter
    def enabled(self, value):
        cmd="le="+str( int( value) )+"\n"
        self.serial_port.write (cmd.encode() )
        self.enabled_state = value

    # LASER'S CONTROL MODE AND SET POINT

    @property
    def power_sp(self):
        """To handle output power set point (mW)
        """
        return self.power_setpoint * 100 * self.mW

    @power_sp.setter
    def power_sp(self, value):
        """Handles output power. Sends a RS232 command to the laser specifying the new intensity."""
        value=value.magnitude/1000  #Conversion from mW to W
        if(self.power_setting!=1):
            print("Wrong mode: impossible to change power value. Please change the power settings")
            return
        if(value<0):
            value=0
        if(value>self.intensity_max):
            value=self.intensity_max #A too high intensity value can damage the SLM
        value=round(value,3)
        cmd="lp="+str(value)+"\n"
        self.serial_port.write(cmd.encode())
        self.power_setpoint = value

    # LASER'S CURRENT STATUS

    @property
    def power(self):
        """To get the laser emission power (mW)
        """
        return self.intensity_max * 100 * self.mW
            
    def getInfo(self):
        """Returns available commands"""
        if self.info is None:
            self.serial_port.write(b"h\n")
            time.sleep(0.5)
            self.info=self.serial_port.read_all().decode()
        else:
            print(self.info)
        
    def setPowerSetting(self,manual=1):
        """if manual=0, the power can be changed via this interface
        if manual=1, it has to be changed by turning the button (manually)"""
        if(manual!=1 and manual !=0):
            print("setPowerSetting: invalid argument")
            self.power_setting=0
        self.power_setting=manual
        value="lps"+str(manual)+"\n"
        self.serial_port.write(value.encode())

        
    def setMode(self,value) :
        """value=1: constant current mode
        value=0 : constant power mode"""   
        if(value!=1 and value !=0):
            print("wrong value")
            return
        self.mode=value
        cmd="lip="+str(value)+"\n"
        self.serial_port.write(cmd.encode())
        
    def setCurrent(self,value):
        """sets current in constant current mode."""
        if (self.mode!=1):
            print("You can't set the current in constant power mode")
            return
        if(value<0):
            value=0
        if(value>6):
            value=6 #Arbitrary limit to not burn the components
        value=round(value,2)
        cmd="li="+str(value)+"\n"
        self.serial_port.write(cmd.encode())      
    
    def setFrequency(self,value):
        """sets the pulse frequency in MHz"""
        if(value<18 or value>80):
            print("invalid frequency values")
            return
        value*=10**6
        cmd="lx_freq="+str(value)+"\n"
        self.serial_port.write(cmd.encode())
    
    def setTriggerSource(self,source):
        """source=0: internal frequency generator
        source=1: external trigger source for adjustable trigger level, Tr-1 In
        source=2: external trigger source for TTL trigger, Tr-2 In
        """
        if(source!=0 and source!=1 and source!=2):
            print("invalid source for trigger")
            return
        cmd="lts="+str(source)+"\n"        
        self.triggerMode=source
        self.serial_port.write(cmd.encode())
        
    def setTriggerLevel(self,value):
        """defines the trigger level in Volts, between -5 and 5V"""
        if(np.absolute(value)>5):
            print("incorrect value")
            return
        if(self.triggerMode!=1):
            print("impossible to change the \
            trigger level with this trigger. Please change the trigger source first")
            return
        value=round(value,2)
        cmd="ltll="+str(value)+"\n"
        self.serial_port.write(cmd.encode())
        
        #The get... methods return a string giving information about the laser
    def getPower(self):
        """Returns internal measured Laser power"""
        self.serial_port.flushInput()
        self.serial_port.write(b"lpa?\n")
        value=self.serial_port.readline().decode()
        return value
        
    def getMode(self):    
        """Returns mode of operation: constant current or current power"""
        self.serial_port.flushInput()
        self.serial_port.write(b"lip?\n")
        value=self.serial_port.readline().decode()
        if(value=="lip=0\n"):
            value="Constant power mode"
        else:
            value="Constant current mode"
        return value
        
    def getPowerCommand(self):    
        """gets the nominal power command in W"""
        self.serial_port.flushInput()
        self.serial_port.write(b"lp?\n")
        value=self.serial_port.readline().decode()
        return "power command: "+value+"W"  

    def getTemperature(self):
        """Gets Temperature of SHG cristal"""
        self.serial_port.flushInput()
        self.serial_port.write(b"lx_temp_shg?\n")
        value=self.serial_port.readline().decode()
        return value
        
    def getCurrent(self):
        """Returns actual current"""
        self.serial_port.flushInput()
        self.serial_port.write(b"li?\n")
        value=self.serial_port.readline().decode()
        return value
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args, **kwargs):
        self.close()
        
    def close(self):
        self.enabled = False
        self.serial_port.close()
        
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

class SLM(object):
    """This object communicates with an SLM as a second monitor, using a wxpython interface defined in slmpy.py. 
    If no second monitor is detected, it replaces it by a Mocker with the same methods as the normal SLM object"""
    def __init__(self):
        """"""
        super(SLM).__init__()
        try:
            self.slm=slmpy.SLMdisplay()
        except:
            self.slm = mockers.MockSLM()
        
        
    def __enter__(self, *args, **kwargs):
        return self.slm
        
    def __exit__(self, *args, **kwargs):
        self.slm.close()