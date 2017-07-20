# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 20:02:08 2014

@author: Federico Barabas
"""

import logging

from lantz import Driver
from lantz import Q_

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S')


class constants:

    def __init__(self):
        self.GND = 0


class MockLaser(Driver):

    def __init__(self):
        super().__init__()

        self.mW = Q_(1, 'mW')

        self.enabled = False
        self.power_sp = 0 * self.mW

    @property
    def idn(self):
        return 'Simulated laser'

    @property
    def status(self):
        """Current device status
        """
        return 'Simulated laser status'

    # ENABLE LASER
    @property
    def enabled(self):
        """Method for turning on the laser
        """
        return self.enabled_state

    @enabled.setter
    def enabled(self, value):
        self.enabled_state = value

    # LASER'S CONTROL MODE AND SET POINT

    @property
    def power_sp(self):
        """To handle output power set point (mW) in APC Mode
        """
        return self.power_setpoint

    @power_sp.setter
    def power_sp(self, value):
        self.power_setpoint = value

    # LASER'S CURRENT STATUS

    @property
    def power(self):
        """To get the laser emission power (mW)
        """
        return 55555 * self.mW

    def enter_mod_mode(self):

        pass

    @property
    def digital_mod(self):
        """digital modulation enable state
        """
        return True

    @digital_mod.setter
    def digital_mod(self, value):
        pass

    def mod_mode(self):
        """Returns the current operating mode
        """
        return 0

    def query(self, text):

        return 0
