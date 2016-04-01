# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:42:47 2016

@author: testaRES
"""

import control.instruments as instruments
import time

orca = instruments.Camera()

start = time.time()

vsize = 200
hsize= 200
vpos = 1500
hpos = 1500

orca.startAcquisition()
now = time.time() - start
print('01:', now)
orca.stopAcquisition()
now = time.time() - start
print('02:', now)

now = time.time() - start
print('0:', now)
orca.setPropertyValue('subarray_vsize', vsize)
now = time.time() - start
print('1:', now)
orca.setPropertyValue('subarray_hsize', hsize)
now = time.time() - start
print('2:', now)
orca.setPropertyValue('subarray_vpos', vpos)
now = time.time() - start
print('3:', now)
orca.setPropertyValue('subarray_hpos', hpos)
now = time.time() - start
print('4:', now)

orca.startAcquisition()
now = time.time() - start
print('03:', now)
orca.stopAcquisition()
now = time.time() - start
print('04:', now)