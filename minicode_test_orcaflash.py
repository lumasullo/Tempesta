# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 13:57:46 2015

@author: testaRES
"""

import matplotlib.pyplot as plt
import numpy as np

from hamamatsu import hamamatsu_camera as hm

orcaflash = hm.HamamatsuCameraMR(0)
orcaflash.startAcquisition()
a = orcaflash.getFrames()
pic = a[0][1].getData()
pic2 = np.reshape(pic, (2048, 2048))
plt.imshow(pic2)