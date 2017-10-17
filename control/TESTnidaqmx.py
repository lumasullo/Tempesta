# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:39:16 2017

@author: Testa4
"""

import nidaqmx
import numpy as np

print('Running')
task = nidaqmx.Task()
samp = 70

## Analog output
#signal = np.concatenate((np.ones(samp), np.zeros(samp)))
#task.ao_channels.add_ao_voltage_chan(
#        physical_channel='Dev1/ao2',
#        name_to_assign_to_channel='chan1',
#        min_val=-10, max_val=10)

# Digital output
signal = np.concatenate((np.ones(samp, dtype=bool),
                         np.zeros(samp, dtype=bool)))
task.do_channels.add_do_chan(lines='Dev1/port0/line2')

task.timing.cfg_samp_clk_timing(
        rate=100000,
        source=r'100kHzTimeBase',
        sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
        samps_per_chan=200000)

task.write(signal, auto_start=False, timeout=10)
task.start()
task.wait_until_done()


task.close()
