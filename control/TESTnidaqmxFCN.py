# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:39:16 2017

@author: Testa4
"""

import nidaqmx
import numpy as np


def run():
    print('Running')
    task = nidaqmx.Task()
    samp = 1000
    signal1 = np.array(np.concatenate((np.ones(samp), np.zeros(samp))))
    TwoSigArray = np.array((signal1, signal1))
    print(signal1)
    task.ao_channels.add_ao_voltage_chan(
            physical_channel='Dev1/ao0',
            name_to_assign_to_channel='chan1',
            min_val=-10, max_val=10)
    task.ao_channels.add_ao_voltage_chan(
            physical_channel='Dev1/ao2',
            name_to_assign_to_channel='chan2',
            min_val=-10, max_val=10)

    task.timing.cfg_samp_clk_timing(
            rate=100000,
            source=r'100kHzTimeBase',
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=200000)
#    task.start()
    task.write(TwoSigArray, auto_start=False)
    task.start()
    task.wait_until_done()
    print('Finished')


if __name__ == "__main__":
    run()
