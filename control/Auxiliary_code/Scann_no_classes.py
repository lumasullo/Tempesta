# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 13:13:12 2016

@author: testaRES
"""
try:
    import libnidaqmx
except:
    from control import libnidaqmx
    
import numpy as np
import copy


def make_ramp(start, end, samples):
    print('samples in make ramp = %s'%samples, 'in make_ramp')
    ramp = []
    k  = (end - start) / (samples - 1)
    print('k = ',k, 'in make_ramp')
    for i in range(0, samples):
        ramp.append(start + k * i)
        
    return [ramp, k]


def run():
    current_aochannels = {'x': 0, 'y': 1}
    current_dochannels = {'355': 0, '405': 1, '488': 2, 'CAM': 3}
    samples_in_scan = 10000
    
    nidaq = libnidaqmx.Device('Dev1')
    nidaq.reset()
    
    aotask = libnidaqmx.AnalogOutputTask('')
    dotask = libnidaqmx.DigitalOutputTask('')  
    
    
    full_ao_signal = []
    final_samps = [1, 1]
    temp_aochannels = copy.copy(current_aochannels)
    min_ao = -10
    max_ao = 10
    for i in range(0,2):
        dim = min(temp_aochannels, key = temp_aochannels.get)
        chanstring = 'Dev1/ao%s'%temp_aochannels[dim]
        aotask.create_voltage_channel(phys_channel = chanstring, channel_name = 'chan%s'%dim, min_val = min_ao, max_val = max_ao)
        temp_aochannels.pop(dim)
    #            signal = self.stage_scan.sig_dict[dim+'_sig']
    #            if i == 1 and len(full_ao_signal) != len(signal):
    #                print('Length of signals are not equal (printed from RunScan()')
    #            full_ao_signal = np.append(full_ao_signal, signal)
    #            final_samps = np.append(final_samps, signal[-1])
    #            print('Length of signal %s = '%i, len(signal))
    #            print('Final samples : ', final_samps)
    full_ao_signal = make_ramp(0, 1, 10000)[0]
    
    print('Samples in scan =', samples_in_scan)
    
    full_do_signal = []
    temp_dochannels = copy.copy(current_dochannels)
    for i in range(0,4):
        dev = min(temp_dochannels, key = temp_dochannels.get)
        chanstring = 'Dev1/port0/line%s'%temp_dochannels[dev]
        dotask.create_channel(lines = chanstring, name = 'chan%s'%dev)
        temp_dochannels.pop(dev)
    #            signal = self.pixel_cycle.sig_dict[dev+'sig']
    #            if len(full_ao_signal)%len(signal) != 0 and len(full_do_signal)%len(signal) != 0:
    #                print('Signal lengths does not match (printed from run)')
    #            full_do_signal = np.append(full_do_signal, signal)
    full_do_signal = np.zeros(10000)
    
    aotask.configure_timing_sample_clock(rate = 100000,
                    sample_mode = 'finite',
                    samples_per_channel = samples_in_scan)
                    
    dotask.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                         rate = 100000, 
                                         sample_mode = 'finite', 
                                         samples_per_channel = samples_in_scan)
                                        
    ###Trigger sig
    #        trigger = np.zeros(x_length)
    #        trigger[range(1, 1000)] = 1
                    
    return_ramps = []
    for i in range(0,2):
        ramp_and_k = make_ramp(final_samps[i], 0, 100000)
        return_ramps = np.append(return_ramps, ramp_and_k[0]) 
    
    print('length of full_do_signal = ', len(full_do_signal))
    print('channels = ', dotask.get_number_of_channels())
    print(dotask,aotask)
    dotask.write(full_do_signal, layout = 'group_by_channel', auto_start = False)
    aotask.write(full_ao_signal, layout = 'group_by_channel', auto_start = False)
    
    
    dotask.start()
    aotask.start()
    aotask.wait_until_done() ##Need to wait for task to finish, otherwise aotask will be deleted 
    dotask.wait_until_done()
    
    aotask.stop()
    dotask.stop()
    
        
    aotask.configure_timing_sample_clock(rate = 100000,
                                             sample_mode = 'finite',
                                             samples_per_channel = 100000)
    print('length of return ramps : ', len(return_ramps))
    #        print('return ramps : ', return_ramps)                                     
                                         
    aotask.write(return_ramps, layout = 'group_by_channel', auto_start = False)
    aotask.start()
    aotask.wait_until_done()
    
    #        aotask.alter_state('unreserve')
    #        dotask.alter_state('unreserve')
    aotask.clear()          ## when function is finished and task aborted
    dotask.clear()        
    #        self.nidaq.reset()        
    del aotask
    del dotask
    #        self.doneSignal.emit()


