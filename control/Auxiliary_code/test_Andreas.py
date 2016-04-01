
"""
Created on Tue Mar  1 16:04:06 2016

@author: testaRES
"""
import libnidaqmx
import numpy as np
import time



class Runner():
    
    def __init__(self, device):
        self.nidaq = device
        self.nidaq.reset()
    def run(self):
        #nidaq = Device('PCI6731')

        a = libnidaqmx.AnalogOutputTask('atask')
        d = libnidaqmx.DigitalOutputTask('dtask')
        #d2 = libnidaqmx.DigitalOutputTask('dtask2')
        #d = DigitalOutputTask('dtask')
        
        d.create_channel('Dev1/port0/line1', 'channel0')
        a.create_voltage_channel('Dev1/ao0', 'ch1')
        #d2.create_channel('PCI6731/port0/line1', 'channel1')
        a.configure_timing_sample_clock(rate = 100000, sample_mode = 'finite', samples_per_channel = 100000)
        d.configure_timing_sample_clock(source = r'ao/SampleClock',  rate = 100000, active_edge = 'rising', sample_mode = 'finite', samples_per_channel = 100000)
        
        #d.set_buffer_size(100)
        #
        #
        signal = np.zeros(100000)
        signal[range(0,50000)] = 1
        
        ##
        #
        #
        ##cont = ''
        ##while cont == '':
        ##    d = libnidaqmx.DigitalOutputTask('dtask')
        ##    d.create_channel('PCI6731/port0/line0', 'channel0')
        ##    d.write(signal)
        ##    d.wait_until_done()
        ##    time.sleep(1)
        ##    d.stop()
        ##    del d
        ##    nidaq.reset()
        ##    cont = input('Continue?')
        ##    print(cont)
        ##
        d.write(signal, layout = 'group_by_channel', auto_start = False)
        a.write(signal, auto_start = False)
        d.start()
        a.start()
        d.wait_until_done()
        d.clear()
        a.clear()
        self.nidaq.reset()
        
if __name__ == '__main__':
    runner = Runner(libnidaqmx.Device('Dev1'))