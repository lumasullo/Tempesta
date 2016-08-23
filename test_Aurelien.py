# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:24:58 2016

@author: aurelien.barbotin
"""
import nidaqmx
from nidaqmx import AnalogOutputTask
from nidaqmx import CounterInputTask
import numpy as np
import scipy.signal as sg
from pylab import plot, show
from nidaqmx import AnalogInputTask
import nidaqmx
from PyDAQmx import Task
import matplotlib.pyplot as plt
import time
mode="readDigital"

def func(task, event_type, samples, cb_data):
    print("le million",time.clock())
    return 0
    
if mode=="write":
    data = 1.5*sg.sawtooth(np.arange(10000, dtype=np.float64)*2*np.pi/1000)
    task = AnalogOutputTask()
    task.create_voltage_channel('Dev1/ao2', min_val=-2.0, max_val=2.0)
    task.configure_timing_sample_clock(rate = 1000.0)
    task.write(data, auto_start=False)
    task.start()
    input('Generating voltage continuously. Press Enter to interrupt..')
    task.stop()
    del task
    print("-Euh Gaël?\n-Quoi?\n-Ca marche bien!")
    
if mode=="read":
    
    i=0
    
    result=np.zeros(12)    
    
    taskao = AnalogOutputTask()
#    taskao.create_voltage_channel('Dev1/ao2', min_val=0, max_val=1.25)
#    sensitivity=0.2
#    taskao.start()
#    taskao.write(sensitivity,auto_start=True)   
#    time.sleep(0.4)
    
    task = AnalogInputTask()
    task.create_voltage_channel('Dev1/ai0', terminal = 'rse', min_val=-10.0, max_val=10.0)
    task.configure_timing_sample_clock(rate = 1000.0)
    task.start()

    trigger = nidaqmx.CounterOutputTask("trigger")
    trigger.create_channel_ticks('Dev1/ctr1',name="pasteque",  low_ticks=100000, high_ticks=1000000)
    trigger.set_terminal_pulse('Dev1/ctr1',"PFI12")
    trigger.configure_timing_sample_clock(source = r'ao/SampleClock', 
                                         sample_mode = "hwtimed")
    print("avant data")
    t=time.clock()
    data = task.read(5000, fill_mode='group_by_channel')
    print(time.clock()-t)
    print("apres data")
    print(time.clock()-t)

#    plot (result)
#    show ()
#    i+=1
#    taskao.write(0,auto_start=True)
##    time.sleep(3)
    task.stop()
#    taskao.stop()
##    taskao.write(0)
    del task
#    del taskao
    
    
if mode=="read2":
    task = AnalogInputTask()
    task.create_voltage_channel('Dev1/ai0', terminal = 'rse', min_val=-10.0, max_val=10.0)
    task.configure_timing_sample_clock(rate = 1000.0,sample_mode='finite',samples_per_channel=2000)
    task.create_voltage_channel('Dev1/ai5',terminal = 'rse', min_val=-10.0, max_val=10.0)
    task.start()
    d=task.read(samples_per_channel=2000)
    plt.figure()
    plt.plot(d)
    plt.show()
if mode=="readDigital":
    rate=100000
    nidaq=nidaqmx.Device("Dev1")
    task = nidaqmx.DigitalInputTask()
    task2 = AnalogInputTask()
    task2.create_voltage_channel('Dev1/ai0', min_val=-2.0, max_val=2.0)
    task2.configure_timing_sample_clock(rate = rate)
    
#    task.register_every_n_samples_event(func ,samples=10**4-1)
    counter=0
    a=task.create_channel_count_edges("Dev1/ctr0", init=0 )
    task.set_terminal_count_edges("Dev1/ctr0","PFI0")
    
    samp_per_chan=3*rate
    
    task.configure_timing_sample_clock(source=r'ai/SampleClock',samples_per_channel=samp_per_chan+500,sample_mode="finite")
    tim=0
    task2.start()
#    time.sleep(0.1)
    task.start()
    tim=time.clock()
#    time.sleep(0.2)
    t=time.clock()
    print(task.get_samples_per_channel_acquired())
#    while task.get_samples_per_channel_acquired()<2 and t-time.clock()<3:
#            print("samples acquired yet:",task.get_samples_per_channel_acquired())
    for u in range(3):
        c=task.read(samples_per_channel=samp_per_chan//3,timeout=10)   
    d=task.read(500)
    print(d.shape)
    tim=time.clock()-tim
    task.stop()
    task2.stop()
    del task
    
    d=np.zeros(len(c))
    for i in range(len(c)-1):
        d[i]=c[i+1]-c[i]
    plt.figure()
    plt.subplot(121)
    plt.plot(d)
    plt.subplot(122)
    plt.plot(c)
    print("time to read:",tim,"freq:",1/tim,"Hz")
    plt.show()
    
if mode=="trigger":
        task = AnalogInputTask()
        task.create_voltage_channel('Dev1/ai5', terminal = 'rse', min_val=-10.0, max_val=10.0)
        task.configure_timing_sample_clock(rate = 100000.0,sample_mode="finite",samples_per_channel=2000)
        task.configure_trigger_digital_edge_start("PFI12")
        task.start()
        
        trigger = nidaqmx.CounterOutputTask("trigger")
        trigger.create_channel_ticks('Dev1/ctr1',name="pasteque",  low_ticks=1000, high_ticks=1000)
        trigger.set_terminal_pulse('Dev1/ctr1',"PFI12")
#        trigger.
        trigger.configure_timing_sample_clock(source=r'ao/SampleClock',
                                             sample_mode = "hwtimed") 
                                             
        data = 1.5*sg.sawtooth(np.arange(10000, dtype=np.float64)*2*np.pi/1000)
        task2 = AnalogOutputTask()
        task2.create_voltage_channel('Dev1/ao0', min_val=-2.0, max_val=2.0)
        task2.configure_timing_sample_clock(rate = 1000.0)
        task2.write(data, auto_start=False)
        task2.configure_trigger_digital_edge_start("PFI12")
        task2.start()
        trigger.start()
        d=task.read(2000)
        plt.figure()
        plt.subplot(121)
        plt.plot(d)
        plt.subplot(122)
        plt.plot(data[0:2000]-d[:,0])
#        d2=task.read(2000)
#        plt.plot(d2)