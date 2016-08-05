# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:32:36 2016

@author: aurelien.barbotin et le monsieur de l'internet
"""

import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
	port='COM10',
	baudrate=38400,
	stopbits=serial.STOPBITS_ONE,
	bytesize=serial.EIGHTBITS,
      timeout=1,
)

if not ser.isOpen():
    ser.open()

a=ser.write(b'h\n')
import time
time.sleep(0.5)
text=ser.read_all()
text=text.decode()
#print(text)
#def readLine():
#    a=b""
#    line=""
#    while(a!=b"\n"):
#        a=ser.read()
#        line+=a.decode()
#    print(line)
#    return line
ser.write(b"lpa?\n")
print("internal measured laser power",ser.readline().decode())

ser.write(b"lp?\n")
print("return set laser power",ser.readline().decode())
ser.write(b"lip?\n")
print(ser.readline().decode())
ser.write(b"li?\n")
print(ser.readline().decode())
ser.write(b"lx_temp_shg?\n")
print(ser.readline().decode())
ser.close()