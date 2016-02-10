# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:56:16 2016

@author: andreas.boden
"""
import ctypes
import ctypes.util
import logging
import numpy as np
import pygame

from lantz import Driver
from lantz import Q_

class HMockCamData():

    ## __init__
    #
    # Create a data object of the appropriate size.
    #
    # @param size The size of the data object in bytes.
    #
    def __init__(self, size):
        self.np_array = np.random.randint(1, 256, size)
        self.size = size

    ## __getitem__
    #
    # @param slice The slice of the item to get.
    #
    def __getitem__(self, slice):
        return self.np_array[slice]

    ## copyData
    #
    # Uses the C memmove function to copy data from an address in memory
    # into memory allocated for the numpy array of this object.
    #
    # @param address The memory address of the data to copy.
    #
    def copyData(self, address):
        ctypes.memmove(self.np_array.ctypes.data, address, self.size)

    ## getData
    #
    # @return A numpy array that contains the camera data.
    #
    def getData(self):
        return self.np_array

    ## getDataPtr
    #
    # @return The physical address in memory of the data.
    #
    def getDataPtr(self):
        return self.np_array.ctypes.data



class MockHamamatsu(Driver):


    def __init__(self):

        self.buffer_index = 0
        self.camera_id = 9999
        self.camera_model = 'Mock Hamamatsu camera'
        self.debug = False
        self.frame_bytes = 0
        self.frame_x = 0
        self.frame_y = 0
        self.last_frame_number = 0
        self.properties = {}
        self.max_backlog = 0
        self.number_image_buffers = 0

        # Open the camera.
#        self.camera_handle = ctypes.c_void_p(0)
#        self.checkStatus(dcam.dcam_open(ctypes.byref(self.camera_handle),
#                                        ctypes.c_int32(self.camera_id),
#                                        None),
#                         "dcam_open")
        # Get camera properties.
        self.properties = {'Name': 'MOCK Hamamatsu',
        'exposure_time': 9999,
        'image_width': 2048, 
        'image_height': 2048, 
        'image_framebytes': 8, 
        'subarray_hsize': 2048, 
        'subarray_vsize': 2048,
        'subarray_mode': 'OFF'};

        # Get camera max width, height.
        self.max_width = self.getPropertyValue("image_width")[0]
        self.max_height = self.getPropertyValue("image_height")[0]

    ## captureSetup
    #
    # Capture setup (internal use only). This is called at the start
    # of new acquisition sequence to determine the current ROI and
    # get the camera configured properly.
    #
    def captureSetup(self):
        self.buffer_index = -1
        self.last_frame_number = 0

        # Set sub array mode.
        self.setSubArrayMode()

        # Get frame properties.
        self.frame_x = self.getPropertyValue("image_width")[0]
        print('frame_x set to ', self.getPropertyValue('image_width')[0])
        self.frame_y = self.getPropertyValue("image_height")[0]
        self.frame_bytes = self.getPropertyValue("image_framebytes")[0]


    ## checkStatus
    #
    # Check return value of the dcam function call.
    # Throw an error if not as expected?
    #
    # @return The return value of the function.
    #
    def checkStatus(self, fn_return, fn_name= "unknown"):
        pass


    ## getFrames
    #
    # Gets all of the available frames.
    #
    # This will block waiting for new frames even if 
    # there new frames available when it is called.
    #
    # @return [frames, [frame x size, frame y size]]
    #
    def getFrames(self):
        
        frames = []

        for i in range(2):
        # Create storage
            hc_data = HMockCamData(self.frame_x*self.frame_y)
    
    
            frames.append(hc_data)

        return [frames, [self.frame_x, self.frame_y]]
        
    ## getModelInfo
    #
    # Returns the model of the camera
    #
    # @param camera_id The (integer) camera id number.
    #
    # @return A string containing the camera name.
    #
    def getModelInfo(self):

        return 'WARNING!: This is a Mock Version of the Hamamatsu Orca flash camera'

    ## getProperties
    #
    # Return the list of camera properties. This is the one to call if you
    # want to know the camera properties.
    #
    # @return A dictionary of camera properties.
    #
    def getProperties(self):
        return self.properties

    ## getPropertyAttribute
    #
    # Return the attribute structure of a particular property.
    #
    # FIXME (OPTIMIZATION): Keep track of known attributes?
    #
    # @param property_name The name of the property to get the attributes of.
    #
    # @return A DCAM_PARAM_PROPERTYATTR object.
    #
    def getPropertyAttribute(self, property_name):
        pass

    ## getPropertyText
    #
    # Return the text options of a property (if any).
    #
    # @param property_name The name of the property to get the text values of.
    #
    # @return A dictionary of text properties (which may be empty).
    #
    def getPropertyText(self, property_name):
        pass

    ## getPropertyRange
    #
    # Return the range for an attribute.
    #
    # @param property_name The name of the property (as a string).
    #
    # @return [minimum value, maximum value]
    #
    def getPropertyRange(self, property_name):
        pass

    ## getPropertyRW
    #
    # Return if a property is readable / writeable.
    #
    # @return [True/False (readable), True/False (writeable)]
    #
    def getPropertyRW(self, property_name):
        pass

    ## getPropertyVale
    #
    # Return the current setting of a particular property.
    #
    # @param property_name The name of the property.
    #
    # @return [the property value, the property type]
    #
    def getPropertyValue(self, property_name):

        prop_value = self.properties[property_name]
        prop_type = property_name

        return [prop_value, prop_type]

    ## isCameraProperty
    #
    # Check if a property name is supported by the camera.
    #
    # @param property_name The name of the property.
    #
    # @return True/False if property_name is a supported camera property.
    #
    def isCameraProperty(self, property_name):
        if (property_name in self.properties):
            return True
        else:
            return False

    ## newFrames
    #
    # Return a list of the ids of all the new frames since the last check.
    #
    # This will block waiting for at least one new frame. 
    #
    # @return [id of the first frame, .. , id of the last frame]
    #
    def newFrames(self):

        # Create a list of the new frames.
        new_frames = [9999, 9999, 9999]

        return new_frames

    ## setPropertyValue
    #
    # Set the value of a property.
    #
    # @param property_name The name of the property.
    # @param property_value The value to set the property to.
    #
    def setPropertyValue(self, property_name, property_value):

        # Check if the property exists.
        if not (property_name in self.properties):
            print(" unknown property name:", property_name)
            return False

        # If the value is text, figure out what the 
        # corresponding numerical property value is.

        self.properties[property_name] = property_value
        print(property_name, 'set to:', self.properties[property_name])
#            if (property_value in text_values):
#                property_value = float(text_values[property_value])
#            else:
#                print(" unknown property text value:", property_value, "for", property_name)
#                return False
            
        return property_value

    ## setSubArrayMode
    #
    # This sets the sub-array mode as appropriate based on the current ROI.
    #
    def setSubArrayMode(self):

        # Check ROI properties.
        roi_w = self.getPropertyValue("subarray_hsize")[0]
        roi_h = self.getPropertyValue("subarray_vsize")[0]

        # If the ROI is smaller than the entire frame turn on subarray mode
        if ((roi_w == self.max_width) and (roi_h == self.max_height)):
            self.setPropertyValue("subarray_mode", "OFF")
        else:
            self.setPropertyValue("subarray_mode", "ON")

    ## startAcquisition
    #
    # Start data acquisition.
    #
    def startAcquisition(self):
        self.captureSetup()
        pass

    ## stopAcquisition
    #
    # Stop data acquisition.
    #
    def stopAcquisition(self):
        pass

    ## shutdown
    #
    # Close down the connection to the camera.
    #
    def shutdown(self):
        pass