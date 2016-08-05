# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 14:11:02 2016

@author: aurelien.barbotin
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from scipy import signal as sg
import control.SLM.PhaseMask as pm
from PIL import Image
import glob

s_pix=0.02  #pixel size in mm (SLM)
#Path to the correction files for the SLM
path_to_correction=u"C:\\Users\\aurelien.barbotin\Documents\SLM control software\deformation_correction_pattern"

#Getting the list of the correction wavelength
correction_list=glob.glob(path_to_correction+"\*.bmp")
correction_wavelength=[int(x[-9:-6]) for x in correction_list ]

class Mask(object):
    """class creating a mask to be displayed by the SLM """
    def __init__(self,m,n,lbd):
        """initiates the mask as an empty array
        n,m corresponds to the width,height of the created image
        lbd is the illumination wavelength in nm"""
        self.img=np.zeros((m,n),dtype=np.uint8)
        self.m=m
        self.n=n
        self.value_max=255
        self.lbd=lbd
        if lbd==561:
            self.value_max=148
        elif lbd==491:
            self.value_max=129
        elif lbd<780 and lbd>800:
            # Here we infer the value of the maximum with a linear approximation from the ones provided by the manufacturer
            #Better ask them in case you need another wavelength
            self.value_max= int(lbd* 0.45 - 105)
            print("Caution: a linear approximation has been made")
                        
    def __str__(self):
        plt.figure()
        plt.imshow(self.img)
        return "image of the mask"
    
    def two_PiToUInt8(self):
        """Method converting a phase image (values from 0 to 2Pi) into a uint8 image"""
        self.img*=self.value_max/(2*math.pi)
        self.img=np.round(self.img).astype(np.uint8)
        return
        
    def setCircular(self,radius,x=-1,y=-1):
        """This method sets to 0 all the values within Mask except the ones included
        in a circle centered in (x,y) with a radius r"""          
        self.img=pm.setCircular(self.img,radius,x,y)
        return
        
    def setRectangular(self,xtop_left,ytop_left,xlen,ylen):
        mask=np.zeros( (self.m,self.n),dtype=np.uint8 )
        rect=np.ones((ylen,xlen),dtype=np.uint8)
        mask[ytop_left:ytop_left+ylen,xtop_left:xtop_left+xlen]=rect
        self.img*=mask
        return
        
    def load(self,img):
        """initiates the mask with an existing image"""
        tp=img.dtype
        if tp != np.uint8:
            max_val=np.max(img)
            print("input image is not of format uint8")
            if max_val!=0:         
                img=self.value_max*img.astype('float64')/np.max(img)
            img=img.astype('uint8')
        self.img=img
        return
        
    def loadBMP(self,name,path=path_to_correction):
        """loads a bmp image in the img of the current mask"""
        with Image.open(os.path.join(path,name+".bmp")) as data:
            img=np.array(data)
        u,v=img.shape
        if u>self.m:
            diff=u-self.m
            img=img[diff//2:self.m+diff//2,:]
        if v>self.n:
            diff=v-self.n
            img=img[:,diff//2:self.n+diff//2]
            
        if u<=self.m and v<=self.n:
            result=np.zeros((self.m,self.n))
            diffx=(self.n-v)//2
            diffy=(self.m-u)//2
            result[diffy:u+diffy,diffx:v+diffx]=img
            img=result            
            
        self.m,self.n=img.shape
        self.img[:,:]=img[:,:]
        self.scaleToLut()
        return

#        try:
#            image= Image.open(os.path.join(path,name+".bmp"))
#            img=np.array(image)
#            u,v=img.shape
#            if u>self.m:
#                diff=u-self.m
#                img=img[diff//2:self.m+diff//2,:]
#            if v>self.n:
#                diff=v-self.n
#                img=img[:,diff//2:self.n+diff//2]
#                
#            if u<=self.m and v<=self.n:
#                result=np.zeros((self.m,self.n))
#                diffx=(self.n-v)//2
#                diffy=(self.m-u)//2
#                result[diffy:u+diffy,diffx:v+diffx]=img
#                img=result            
#                
#            self.m,self.n=img.shape
#            self.img[:,:]=img[:,:]
#            self.scaleToLut()
#            return
#        except:
#            print("file not found")
#            return
                
    def scaleToLut(self):
        """Scales the values of the pixels according to the LUT"""
        self.img=self.img.astype("float")
        self.img*=self.value_max/np.max(self.img)
        self.img=self.img.astype("uint8")   
        return
        
    def setHelicoidal(self,R,x=0,y=0):
        """Transforms the current mask in a centered helicoidal mask with radius r
        the centre of the helicoid is (x,y)"""
        
        self.img=pm.createHelMask(self.m,self.n,R,u=x,v=y)
        self.two_PiToUInt8()
        return self.img
        
    def flip180(self):
        self.img=np.flipud(self.img)
        
    def fliplr(self):
        self.img=np.fliplr(self.img)
        
    def tilt(self,angle):
        """creates a phase mask which tilts the beam by creating a blazed grating
        angle must be in degrees"""
        angle*= math.pi/180  #conversion in radians
        lbd=self.lbd*10**-6     #conversion in mm
        mask=np.indices((self.m,self.n),dtype="float")[1,:,:]
        
        #Round spatial frequency to avoid aliasing
        f_spat=np.round(lbd/(s_pix*np.sin(angle)))
        
        if np.absolute(f_spat)<3:
            print("spatial frequency:",f_spat,"pixels")
            
        period=2*math.pi/f_spat
        mask*=period
        tilt=sg.sawtooth(mask)+1
        tilt*=self.value_max/2
        tilt=np.round(tilt).astype(np.uint8)
        self.img=tilt
        return self.img   
        
    def topHat(self,radius,sigma,x=0,y=0):
        """replaces self. img with a top hat shape. radius corresponds to the radius
        of the desired phase mask in pixels and sigma to the std of the gaussian beam
        for illumination"""
        self.img=pm.topHat(self.m,self.n,radius,sigma,x,y)        
        #Conversion into a uint8 image
        self.two_PiToUInt8()
        return
        
    def multipleHel(self,N,R,sigma):
        self.img=pm.multipleHel(self.m,self.n,N,R,sigma)
        
        #Conversion into a uint8 image
        self.two_PiToUInt8()
        return
     
    def composite(self,R,sigma,N,frac=0.5):
        """Creates a mask composed with top hat and helicoids.
        R: max radius in pixels
        sigma: std of input gaussian beam in pix
        N: number of top hats, N+1 is the number of helicoids
        f:focal length of the fourier lens in mm
        """
        self.img=pm.spiral(self.m,self.n,R,sigma,N,frac)
        self.img=self.img%(2*math.pi)
        self.two_PiToUInt8()
        return
    
    def setWhite(self):
        """Creates a blank mask"""      
        self.img=np.ones((self.m,self.n),dtype=np.uint8)*self.value_max
        return
        
    def setBlack(self):
        self.img=np.zeros((self.m,self.n),dtype=np.uint8)
        return
        
    def setSpiral(self,rmax,sigma,N,energy_frac=2/3,n_rings=1):
        """sets the mask as a mix between a top hat and a spiral mask"""
        self.img=pm.compositeMask(self.n,self.m,rmax,sigma,N,energy_frac,n_rings)
        self.two_PiToUInt8()
        return
        
    def image(self):
        return self.img
        
    def __add__(self,other):
        if self.m==other.m and self.n==other.n:
            out=Mask(self.m,self.n,self.lbd)
            out.load( ((self.image()+other.image())%(self.value_max+1)).astype(np.uint8) )
            return out
        else:
            raise TypeError("add two arrays with different shapes")
            
    def __sub__(self,other):
        if self.m==other.m and self.n==other.n:
            out=Mask(self.m,self.n,self.lbd)
            out.load(((self.image()-other.image())%(self.value_max+1)).astype(np.uint8) )
            return out
        else:
            raise TypeError("substract two arrays with different shapes")
        
class EvMask(Mask):
    """class creating an Evolued Mask which contains the phase correction relative to the wavelength"""
    def __init__(self,m,n,lbd):
        Mask.__init__(self,m,n,lbd)
        #correc corresponds to the correction automatically applied 
        self.correc=Mask(m,n,lbd)
        #Find the closest correction pattern within the list of patterns available
        lbdCorrec=min(correction_wavelength, key=lambda x:abs(x-lbd))   
        self.correc.loadBMP("CAL_LSH0701153_"+str(lbdCorrec)+"nm")
        
    def __add__(self,other):
        if self.m==other.m and self.n==other.n:
            out=EvMask(self.m,self.n,self.lbd)
            out.load(((self.image()+other.image())%(self.value_max+1)).astype(np.uint8) )
            return out
        else:
            raise TypeError("add two arrays with different shapes")
            
    def __sub__(self,other):
        if self.m==other.m and self.n==other.n:
            out=EvMask(self.m,self.n,self.lbd)
            out.load(((self.image()-other.image())%(self.value_max+1)).astype(np.uint8) )
            return out
        else:
            raise TypeError("substract two arrays with different shapes")


class DoubleMask(EvMask):
    """Class splitting the SLM in 2 parts: left and right, to create two different phase patterns"""
    def __init__(self,m,n,lbd,left_center=(0,0),right_center=(0,0),frac=0.5):
        """Initiates a DoubleMask like a normal EvMask, with two submasks left and right. Frac determines how much of the
        SLM is dedicated to left."""
        EvMask.__init__(self,m,n,lbd)
        self.size_left=int(np.round(frac*n))
        self.left=Mask(m,self.size_left,lbd)
        self.right=Mask(m,n-self.size_left,lbd)        
        self.left_center = left_center
        self.right_center = right_center
        
    def update(self):
        """Blends left and right parts into self.img"""
        self.img[:,0:self.size_left]=self.left.image()
        self.img[:,self.size_left:self.n]=self.right.image()
        
#Sub classes to rapidly create specific masks

        
class Helix_Hat(DoubleMask):
    """class creating a mask containing a helix on the left part of the chip and a top hat
    on the right part. R corresponds to the radius of each of the masks in pixels"""
    def __init__(self,m,n,lbd,R,sigma,left_pos=(0,0),right_pos=(0,0)):
        DoubleMask.__init__(self,m,n,lbd, left_center= left_pos, right_center = right_pos)
        self.R=R
        
        if self.R>min(m//2,n//2):
            print("R out of range")
            self.R=min(m//2,n//2)
            
        self.left.setHelicoidal(R,left_pos[0],left_pos[1])
        self.right.topHat(R,sigma,right_pos[0],right_pos[1])
        self.update()

    def tilt(self,angle):
        """Adds a tilt for off-axis holography to the 2 masks"""
        left=Mask(self.m,self.size_left,self.lbd)     
        right=Mask(self.m,self.n-self.size_left,self.lbd)
        left.tilt(angle)
        left.setCircular(self.R,self.left_center[0]+self.m//2,self.left_center[1]+self.size_left//2)
        #Necessary inversion because going through 4f-system
        right.tilt(-1*angle)
        right.setCircular(self.R,self.right_center[0]+self.m//2,self.right_center[1]+(self.n-self.size_left)//2 )
        self.left+=left
        self.right+=right
        self.update()
        
        
    def moveLeft(self,x,y):
        """Moves the right centre from x,y"""
        
        (m,n)=self.left.img.shape
        #leftCenter represents the actual position of the right mask, in the coordinates
        #understood by Python. self.right_center represents the position of the mask
        #in relation to its center
        leftCenter = (m//2+self.left_center[0],n//2+self.left_center[1])
        if x>0 and x>max(m-self.R-leftCenter[0],0):
            x=m-self.R-leftCenter[0]
            print("x value out of range: x has been constrained to",x)
            
        if x<0 and x<min(0,self.R-leftCenter[0]):
            x=min(0,self.R-leftCenter[0])
            print("x value out of range: x has been constrained to",x)
            
        if y>0 and y>max(n-self.R-leftCenter[1],0):
            y=n-self.R-leftCenter[1]
            print("y value out of range: y has been constrained to",y)
            
        if y<0 and y<min(0,self.R-leftCenter[1]):
            y=min(0,self.R-leftCenter[1])
            print("y value out of range: y has been constrained to",y)
            
#        if np.abs(y)>min(self.left_center[1]-self.R,n-self.R-self.left_center[1]):
#            y=min(self.left_center[1]-self.R,n-self.R-self.left_center[1])*np.sign(y)
#            print "y value out of range: y has been constrained to",y
            
        out=np.zeros((m,n))        
        out[max(0,x):min(m,m+x),max(0,y):min(n,n+y)]=self.left.img[max(0,-x):min(m,m-x),max(0,-y):min(n,n-y)]
        self.left.img=out
        self.update()
        self.left_center=self.left_center[0]+x,self.left_center[1]+y
        
    def moveRight(self,x,y):
        """Moves the right centre from x,y"""

        (m,n)=self.right.img.shape
        #rightCenter represents the actual position of the right mask, in the coordinates
        #understood by Python. self.right_center represents the position of the mask
        #in relation to its center
        rightCenter = (m//2+self.right_center[0],n//2+self.right_center[1])
        
        if x>0 and x>max(m-self.R-rightCenter[0],0):
            x=m-self.R-rightCenter[0]
            print("x value out of range: x has been constrained to",x)
            
        if x<0 and x<min(0,self.R-rightCenter[0]):
            x=min(0,self.R-rightCenter[0])
            print("x value out of range: x has been constrained to",x)
            
        if y>0 and y>max(n-self.R-rightCenter[1],0):
            y=n-self.R-rightCenter[1]
            print("y value out of range: y has been constrained to",y)
            
        if y<0 and y<min(0,self.R-rightCenter[1]):
            y=min(0,self.R-rightCenter[1])
            print("y value out of range: y has been constrained to",y    )    
        
        out=np.zeros((m,n))
        out[max(0,x):min(m,m+x),max(0,y):min(n,n+y)]=self.right.img[max(0,-x):min(m,m-x),max(0,-y):min(n,n-y)]
        self.right.img=out
        self.update()
        
        self.right_center=self.right_center[0]+x,self.right_center[1]+y        
       
            
class Helix(EvMask):
    """class creating a helicoidal Mask"""
    def __init__(self,m,n,lbd,radius):
        EvMask.__init__(self,m,n,lbd)
        self.setHelicoidal(radius)

class TopHat(EvMask):
    """class creating a top hat Mask"""
    def __init__(self,m,n,lbd,radius,sigma):
        EvMask.__init__(self,m,n,lbd)
        self.topHat(radius,sigma)        