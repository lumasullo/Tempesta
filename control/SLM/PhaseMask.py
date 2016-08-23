# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:39:18 2016

@author: aurelien.barbotin
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
from scipy import signal as sg
"""
SLM model : X10468-02
792 x 600 pixels
800+/-50nm maximum 785nm
"""

n=601   #width of the phase pattern in pixels
m=601   #height of the phase pattern in pixels
pad=600 #padding for the fft, in pixels
r=300   #radius of the phase pattern
s_pix=0.02  #pixel size in mm (SLM)
lbd=785*10**-6    #wavelength in mm
f=4.7    #focal length in mm (20x/1.0 zeiss)

def createHelMask(m,n,r,u=0,v=0,rotation=True):
    """This function generates an helicoidal phase mask centered in (u,v) where (0,0) corresponds to
    the center of the image"""
    x,y=np.ogrid[-m//2-u:m//2-u,-n//2-v:n//2-v]
    d2=x**2+y**2
    theta=np.arctan2(x,y)
    theta[d2>r**2]=0
    theta%=2*math.pi
    #To change the rotation direction of the helix
    if(rotation):
        mask_bis=np.ones((m,n))*np.pi*2
        mask_bis[d2>r**2]=0
        theta = mask_bis - theta
    return theta

def topHat(sizex,sizey,r,sigma,u,v):
    """Creates a top hat mask with radius r. The input beam is supposed gaussain
    with a standard deviation sigma."""
    mask=np.zeros((sizex,sizey),dtype="float")    
    #Looking for the middle of the gaussian intensity distribution:
    mid_radius=sigma*np.sqrt(2*np.log(2/(1+np.exp( -r**2/(2*sigma**2) ) ) ) )    
    y,x=np.ogrid[-sizex//2-u:sizex//2-u,-sizey//2-v:sizey//2-v]
    d2=x**2+y**2
    ring=(d2<=r**2)*(d2>mid_radius**2)
    mask[ring]=np.pi
    return mask    

    
    
def create8bitHelMask(n,m,r):
    """This function generates an helicoidal phase mask with 8bit integer by computing for 
    each pixel in an n*m image the angle theta with the x axis
    returns """
    img=np.zeros((m,n),dtype=np.uint8)
    phase_img=np.zeros((m,n),dtype=np.complex_)
    for u in range(m):
        for v in range(n):
            theta=0
            xp=v-n//2
            yp=u-m//2
            d=np.sqrt(xp**2+yp**2) 
            if d == 0:
                img[u,v]=0
                phase_img[u,v]=1
            elif d>=r:
                img[u,v]=0
                phase_img[u,v]=0
            else:
                cos_theta=xp/d
                sin_theta=yp/d
                
                theta=np.arccos(cos_theta)
                if sin_theta<0:
                    theta=2*math.pi-theta
                theta/=(2*math.pi)  #conversion theta from 0 to 255
                theta*=255    
                theta=np.round(theta)
                img[u,v]=theta
                phase_img[u,v]=cmath.exp(1j*theta)
    return phase_img,img

def createGsMask(sigma,sizex,sizey,xshift=0,yshift=0):
    """Creates a gaussian mask of size sizex,sizey and std sigma, centered in (xshift, yshift)"""
    if sizex//2==0:
        sizex+=1
    if sizey//2==0:
        sizey+=1
    xshift=int(xshift)
    yshift=int(yshift)
    mask=np.zeros((sizey,sizex))
    centerx=sizex//2
    centery=sizey//2
    for u in range(sizey):
        for v in range(sizex):
            d2=(u-centery-yshift)**2+(v-centerx-xshift)**2
            mask[u,v]=1/np.sqrt(2*math.pi)*np.exp(-d2/(2*sigma**2) )
    return mask
  
def gsMask(sigma,sizex,sizey,xCenter=-1,yCenter=-1):
    """Creates a gaussian mask of size sizex,sizey and std sigma, centered in (xshift, yshift)"""
    if xCenter<0:
        xCenter=sizex//2
    if yCenter<0:
        yCenter=sizey//2
    x,y=np.ogrid[-yCenter:sizey-yCenter,-xCenter:sizex-xCenter]
    val= -(x**2+y**2)/(2*sigma**2)
    mask=np.exp(val)
    return mask
      
def createTiltMask(sizex,sizey,lbd,alpha):
    """Creates a phase pattern to transform an incident beam with an angle alpha into an orthogonal beam"""
    mask=np.indices((sizey,sizex),dtype="float")[1,:,:]
    mask*=s_pix
    lbd*=10**-6 #conversion in mm
    alpha=np.deg2rad(alpha) #conversion in radian
    period=2*math.pi*alpha/lbd
    print(period,"pixels",period/s_pix,"mm")
    mask*=period
    tilt=sg.sawtooth(mask)+1
    tilt*=255/2
    return tilt
    
def displayMask(img):
    """This funciton displays an image and the corresponding colorbar"""
    fig,ax = plt.subplots()
    cax=ax.imshow(img)
    ax.set_title("Phase mask (rad)")
    fig.colorbar(cax)
    plt.show()
    
def myFFT(img,n,m):
    """Returns the FFT of img with a m x n resolution"""
    (y,x)=img.shape
    padx=0
    pady=0
    if y<m:
        pady=(m-y)//2
    if x<n:
        padx=(n-x)//2
    pad=( (max(m-pady-1,0), pady),( max(n-padx-1,0),padx ))
    img=np.lib.pad(img,pad, 'constant', constant_values=(0,0))
    fftmask=np.fft.fft2(img)
    fftmask=np.fft.fftshift(fftmask)
    return fftmask
    
#New functions returning 0-2Pi masks instead of 0-255    
    
def setCircular(mask,radius,x=-1,y=-1):
    """transforms an array-like phase pattern into a circular pattern"""
    (m,n)=mask.shape
    if x<0:
        centerx=m//2
    else:
        centerx=x
    if y<0:
        centery=n//2
    else:
        centery=y
        
    x,y = np.ogrid[-centerx:m-centerx, -centery:n-centery]
    mask_bin = x*x + y*y <= radius*radius
    result=np.zeros((m,n))
    result[mask_bin]=mask[mask_bin]
    return result
 
def setHole(mask,radius):
    """fills the centre of a phase pattern with zeros """
    (m,n)=mask.shape
    for u in range(m):
        for v in range(n):
            up=u-m//2
            vp=v-n//2
            d=np.sqrt(up**2+vp**2)
            if d<radius:
                mask[u,v]=0
    
def holo(intensity,phase,x,y):
    pattern=intensity*np.exp(1j*phase)
    return np.absolute(myFFT(pattern,x,y))

def annularHel(n,m,rmin,rmax,angle=0,binaire=0):
    """OBSOLETE """
    return anHel(n,m,rmin,rmax,angle=0,binaire=0)

def annular(n,m,rmin,rmax,phase):
    """This function generates an annular phase mask between rmin and rmax, with
    the value phase"""
    img=np.zeros((m,n))
    for u in range(m): 
        for v in range(n):
            xp=v-n//2
            yp=u-m//2
            d=np.sqrt(xp**2+yp**2) 
            if d>=rmax or d<rmin or d==0:
                img[u,v]=0
            else:
                img[u,v]=phase
    return img

def anHel(n,m,rmin,rmax,angle=0,binaire=0):
    """This function generates an helicoidal phase mask, hopefully in a more convenient
    manner than the previous one"""
    x,y=np.ogrid[-n//2:n//2,-m//2:m//2]
    d2=x**2+y**2
    theta=np.arctan2(x,y)
    theta+=angle
    
    if binaire%2==1:
        theta=np.pi-theta
        
    theta[d2>rmax**2]=0
    theta[d2<=rmin**2]=0
    theta%=2*math.pi
    return theta
                
def radius(n,sigma,R):
    """returns the radius rn corresponding to the circle containing the fraction 1/n of the energy of the gaussian
    with std sigma contained in the circle of radius R"""
    
    if n==0:
        return 0
    return sigma*np.sqrt( 2*np.log(n/ (n-1+np.exp(-R**2/(2*sigma**2))) ) )

def multipleHel(xsize,ysize,N,R,sigma):
    out=np.zeros((xsize,ysize))
    rmin=0
    for u in range(N):
        frac=(u+1)/N
        rmax=radius(N/(u+1),sigma,R)
        out+=annularHel(ysize,xsize,rmin,rmax,math.pi*frac,u)
        rmin=radius(N/(u+1),sigma,R)
    
    return out

def compositeMask(n,m,rmax,sigma,N,energy_frac=2/3,n_rings=1):
    """Create a circular mask made of a top hat mask in its center and of a helicoidal annulus in its periphery
    n,m: dimensions of the squared window
    energy_frac: fraction of the beam energy sent through the top hat"""
    #r1 corresponds to the radius of the tophat phase mask
    r1=radius(1/energy_frac,sigma,rmax)
    part1=topHat(n,m,r1,sigma)
    
    #splits the zone comprised between r1 and rmax into n zones with an equal incident intensity
    part2=np.zeros((m,n))
    
    for u in range(N):
        energy_portion= energy_frac+(1-energy_frac)*(u+1)/N
        r=radius(1/energy_portion , sigma,rmax)
        print("Radius:",r)
        part2+=annularHel(n,m,r1,r,n_rings*2*math.pi*u/N)
        r1=r
    out=part1+part2
    return out
    
def ring(n,m,rmin,rmax,xCenter=-1,yCenter=-1):
    """Creates a ring between rmin and rmax"""
    out=np.zeros((m,n))    
    y,x=np.ogrid[-m//2:m//2,-n//2:n//2]
    d2=x**2+y**2
    ring=(d2<=rmax**2)*(d2>rmin**2)
    out[ring]=1
    return out
    
def spiral(m,n,rmax,sigma,N,energy_frac=2/3,factor=0.6):
    """Creates a spiral turning with a 90deg angle to destroy the symetry in the phase plate"""
    #r1 corresponds to the radius of the tophat phase mask
    r1=0
    part1=np.zeros((m,n))
    if energy_frac!=0:
        r1=radius(1/energy_frac,sigma,rmax)
        part1=topHat(m,n,r1,sigma)
    
    #splits the zone comprised between r1 and rmax into n zones with an equal incident intensity
    part2=np.zeros((m,n))
    
    for u in range(N):
        energy_portion= energy_frac+(1-energy_frac)*(u+1)/N
        r=radius(1/energy_portion , sigma,rmax)
        #a correction is added in accordance with the experimental data
        correc=ring(n,m,r1,r)*(0.5-factor*(u+1)/N)
#        correc=ring(n,m,r1,r)*(0.5-0.2*np.pi*(u+1)/N)

        """!!! Add Correc!!!"""        
        
        part2+=anHel(m,n,r1,r,0.5*math.pi*u/N)+correc
        r1=r
    out=part1+part2
    return out  
    
def computeIntAtDelta(lbd,f,img,radius,sigma,delta):
    """Computes the integral of the complex beam after the SLM ponderated by the 
    phase mask corresponding to a height of delta"""
    (m,n)=img.shape
    mask=createGsMask(sigma,n,m)
    setCircular(mask,radius)
    mask=mask.astype(np.complex)
    for u in range(m):
        for v in range(n):
            up=u-m//2
            vp=v-n//2
            d=np.sqrt(up**2+vp**2)
            if d>=radius:
                mask[u,v]=0
            else:
                d*=s_pix
                mask[u,v]*=np.exp(1j*d**2*delta*2*math.pi/lbd/4/f**2)
                    
    mask*=np.exp(1j*img)
    return np.absolute(np.sum(mask))
   
def intensityProfile(lbd,f,img,radius,sigma,N):
        out=np.ones(2*N+1)
        for u in range(2*N+1):
            out[u]=computeIntAtDelta(lbd,f,img,radius,sigma,(u-N)*lbd/2 )
            print("iteration:",u)
        plt.figure()
        plt.plot(out)
        return out   
        
        
