# Tempesta
Measurement control and analysis for optical microscopy. Adaption of Fede Barabas' Tormenta software https://github.com/fedebarabas/Tormenta.

This version of Tempesta can drive a point-scanning microscope like a confocal or STED microscope. It includes a feature to control a Spatial Light Modulator (SLM), adapted from wavefrontshaping.net tutorial:
http://wavefrontshaping.net/index.php/57-community/tutorials/spatial-lights-modulators-slms/124-how-to-control-a-slm-with-python

### Dependencies
 - numpy
 - scipy
 - h5py
 - PyQt4
 - pyqtgraph dev
  - https://github.com/pyqtgraph/pyqtgraph
 - tifffile
  - http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd
 - lantz
  - https://github.com/hgrecco/lantz
 -pint
 - openCV (for webcam)
 - wxpython (for slmpy)
 - pyserial (to control OneFiveLaser)
