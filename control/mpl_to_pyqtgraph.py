# -*- coding: utf-8 -*-
# author: Sebastian Höfer
"""
taken from
https://github.com/honkomonk/pyqtgraph_sandbox

This is an example how to use an ImageView with Matplotlib Colormaps (cmap).

The function 'cmapToColormap' converts the Matplotlib format to the internal
format of PyQtGraph that is used in the GradientEditorItem. The function
itself has no dependencies on Matplotlib! Hence the weird if clauses with
'hasattr' instead of 'isinstance'.
"""


import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph

import collections


def cmapToColormap(cmap, nTicks=16):
    """
    Converts a Matplotlib cmap to pyqtgraphs colormaps. No dependency on
    matplotlib.

    Parameters:
    *cmap*: Cmap object. Imported from matplotlib.cm.*
    *nTicks*: Number of ticks to create when dict of functions is used.
    Otherwise unused.
    """

    # Case #1: a dictionary with 'red'/'green'/'blue' values as list of ranges
    # (e.g. 'jet')
    # The parameter 'cmap' is a 'matplotlib.colors.LinearSegmentedColormap'
    # instance ...
    if hasattr(cmap, '_segmentdata'):
        colordata = getattr(cmap, '_segmentdata')
        if ('red' in colordata) and isinstance(
                colordata['red'], collections.Sequence):

            # collect the color ranges from all channels into one dict to get
            # unique indices
            posDict = {}
            for idx, channel in enumerate(('red', 'green', 'blue')):
                for colorRange in colordata[channel]:
                    posDict.setdefault(
                        colorRange[0], [-1, -1, -1])[idx] = colorRange[2]

            indexList = sorted(posDict.keys())
            # interpolate missing values (== -1)
            for channel in range(3):  # R,G,B
                startIdx = indexList[0]
                emptyIdx = []
                for curIdx in indexList:
                    if posDict[curIdx][channel] == -1:
                        emptyIdx.append(curIdx)
                    elif curIdx != indexList[0]:
                        for eIdx in emptyIdx:
                            rPos = (eIdx - startIdx) / (curIdx - startIdx)
                            vStart = posDict[startIdx][channel]
                            vRange = (
                                posDict[curIdx][channel] -
                                posDict[startIdx][channel])
                            posDict[eIdx][channel] = rPos * vRange + vStart
                        startIdx = curIdx
                        del emptyIdx[:]
            for channel in range(3):  # R,G,B
                for curIdx in indexList:
                    posDict[curIdx][channel] *= 255

            rgb_list = [[i, posDict[i]] for i in indexList]

        # Case #2: a dictionary with 'red'/'green'/'blue' values as functions
        # (e.g. 'gnuplot')
        elif ('red' in colordata) and isinstance(colordata['red'],
                                                 collections.Callable):
            indices = np.linspace(0., 1., nTicks)
            luts = [
                np.clip(
                    np.array(
                        colordata[rgb](indices),
                        dtype=np.float),
                    0,
                    1) *
                255 for rgb in (
                    'red',
                    'green',
                    'blue')]
            rgb_list = zip(indices, list(zip(*luts)))

    # If the parameter 'cmap' is a 'matplotlib.colors.ListedColormap'
    # instance, with the attributes 'colors' and 'N'
    elif hasattr(cmap, 'colors') and hasattr(cmap, 'N'):
        colordata = getattr(cmap, 'colors')
        # Case #3: a list with RGB values (e.g. 'seismic')
        if len(colordata[0]) == 3:
            indices = np.linspace(0., 1., len(colordata))
            scaledRgbTuples = [
                (rgbTuple[0] * 255,
                 rgbTuple[1] * 255,
                    rgbTuple[2] * 255) for rgbTuple in colordata]
            rgb_list = zip(indices, scaledRgbTuples)

        # Case #4: a list of tuples with positions and RGB-values
        # (e.g. 'terrain') -> this section is probably not needed anymore!?
        elif len(colordata[0]) == 2:
            rgb_list = [(idx, (vals[0] * 255, vals[1] * 255, vals[2] * 255))
                        for idx, vals in colordata]

    # Case #X: unknown format or datatype was the wrong object type
    else:
        raise ValueError("[cmapToColormap] Unknown cmap format or not a cmap!")

    # Convert the RGB float values to RGBA integer values
    return list([(pos, (int(r), int(g), int(b), 255))
                 for pos, (r, g, b) in rgb_list])


class MplCmapImageView(pyqtgraph.ImageView):
    def __init__(self, additionalCmaps=[], setColormap=None, **kargs):
        super(MplCmapImageView, self).__init__(**kargs)

        self.gradientEditorItem = self.ui.histogram.item.gradient

        self.activeCm = "grey"
        self.mplCmaps = {}

        if len(additionalCmaps) > 0:
            self.registerCmap(additionalCmaps)

        if setColormap is not None:
            self.gradientEditorItem.restoreState(setColormap)

    def registerCmap(self, cmapNames):
        """ Add matplotlib cmaps to the GradientEditors context menu"""
        self.gradientEditorItem.menu.addSeparator()
        savedLength = self.gradientEditorItem.length
        self.gradientEditorItem.length = 100

        # iterate over the list of cmap names and check if they're avaible in
        # MPL
        for cmapName in cmapNames:
            if not hasattr(matplotlib.cm, cmapName):
                print(
                    '[MplCmapImageView] Unknown cmap name: \'{}\'. Your '
                    'Matplotlib installation might be '
                    'outdated.'.format(cmapName))
            else:
                # create a Dictionary just as the one at the top of
                # GradientEditorItem.py
                cmap = getattr(matplotlib.cm, cmapName)
                self.mplCmaps[cmapName] = {
                    'ticks': cmapToColormap(cmap), 'mode': 'rgb'}

                # Create the menu entries
                # The following code is copied from
                # pyqtgraph.ImageView.__init__() ...
                px = QtGui.QPixmap(100, 15)
                p = QtGui.QPainter(px)
                self.gradientEditorItem.restoreState(self.mplCmaps[cmapName])
                grad = self.gradientEditorItem.getGradient()
                brush = QtGui.QBrush(grad)
                p.fillRect(QtCore.QRect(0, 0, 100, 15), brush)
                p.end()
                label = QtGui.QLabel()
                label.setPixmap(px)
                label.setContentsMargins(1, 1, 1, 1)
                act = QtGui.QWidgetAction(self.gradientEditorItem)
                act.setDefaultWidget(label)
                act.triggered.connect(self.cmapClicked)
                act.name = cmapName
                self.gradientEditorItem.menu.addAction(act)
        self.gradientEditorItem.length = savedLength

    def cmapClicked(self, b=None):
        """onclick handler for our custom entries in the GradientEditorItem's
        context menu"""
        act = self.sender()
        self.gradientEditorItem.restoreState(self.mplCmaps[act.name])
        self.activeCm = act.name


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    import matplotlib.cm

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication([])

        # Create window with ImageView widget
        win = pyqtgraph.GraphicsLayoutWidget()
        win.resize(800, 800)
        win.show()

        imv = pyqtgraph.ImageItem()

        view = win.addViewBox()
        view.addItem(imv)
        win.setCentralWidget(view)
        win.show()
        win.setWindowTitle(
            'Example: Matplotlib colormaps in pyqtgraph ImageView')

        # Create random data set with noisy signals
        data = np.random.normal(size=(200, 200))
        gradient = np.linspace(0., 10., 200)
        data += gradient[:, np.newaxis]

        # Convert a matplotlib colormap into a list of (tickmark, (r,g,b))
        # tuples
        pos, rgba_colors = zip(*cmapToColormap(matplotlib.cm.cubehelix))
        # Set the colormap
        pgColormap = pyqtgraph.ColorMap(pos, rgba_colors)
        imv.setLookupTable(pgColormap.getLookupTable())

        # Display the data
        imv.setImage(data)

        QtGui.QApplication.instance().exec_()
