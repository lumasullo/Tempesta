# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:25:56 2016

@author: aurelien.barbotin
"""
import pyqtgraph as pg
import numpy as np


class TableWidget(pg.TableWidget):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.verticalHeader().hide()
        f = [('Rueda 1', 4, 'ZET642NF',    'Notch', ''),
             ('Rueda 2', 5, 'ET700/75m',   'Bandpass', 'Alexa647, Atto655'),
             ('Rueda 3', 6, 'FF01-725/40', 'Bandpass', 'Alexa700 (2 colores)'),
             ('Rueda 4', 1, '',            '', ''),
             ('Rueda 5', 2, 'FF03-525/50', 'Bandpass', 'GFP'),
             ('Rueda 6', 3, '',            'Bandpass', ''),
             ('Tubo', '',   'FF01-582/75', 'Bandpass', 'Alexa532, Alexa568, '
                                                       'Alexa700, \nAtto550, '
                                                       'Atto565, Nile Red')]
        data = np.array(f, dtype=[('Ubicación', object),
                                  ('Antiposición', object),
                                  ('Filtro', object),
                                  ('Tipo', object),
                                  ('Fluoróforos', object)])
        self.setData(data)
        self.resizeRowsToContents()