"""
Useful constants imported in other modules
"""

import numpy as np

#: :obj:`float`
#: CGS Mass of Sun
MSUN = 1.98847e33

#: :obj:`float`
#: CGS Mass of Jupiter
MJUP = 1.898e30

#: :obj:`float`
#: CGS Mass of Earth
MEARTH = 5.97e27

#: :obj:`float`
#: CGS Gravitational constant
G = 6.67430e-8

#: :obj:`float`
#: CGS Gravitational constant times mass of Sun
GMSUN = G*MSUN

#: :obj:`float`
#: 1 au in cm
AU = 1.495978707e13

#: :obj:`float`
#: CGS Boltzmann constant
BK = 1.380649e-16

#: :obj:`float`
#: Mass of proton in g
MH = 1.67262192369e-24

#: :obj:`float`
#: 1 Year in seconds
YR = 3.1557600e7

#: :obj:`float`
#: Average molecular mass in g
MBAR = 2.3*MH

#: :obj:`float`
#: Pi
PI = np.pi
# except ImportError:
# 	PI = 3.14519265358979323846264338327950288419716

#: :obj:`float`
#: 2pi
TWOPI = 2*PI