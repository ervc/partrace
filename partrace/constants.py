"""
Useful constants for other libraries
"""


MSUN = 1.98847e33
MJUP = 1.898e30
MEARTH = 5.97e27
G = 6.67430e-8
GMSUN = G*MSUN
AU = 1.495978707e13
BK = 1.380649e-16
MH = 1.67262192369e-24
YR = 3.1557600e7
MBAR = 2.3*MH
try:
	import numpy as np
	PI = np.pi
except ImportError:
	PI = 3.14519265358979323846264338327950288419716
TWOPI = 2*PI