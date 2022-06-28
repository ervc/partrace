import numpy as np
import astropy.constants as const
from astropy import units as u

MSUN = const.M_sun.cgs.value
G = const.G.cgs.value
GM = G*MSUN
AU = const.au.cgs.value
BK = const.k_B.cgs.value
MH = const.m_p.cgs.value
MBAR = 2.3*MH
YR = ((1*u.yr).to(u.s)).value
PI = np.pi
TWOPI = 2*PI