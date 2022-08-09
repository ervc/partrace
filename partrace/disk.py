"""
Useful analytic functions for a typical disk (largely deprecated)
"""


import numpy as np
from .constants import *

ALPHA = 1e-3
PERTURBED = False

def get_Tmid(r):
    '''Tmid powerlaw'''
    T0 = 150 #K
    q = 1
    return T0*(r/AU)**(-q)

def get_T(r,z):
    '''vertically isothermal approx'''
    return get_Tmid(r)

def get_soundspeed(r):
    '''soundspeed in gas'''
    T = get_Tmid(r)
    return np.sqrt(BK*T/MBAR)

def get_Omegak(r,z):
    '''keplerian frequency'''
    r3 = (r*r + z*z)**(3/2)
    return np.sqrt(GM/r3)

def get_scaleheight(r):
    '''scaleheight of gas'''
    cs = get_soundspeed(r)
    om = get_Omegak(r,0)
    return cs/om

def unperturbed_sigma(r):
    '''unperturbed surface density'''
    sig0 = 2000 # g cm-2
    p = 1
    return sig0*(r/AU)**(-p)

def perturbed_sigma(r):
    '''Perturbed disk profile a la Alarcon et al 2022'''
    r0 = 4.5*AU      # gap center
    rw = 0.16*r      # gap width
    rwh = 0.5*rw    # gap half width
    dep = 0.1     # gap depth
    unpert = unperturbed_sigma(r)
    pert = unpert*(1-(1-dep)*np.exp(-(r-r0)**2/2/rw**2))
    return pert

def get_sigma(r):
    '''Surface density'''
    if PERTURBED:
        return perturbed_sigma(r)
    else:
        return unperturbed_sigma(r)

def get_rho(r,z):
    '''local density (from surface density)'''
    sig = get_sigma(r)
    H = get_scaleheight(r)
    rhomid = sig/(np.sqrt(TWOPI)*H)
    zterm = np.exp(-0.5*(z/H)**2)
    return rhomid*zterm

def get_pressure(r,z):
    '''local pressure'''
    rho = get_rho(r,z)
    cs = get_soundspeed(r)
    return rho*cs*cs

def get_dpdr(r,z):
    '''radial pressure grad approximation'''
    epsilon = r*0.02 # small variation of 1%
    Ppe = get_pressure(r+epsilon,z)
    Pme = get_pressure(r-epsilon,z)
    return (Ppe - Pme)/(2*epsilon)

def get_eta(r,z):
    '''gas velocity differential'''
    rho = get_rho(r,z)
    omk = get_Omegak(r,z)
    vk = omk*r
    dpdr = get_dpdr(r,z)
    return -r/rho/vk**2 * dpdr

def get_velocity(r,z):
    '''get subkeplerian gas velocity'''
    omk = get_Omegak(r,z)
    vk = omk*r
    eta = get_eta(r,z)
    return vk*(1-eta)**(1/2)

def get_diffusivity(r,z,alpha=ALPHA):
    '''return the diffusivity of the gas'''
    H = get_scaleheight(r)
    cs = get_soundspeed(r)
    return alpha*cs*H