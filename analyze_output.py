import matplotlib.pyplot as plt
import numpy as np

import partrace as pt
from partrace import constants as const

fargodir = 'fargoout/fargo_mid'
partdir = 'particleout/a10_yesdiff_test'
npart = 16

def partname(n):
    return f'particle_{n}.npz'

def get_drdt(x,y,vx,vy):
    r = np.sqrt(x*x + y*y)
    return (x*vx + y*vy)/r

def calc_stokes_numbers(r,a,rho_s=2):
    sigma0 = 209 # g cm^-2       
    aspectratio = 0
    flaringindex = 0
    sigmas = np.ones_like(r)*sigma0  #r^0
    Hs = 0.05*r                      #r^1
    GM = const.GMSUN
    Omegas = np.sqrt(GM/r/r/r)       #r^-3/2
    cs = Hs*Omegas                   #r^-1/2
    rhos = sigmas/Hs/np.sqrt(const.TWOPI) #r^-1

    return a*rho_s/(rhos*cs)*Omegas

def analytic_drdt(r,a):
    sigma0 = 209 # g cm^-2       
    aspectratio = 0
    flaringindex = 0
    sigmas = np.ones_like(r)*sigma0 #r^0
    Hs = 0.05*r                      #r^1
    GM = const.GMSUN
    Omegas = np.sqrt(GM/r/r/r)       #r^-3/2
    cs = Hs*Omegas                   #r^-1/2
    rhos = sigmas/Hs/np.sqrt(const.TWOPI) #r^-1
    Ps = cs*cs*rhos

    St = calc_stokes_numbers(r,a)
    vks = r*Omegas
    dPdr = np.gradient(Ps,r)
    eta = -r/rhos/vks/vks * dPdr

    return -eta*vks/(St + 1/St)

def main():
    fig,axd = plt.subplot_mosaic(
    """
    MMrr
    MMdd
    """,
    tight_layout=True)

    X = np.linspace(1.5*const.AU,20*const.AU,200)
    axd['M'].plot(X/const.AU,analytic_drdt(X,10),ls='--',c='k')

    for i in range(npart):
        ptout = np.load(f'{partdir}/{partname(i)}')
        times = ptout['times']
        x,y,z,vx,vy,vz = ptout['history'].T
        r = np.sqrt(x*x + y*y)
        drdt = get_drdt(x,y,vx,vy)
        axd['M'].plot(r/const.AU,drdt)
        axd['r'].plot(times/const.YR,r/const.AU,c='k',alpha=0.7)
        axd['d'].plot(times/const.YR,drdt,c='k',alpha=0.7)

    plt.show()

if __name__ == '__main__':
    main()
