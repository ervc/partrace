import matplotlib.pyplot as plt
import numpy as np

import partrace as pt
from partrace import constants as const

fargodir = 'fargoout/fargo_mid'
partdir = 'particleout/a10_nodiff_test'
npart = 16

mesh = pt.create_mesh(fargodir,n=-1,quiet=True)

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

def get_stopping_time(x,y,a,rho_s=2):
    rhog = mesh.get_rho(x,y)
    cs = mesh.get_soundspeed(x,y)
    return a*rho_s/rhog/cs

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

def mesh_drdt(r,a):
    vx,vy,vz = mesh.get_gas_vel(r,0)
    vgas = np.sqrt(vx*vx + vy*vy)
    GM = float(mesh.variables['G'])*float(mesh.variables['MSTAR'])
    vkep = np.sqrt(GM/r)
    eta = 1-(vgas/vkep)**2
    St = calc_stokes_numbers(r,a)

    return -eta*vkep/(St + 1/St)


def main():
    fig,axd = plt.subplot_mosaic(
    """
    MMrr
    MMdd
    """,
    tight_layout=True)

    X = np.linspace(1.5*const.AU,20*const.AU,200)
    axd['M'].plot(X/const.AU,mesh_drdt(X,10),ls='--',c='k')

    for i in range(npart):
        ptout = np.load(f'{partdir}/{partname(i)}')
        times = ptout['times']
        x,y,z,vx,vy,vz = ptout['history'].T
        r = np.sqrt(x*x + y*y)
        st0 = get_stopping_time(x[0],y[0],10)
        drdt = get_drdt(x,y,vx,vy)
        axd['M'].plot(r/const.AU,drdt,alpha=0.3,c='b')
        axd['r'].plot(times/const.YR,r/const.AU,c='b',alpha=0.4)
        label = ''
        if i == 0:
            label = 'stopping time'
        axd['r'].axvline(st0/const.YR,ls=':',c='grey',label=label)
        axd['d'].plot(times/const.YR,drdt,c='b',alpha=0.4)
        axd['d'].axvline(st0/const.YR,ls=':',c='grey',label=label)
        print(f'stopping time = {st0/const.YR}')

    axd['r'].legend()
    axd['M'].set(xlabel='r [au]',ylabel = 'dr/dt [cm/s]')
    axd['d'].set(xlabel='time [yrs]',ylabel='dr/dt [cm/s]',xscale='log')
    axd['r'].set(ylabel='r [au]',xscale='log')

    plt.show()

if __name__ == '__main__':
    main()
