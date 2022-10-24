#!/usr/bin/env python

# standard imports
import numpy as np
from numpy.random import default_rng
import multiprocessing as mp
import os
import subprocess
from time import time

# partrace imports
import partrace as pt
import partrace.constants as const
from partrace.integrate import integrate

# constants
DIFFUSION = False
FARGODIR = '../fargo/outputs/alpha3d_moreaz_cont'
OUTPUTDIR = 'particleout/alpha3d_moreaz_nodiff_a1_n120'
A = 1 # cm
TF = 1e4*const.YR
SOLVER = 'DOP'
MAXSTEP = False

# make the output directory if doesn't exist
if not os.path.exists(OUTPUTDIR):
    subprocess.run(['mkdir',OUTPUTDIR])


def main():
    start = time()

    # global
    fargodir = FARGODIR
    n = 120
    # number of particles
    npart = 16

    # create mesh
    mesh = pt.create_mesh(fargodir,n=n)
    minr = mesh.yedges.min()
    maxr = mesh.yedges.max()
    print(f'{minr/const.AU = }\t{maxr/const.AU = }')
    minv = np.nanmin(np.abs(mesh.state['gasvx']))
    maxv = np.nanmax(np.abs(mesh.state['gasvx']))
    n = mesh.n['gasdens']

    # readin planet
    planet = pt.create_planet(mesh,0,'Jupiter')
    # planet = None

    # set up solver params
    t0 = 0
    tf = TF
    if MAXSTEP:
        maxdt = 1/50*const.TWOPI/mesh.get_Omega(minr,0,0)
    else:
        maxdt = np.inf
    atol = np.zeros(6)
    atol[:3] += 1e-3*minr  # xtol is within 1e-3 of smallest r
    atol[3:] += 1e-3*maxv  # vtol is within 1e-3 of largest velocity
    rtol = 1e-6

    # constant partical parameters:
    rc = 9*const.AU  # central radius
    rw = 0.5*const.AU   # radial spread
    a = A    # size in cm
    rho_s = 2  # density in g/cm^3

    rlargs = (rc,rw)
    pargs = (mesh,a,rho_s)
    kw = {'max_step':maxdt,'atol':atol,'rtol':rtol}
    intargs = (t0,tf,planet,kw)
    
    # r0 = rng.normal(rc,rw)
    # th0 = rng.uniform(-np.pi,np.pi)
    # x0 = r0*np.cos(th0)
    # y0 = r0*np.sin(th0)
    # z0 = 0

    # part = pt.create_particle(mesh,x0,y0,z0,a=100)
    # part.get_drag_coeff()

    with mp.Pool(npart) as pool:
        N = np.arange(npart)
        allargs = [(rlargs,pargs,intargs,n) for n in N]
        allsols = pool.map(helper_func,allargs)
    print('all done:')
    print('statuses: ',allsols)
    print(f'successes : {count_success(allsols)}/'
        +f'{len(allsols)}')

    end = time()
    time2run = end-start
    return time2run

def random_loc_helper(args,n):
    """Return a random location with radius centered on rc and with
    standard deviation rw.
    Parameters
    ----------
    args : tuple
        tuple containing rc and rw
    """
    rng = default_rng(seed=1234+n)
    rc,rw = args
    r0 = rng.normal(rc,rw)
    th0 = rng.uniform(-np.pi,np.pi)
    h0 = rng.uniform(-0.05,0.05)
    x0 = r0*np.cos(th0)
    y0 = r0*np.sin(th0)
    z0 = r0*h0
    return x0,y0,z0

def helper_func(args):
    """
    Helper funtion to allow for integration using multiprocessing map, which only takes one
    argument. I could use starmap in the future, but this is just as good.
    
    Parameters
    ----------
    rlargs : tuple
        (rc,rw) for random location
    pargs : tuple
        (mesh,a,rho_s) for particle creation
    intargs : tuple
        (t0,tf,planet,kw) for integrator where
        kw = keyword args for integrator
          maxstep
          atol
          rtol
    n : int
        particle identifier number
    """
    rlargs,pargs,intargs,n = args
    x0,y0,z0 = random_loc_helper(rlargs,n)
    mesh,a,rho_s = pargs
    p = pt.create_particle(mesh,x0,y0,z0,a,rho_s)
    savefile = f'{OUTPUTDIR}/history_{n}.npz'
    t0,tf,planet,kw = intargs
    sol = integrate(t0,tf,p,planet,savefile=savefile,diffusion=DIFFUSION,**kw)
    return sol.status

def count_success(allsols):
    ns = 0
    for s in allsols:
        if s==0:
            ns+=1
    return ns

if __name__ == '__main__':
    t = main()
    if not os.path.exists('time_to_run.out'):
        with open('time_to_run.out','w+') as f:
            f.write('time [hrs], partsize [cm], tf [yr], solver, maxstep [s], fargodir, outputdir, diffusion\n')
    with open('time_to_run.out','a+') as f:
        f.write(f'{t/3600:.3f}, {A}, {TF/const.YR:.1e}, {SOLVER}, {MAXSTEP}, {FARGODIR}, {OUTPUTDIR}, {DIFFUSION}\n')

