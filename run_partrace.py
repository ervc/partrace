#!/usr/bin/env python

# standard imports
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import multiprocessing as mp

import partrace as pt
import partrace.constants as const
from partrace.integrate import integrate

def main():
    # global
    fargodir = 'fargoout/fargo_mid'
    n = -1
    npart = 16 # number of particles

    # create mesh
    mesh = pt.create_mesh(fargodir,n=n)
    minr = mesh.yedges.min()
    maxr = mesh.yedges.max()
    minv = np.nanmin(np.abs(mesh.state['gasvx']))
    maxv = np.nanmax(np.abs(mesh.state['gasvx']))
    n = mesh.n['gasdens']

    # readin planet
    # planet = pt.create_planet(mesh,0,'Jupiter')
    planet = None

    # set up solver params
    t0 = 0
    tf = 1e2*const.YR
    maxdt = 1/50*const.TWOPI/mesh.get_Omega(minr,0,0)
    atol = np.zeros(6)
    atol[:3] += 1e-3*minr  # xtol is within 1e-3 of smallest r
    atol[3:] += 1e-3*maxv  # vtol is within 1e-3 of largest velocity
    rtol = 1e-6

    # constant partical parameters:
    rc = 10*const.AU  # central radius
    rw = 1*const.AU   # radial spread
    a = 10    # size in cm
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

def random_loc_helper(args,n):
    """Return a random location with radius centered on rc and with
    standard deviation rw.
    INPUTS
    ------
    args : tuple
        tuple containing rc and rw
    """
    rng = default_rng(seed=1234+n)
    rc,rw = args
    r0 = rng.normal(rc,rw)
    th0 = rng.uniform(-np.pi,np.pi)
    x0 = r0*np.cos(th0)
    y0 = r0*np.sin(th0)
    z0 = 0
    return x0,y0,z0

def helper_func(args):
    """
    rlargs = (rc,rw) for random location
    pargs = (mesh,a,rho_s) for particle creation
    intargs = (t0,tf,planet,kw) for integrator
      kw = keyword args for integrator
        maxstep
        atol
        rtol
    n = identifier
    """
    rlargs,pargs,intargs,n = args
    x0,y0,z0 = random_loc_helper(rlargs,n)
    mesh,a,rho_s = pargs
    p = pt.create_particle(mesh,x0,y0,z0,a,rho_s)
    savefile = f'particles/history_{n}_a{a}_newdrag_nodiff.npz'
    t0,tf,planet,kw = intargs
    sol = integrate(t0,tf,p,planet,savefile=savefile,**kw)
    return sol.status

def count_success(allsols):
    ns = 0
    for s in allsols:
        if s==0:
            ns+=1
    return ns

if __name__ == '__main__':
    main()