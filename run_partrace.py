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
import partrace.partraceio as ptio

# read in arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('-n','--nproc',type=int,default=1)
args = parser.parse_args()
nproc = args.nproc

# get parameters from input file
params = ptio.read_input(args.infile)

# constants
DIFFUSION = params['diffusion']
FARGODIR = params['fargodir']
OUTPUTDIR = params['outputdir']
A = params['partsize']
RHOS = params['partdens']
T0 = params['t0']
TF = params['tf']
NOUT = params['nout']

LOCS = ptio.read_locations(params['partfile'])
NPART = len(LOCS)

MAXSTEP = True
SOLVER = 'Radau'

# make the output directory if doesn't exist
if not os.path.exists(OUTPUTDIR):
    subprocess.run(['mkdir',OUTPUTDIR])

ptio.write_paramsfile(params,f'{OUTPUTDIR}/params.ini')

with open(f'{OUTPUTDIR}/variables.out','w+') as f:
    f.write(f'fargodir = {FARGODIR}\n')
    f.write(f't0 = {T0}\n')
    f.write(f'tf = {TF}\n')
    f.write(f'nout = {NOUT}\n')
    f.write(f'nparts = {NPART}\n')

def main():
    print('Read infile: ',args.infile)
    print('Fargodir: ',FARGODIR)
    print('partfile: ',params["partfile"])
    print('number of particles: ',NPART)
    print('particle 0 is at:')
    print(LOCS[0])
    # check number of processors
    print('cpus availables = ',nproc)

    # global
    fargodir = FARGODIR
    n = NOUT
    # number of particles
    npart = NPART

    # create mesh
    mesh = pt.create_mesh(fargodir,n=n,regrid=True)
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
    t0 = T0
    tf = TF
    if MAXSTEP:
        maxdt = 1/10*const.TWOPI/mesh.get_Omega(minr,0,0)
    else:
        maxdt = np.inf
    atol = np.zeros(6)
    atol[:3] += 1e-3*minr  # xtol is within 1e-3 of smallest r
    atol[3:] += 1e-3*maxv  # vtol is within 1e-3 of largest velocity
    rtol = 1e-6

    # constant partical parameters:
    a = A    # size in cm
    rho_s = RHOS  # density in g/cm^3

    locs = LOCS
    pargs = (mesh,a,rho_s)
    kw = {'max_step':maxdt,'atol':atol,'rtol':rtol}
    intargs = (t0,tf,planet,kw)

    with mp.Pool(processes=nproc) as pool:
        N = np.arange(npart)
        allargs = [(locs,pargs,intargs,n) for n in N]
        allsols = pool.map(helper_func,allargs)
    statii = np.zeros(NPART,dtype=int)
    ends = np.zeros((NPART,3),dtype=float)
    starts = np.zeros((NPART,3),dtype=float)
    times = np.zeros(NPART)
    for i in range(NPART):
        stat,hist,time = allsols[i]
        statii[i] = stat
        ends[i] = hist[:3]
        starts[i] = locs[i]
        times[i] = time
    np.savez(f'{OUTPUTDIR}/allparts.npz',starts=starts,ends=ends,status=statii,times=times)
    print('all done:')
    print('statuses: ',statii)
    print(f'successes : {count_success(statii)}/'
        +f'{len(statii)}')


def helper_func(args):
    """
    Helper funtion to allow for integration using multiprocessing map, which only takes one
    argument. I could use starmap in the future, but this is just as good.
    
    Parameters
    ----------
    locs : ndarray[tuple]
        ndarray of len nparts, locs[n] = (x0,y0,z0) for particle n
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
    locs,pargs,intargs,n = args
    x0,y0,z0 = locs[n]
    mesh,a,rho_s = pargs
    p = pt.create_particle(mesh,x0,y0,z0,a,rho_s)
    print('starting particle ',n,flush=True)
    savefile = None
    if n%10 == 0:
        savefile = f'{OUTPUTDIR}/history_{n}.npz'
    t0,tf,planet,kw = intargs
    status,end,time = integrate(t0,tf,p,planet,savefile=savefile,
        diffusion=DIFFUSION,partnum=n,**kw)
    print('    finished particle ',n,flush=True)
    del(p)
    return status,end,time

def count_success(allsols):
    ns = 0
    for s in allsols:
        if s==0:
            ns+=1
    return ns

if __name__ == '__main__':
    start = time()
    main()
    end = time()
    t = end-start
    if not os.path.exists('time_to_run.out'):
        with open('time_to_run.out','w+') as f:
            f.write('time [hrs], partsize [cm], tf [yr], solver, maxstep [s], fargodir, outputdir, diffusion\n')
    with open('time_to_run.out','a+') as f:
        f.write(f'{t/3600:.3f}, {A}, {TF/const.YR:.1e}, {SOLVER}, {MAXSTEP}, {FARGODIR}, {OUTPUTDIR}, {DIFFUSION}\n')

