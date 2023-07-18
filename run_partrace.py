#!/usr/bin/env python

# standard imports
import numpy as np
from numpy.random import default_rng
from multiprocessing import Pool
import os
import subprocess
from time import time

# partrace imports
import partrace as pt
import partrace.constants as const
from partrace.integrate import integrate
import partrace.partraceio as ptio

def main(infile,nproc):

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

    print(f'*** PARTRACE {pt.__version__} ***')
    print('Read infile: ',args.infile)
    print('Fargodir: ',FARGODIR)
    print('partfile: ',params["partfile"])
    print('output dirrectory: ',params["outputdir"])
    print('number of particles: ',NPART)
    print(f'tf = {TF} sec = {TF/const.YR} yr')
    # check number of processors
    print('cpus availables = ',nproc,flush=True)

    # global
    fargodir = FARGODIR
    n = NOUT
    # number of particles
    npart = NPART

    # create mesh
    mesh = pt.create_mesh(fargodir,n=n)
    minr = mesh.yedges.min()
    # maxr = mesh.yedges.max()

    # readin planet
    # planet = pt.create_planet(mesh,0,'Jupiter')

    # set up solver params
    t0 = 0
    tf = TF
    tstop_scale = 1.
    if MAXSTEP:
        # 1/10 of an orbit = 1/10 * TWOPI/OMEGA
        maxdt = 1/10*const.TWOPI/mesh.get_Omega(minr,0,0)
    else:
        maxdt = np.inf

    # constant partical parameters:
    a = A    # size in cm
    rho_s = RHOS  # density in g/cm^3

    print(f'particle size, density = {a} cm, {rho_s} g cm-3')

    locs = LOCS
    pargs = (mesh,a,rho_s)
    kw = {'max_step':maxdt}
    intargs = (tf,OUTPUTDIR,tstop_scale,DIFFUSION)

    if nproc > 1:
        print('Creating multiprocessing pool...', flush=True)
        with Pool(processes=nproc) as pool:
            N = np.arange(npart)
            allargs = [(locs,pargs,intargs,n) for n in N]
            allsols = pool.map(helper_func,allargs)
    else:
        print('looping...')
        allsols = []
        N = np.arange(npart)
        allargs = [(locs,pargs,intargs,n) for n in N]
        for arg in allargs:
            allsols.append(helper_func(arg))

    statii = np.zeros(NPART,dtype='U16')
    ends = np.zeros((NPART,3),dtype=float)
    starts = np.zeros((NPART,3),dtype=float)
    times = np.zeros(NPART)
    for i in range(NPART):
        stat,hist,time = allsols[i]
        statii[i] = stat
        ends[i] = hist[:3]
        starts[i] = locs[i]
        times[i] = time
    np.savez(f'{OUTPUTDIR}/allparts.npz',starts=starts,ends=ends,
            status=statii,times=times)
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
    args : tuple
        tuple made up of arguments used for integration:
        locs : list of all locations
            such that locs[0] = x0,y0,z0, locs[1] = x1,y1,z1, ...
        pargs : particle arguments
            contains mesh, a, rho_s for particle
        intargs : integration arguments
            contains tf, planet, tstop_scale
        n : int
            number of particle to be integrated
    """
    locs,pargs,intargs,n = args
    x0,y0,z0 = locs[n]
    mesh,a,rho_s = pargs
    # mesh = pt.create_mesh(fargodir,n=n)
    planet = pt.create_planet(mesh,0,'Jupiter')
    p = pt.create_particle(mesh,x0,y0,z0,a,rho_s)
    print('starting particle ',n,flush=True)    
    tf,outputdir,tstop_scale,diffusion = intargs
    savefile = None
    if n%10 == 0:
        savefile = f'{outputdir}/history_{n}.npz'
    status,end,time = integrate(p,planet,tf,savefile=savefile,
        tstop_scale=tstop_scale,diffusion=diffusion)
    print('    finished particle ',n,flush=True)
    del(p)
    return status,end,time

def count_success(allsols):
    ns = 0
    for s in allsols:
        if s=='finished':
            ns+=1
    return ns

if __name__ == '__main__':
    start = time()

    # read in arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('-n','--nproc',type=int,default=1)
    args = parser.parse_args()

    main(args.infile,args.nproc)
    end = time()
    t = end-start
    if not os.path.exists('time_to_run.out'):
        with open('time_to_run.out','w+') as f:
            f.write('time [hrs], partsize [cm], tf [yr], solver, maxstep [s], fargodir, outputdir, diffusion\n')
    with open('time_to_run.out','a+') as f:
        f.write(f'{t/3600:.3f}, {A}, {TF/const.YR:.1e}, {SOLVER}, {MAXSTEP}, {FARGODIR}, {OUTPUTDIR}, {DIFFUSION}\n')

