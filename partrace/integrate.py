"""
Functions to help with integration of particles in the mesh
"""

import numpy as np
import math

class Solver():
    """Helper class for ODE solution

    Attributes
    ----------
    status : int
        status of solver. 
            |  -2 = other
            |  -1 = fail
            |  0 = success
            |  1 = particle reached inner edge of mesh
            |  2 = particle reached outer edge of mesh
            |  3 = particle was accreted onto planet
    times : array (nout,)
        long array of output times from solver
    history : array (nout,y.shape)
        long array of output variables
    sol : solver
        rk45 solver just in case
    """
    def __init__(self,status,times,history,sol):
        self.status = status
        self.times = times
        self.history = history
        self.sol = sol


def fun(t,Y,particle,planet,diffusion=True):
    """Time derivative at time t of 6d pos vector Y.
    """
    particle.update_position(*Y[:3])
    particle.update_velocity(*Y[3:])
    if diffusion:
        vx,vy,vz = particle.get_veff()
    else:
        vx,vy,vz = particle.vel
    ax,ay,az = particle.total_accel(planet)
    return np.array([vx,vy,vz,ax,ay,az])


def add_diffusion(particle,rk):
    """Helper function to add random diffusion to integration"""
    rng = np.random.default_rng()
    R = rng.random(3) * 2 - 1 # random number between -1 and 1
    if particle.mesh.ndim == 2:
        # if we're in 2d then no random motion in z direction
        R[2] = 0.
    xi = 1./3.
    D = particle.get_particleDiffusivity()
    dt = rk.step_size
    rk.y[:3] += R*(2*D*dt/xi)**(1/2)



def solve_ode(fun,t0,y0,tf,args=None,savefile=False,diffusion=True,partnum=0,**kwargs):
    """ODE solver including random diffusive movement. Similar in
    practice to scipy.integrate.solve_ode, but with added diffusion
    in solver.

    Parameters
    ----------
    fun : function
        time derivative of y, of form fun(t,y,*args) = d/dt y(t)
    t0 : float
        initial time of integration
    y0 : float or (n,) array
        initial conditions
    tf : float
        end of integration. sets direction of integral
    args : tuple
        args to pass to fun(). If only one arg, format like args=(arg,)
    kwargs : optional keyword arguments to pass to RK45 solver. 
        This includes:
            max_step : float
            rtol, atol : float or array_like
            vectorized : bool

    Return
    -------
    Solver object
    """
    import scipy.integrate as scint
    particle,planet = args
    # define new function to allow args to be passed into solver
    def f(t,y):
        if args is not None:
            return fun(t,y,*args,diffusion)
        else:
            return fun(t,y,diffusion)

    # create solver and set up returnables
    kw = kwargs
    maxh=None
    if 'max_step' in kw:
        maxh = kw['max_step']
    rk = scint.Radau(f,t0,y0,tf,**kwargs)
    ys = [y0]
    ts = [t0]
    status = None
    statii = {0 : 'finished',
        -1 : 'failed',
        1 : 'inner edge',
        2 : 'outer edge',
        3 : 'accreted',
        -2 : 'other'}
    print('starting loop')
    nout = 256
    n = 0
    touts = 'ALL'
    #touts = np.logspace(7,np.log10(tf),nout)
    #touts = np.linspace(500*3.15e7,tf,nout)
    #print(f'{touts = }')
    while status is None:
        # try-except block catches issue with nan or inf in step
        # if nan or inf is involved then the sovler has failed
        try:
            message = rk.step()
        except ValueError:
            print('solver failed, setting status to negative 2\n' + 
                 f'Made it to time {rk.t/3.15e7 = }\n' + 
                 f'Final position is {rk.y[:3]}')
            message = 'failed'
            status = -2
        if diffusion:
            add_diffusion(particle,rk)
            if any(math.isnan(i) for i in rk.y):
                print('diffusion made a value NAN')
            if any(math.isinf(i) for i in rk.y):
                print('diffusion made a value INF')
        particle.update_position(*rk.y[:3])
        particle.update_velocity(*rk.y[3:])
        if rk.status == 'finished':
            status = 0
        elif rk.status == 'failed':
            status = -1
        elif rk.status != 'running':
            status = -2
        r = np.linalg.norm(rk.y[:3])
        if r <= np.nanmin(particle.mesh.ycenters):
            status = 1
        elif r >= np.nanmax(particle.mesh.ycenters):
            status = 2
        if planet is not None:
            xp,yp,zp = rk.y[:3]-planet.pos
            rp = np.sqrt(xp*xp + yp*yp + zp*zp)
            if rp<=planet.envelope:
                status = 3
                print(f'ACCRETED! {rk.t/3.15e7 = }')
                print(f'{touts[n-1]/3.15 = }')
                print(f'{touts[n]/3.15 = }\n')
        if touts == 'ALL':
            ys.append(rk.y)
            ts.append(rk.t)
            times = np.array(ts)
            history = np.stack(ys)
            if savefile:
                print(f'{partnum}: t = {rk.t/3.15e7:.2e}, dt = {rk.step_size/3.15e7:.2e}')
        else:
            while (n<nout) and (rk.t>=touts[n]) and (status is None):
                t = touts[n]
                do = rk.dense_output()
                y = do(t)
                ys.append(y)
                ts.append(t)
                #print(f'time {n}/{nout}',rk.y,'\n',flush=True)
                times = np.array(ts)
                history = np.stack(ys)
                if savefile:
                    print(f'{partnum}: {touts[n]/3.15e7 = }',flush=True)
                    np.savez(savefile,times=times,history=history)
                n+=1
    print(f'Solver stopped, status = {statii[status]}')
    if touts != 'ALL':
        # get the last time:
        if status == 0:
            t = touts[-1]
            #print(f'{rk.t/3.15e7 = }')
            #print(f'{touts[-1]/3.15e7 = }')
            do = rk.dense_output()
            y = do(t)
            ys.append(y)
            ts.append(t)
            #print(f'time {n}/{nout}',rk.y,'\n',flush=True)
            times = np.array(ts)
            history = np.stack(ys)
        # get accretion time for accreted particle
        elif status == 3:
            t = rk.t
            do = rk.dense_output()
            y = do(t)
            ys.append(y)
            ts.append(t)
            print(f'time {n}/{nout}',rk.y,'\n',flush=True)
            times = np.array(ts)
            history = np.stack(ys)
    # convert to arrays
    times = np.array(ts)
    ## use np.stack to convert list of arrays to 2d array
    history = np.stack(ys)
    # remove the last output if solver failed
    # to remove repeated output
    if status == -1:
        times = times[:-1]
        history = history[:-1]
    if savefile:
        np.savez(savefile,times=times,history=history)
        print(f'saved to {savefile}')
    
    flag = 0
    if any(math.isnan(x) for x in history[-1]):
        flag += 1
    if any(math.isinf(x) for x in history[-1]):
        flag += 1
    if math.isnan(times[-1]) or math.isinf(times[-1]):
        flag += 1
    if flag > 0:
        print('INF OR NAN IN RET')
        print(ret)
    ret = (status,history[-1],times[-1])
    del(times)
    del(history)
    return ret

def integrate(t0,tf,particle,planet = None,savefile=None,diffusion=True,partnum=0,**kwargs):
    """
    Do the integration for particle in a mesh including a planet
    from time t0 -> tf. Calls the function solve_ode()

    Parameters
    ----------
    t0 : float
        start time of integration
    tf : float
        end time of integration
    particle : Particle
        particle class that will be integrated in time
    planet : Planet
        Planet to consider if there is one in the mesh
        default: None
    savefile : str
        if not None, output integration results as compressed npz file
        with name savefile
    kwargs : optional keyword arguments to pass to solve_ode()
        This includes:
            max_step : float
            rtol, atol : float or array_like
            vectorized : bool

    Returns
    -------
    Solver object

    """
    args = (particle,planet)
    x0,y0,z0 = particle.pos0
    vx0,vy0,vz0 = particle.vel0
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])
    print('integrating')
    ret = solve_ode(fun,t0,Y0,tf,args=args,savefile=savefile,diffusion=diffusion,partnum=partnum,**kwargs)
    return ret

def one_step(particle,planet,**kwargs):
    """Debugging function to take only one step at a time."""
    x0,y0,z0 = particle.pos0
    print('particle X0, V0')
    print(x0,y0,z0)
    vx0,vy0,vz0 = particle.vel0
    print(vx0,vy0,vz0)
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])

    y1 = fun(0,Y0,particle,planet,kwargs['diffusion'])
    print('dY0/dt')
    print(y1)
    
