"""
Functions to help with integration of particles in the mesh
"""

import numpy as np

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
    R = np.random.uniform(-1.,1.,size=3)
    if particle.mesh.ndim == 2:
        # if we're in 2d then no random motion in z direction
        R[2] = 0.
    xi = 1./3.
    D = particle.get_particleDiffusivity()
    dt = rk.step_size
    rk.y[:3] += R*(2*D*dt/xi)**(1/2)



def solve_ode(fun,t0,y0,tf,args=None,savefile=False,diffusion=True,**kwargs):
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
    rk = scint.DOP853(f,t0,y0,tf,**kwargs)
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
    nout = 128
    n = 0
    touts = np.logspace(0,np.log10(tf),nout)
    print(f'{touts = }')
    while status is None:
        message = rk.step()
        if diffusion:
            add_diffusion(particle,rk)
        particle.update_position(*rk.y[:3])
        particle.update_velocity(*rk.y[3:])
        if rk.status == 'finished':
            status = 0
        elif rk.status == 'failed':
            status = -1
        elif rk.status != 'running':
            status = -2
        r = np.linalg.norm(rk.y[:3])
        minr = np.nanmin(particle.mesh.ycenters)
        if r <= minr:
            status = 1
        elif r >= np.nanmax(particle.mesh.ycenters):
            status = 2
        if planet is not None:
            xp,yp,zp = rk.y[:3]-planet.pos
            rp = np.sqrt(xp*xp + yp*yp + zp*zp)
            if rp<=planet.envelope:
                status = 3
        if savefile:
            while rk.t >= touts[n]:
                t = touts[n]
                print(f'{rk.t = }')
                print(f'{touts[n] = }')
                do = rk.dense_output()
                y = do(t)
                ys.append(y)
                ts.append(t)
                print(f'time {n}/{nout}',rk.y,'\n',flush=True)
                times = np.array(ts)
                history = np.stack(ys)
                np.savez(savefile,times=times,history=history)
                n+=1
    print(f'Solver stopped, status = {statii[status]}')
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
    ret = (status,history[-1])
    del(times)
    del(history)
    return status

def integrate(t0,tf,particle,planet = None,savefile=None,diffusion=True,**kwargs):
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
    status = solve_ode(fun,t0,Y0,tf,args=args,savefile=savefile,diffusion=diffusion,**kwargs)
    return status

def one_step(particle,planet,**kwargs):
    """Debugging function to take only one step at a time."""
    x0,y0,z0 = particle.pos0
    print('particle X0, V0')
    print(x0,y0,z0)
    vx0,vy0,vz0 = particle.vel0
    print(vx0,vy0,vz0)
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])

    y1 = fun(0,Y0,particle,planet)
    print('dY0/dt')
    print(y1)
    
