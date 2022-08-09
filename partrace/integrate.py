import numpy as np

class Solver():
    """Helper class for ODE solution"""
    def __init__(self,status,times,history,sol):
        self.status = status
        self.times = times
        self.history = history
        self.sol = sol


def fun(t,Y,particle,planet):
    """time derivative of 6d pos vector"""
    particle.update_position(*Y[:3])
    particle.update_velocity(*Y[3:])
    vx,vy,vz = particle.get_veff()
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



def solve_ode(fun,t0,y0,tf,args=None,savefile=False,**kwargs):
    """ODE solver including random diffusive movement.

    INPUTS
    ------
    fun : function
        time derivative of y, of form fun(t,y,*args) = d/dt y(t)
    t0 : float
        initial time of integration
    y0 : float or (n,) array
        initial conditions
    tf : float
        end of integration. sets direction of integral
    OPTIONAL
    args : tuple
        args to pass to fun(). If only one arg, format like args=(arg,)
    kwargs : optional keyword arguments to pass to RK45 solver. 
        This includes:
            max_step : float
            rtol, atol : float or array_like
            vectorized : bool

    RETURNS
    -------
    status : int
        status of solver. 
            0 = success
            -1 = fail
            1 = other
    times : array (nout,)
        long array of output times from solver
    history : array (nout,y.shape)
        long array of output variables
    sol : solver
        rk45 solver just in case
    """
    from scipy.integrate import RK45
    particle,planet = args
    # define new function to allow args to be passed into solver
    def f(t,y):
        if args is not None:
            return fun(t,y,*args)
        else:
            return fun(t,y)

    # create solver and set up returnables
    kw = kwargs
    maxh=None
    if 'max_step' in kw:
        maxh = kw['max_step']
    rk = RK45(f,t0,y0,tf,**kwargs)
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
    n = 0
    tout = 0.01
    while status is None:
        message = rk.step()
        #add_diffusion(particle,rk)
        particle.update_position(*rk.y[:3])
        particle.update_velocity(*rk.y[3:])
        ys.append(rk.y)
        ts.append(rk.t)
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
        xp,yp,zp = rk.y[:3]-planet.pos
        rp = np.sqrt(xp*xp + yp*yp + zp*zp)
        if rp<=planet.envelope:
            status = 3
        if rk.t/tf > n*tout:
            n+=1
            print(rk.t/tf,r/minr,rk.step_size/maxh,rk.y)
    print(f'Solver stopped, status = {statii[status]}')
    # convert to arrays
    times = np.array(ts)
    ## use np.stack to convert list of arrays to 2d array
    history = np.stack(ys)
    if savefile:
        np.savez(savefile,times=times,history=history)
    return Solver(status,times,history,rk)

def integrate(t0,tf,particle,planet,savefile=False,**kwargs):
    """
    Do the integration for particle in a mesh including a planet
    from time t0 -> tf. Calls the function solve_ode()

    INPUTS
    ------
    t0 : float
        start time of integration
    tf : float
        end time of integration
    kwargs : optional keyword arguments to pass to solve_ode()
        This includes:
            max_step : float
            rtol, atol : float or array_like
            vectorized : bool

    """
    args = (particle,planet)
    x0,y0,z0 = particle.pos0
    vx0,vy0,vz0 = particle.vel0
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])
    print('integrating')
    sol = solve_ode(fun,t0,Y0,tf,args=args,savefile=savefile,**kwargs)
    return sol

'''
class Integrator():
    """docstring for Integrator"""
    def __init__(self,particle,planet):
        self.particle = particle
        self.planet = planet

'''

def one_step(particle,planet,**kwargs):
    x0,y0,z0 = particle.pos0
    print('particle X0, V0')
    print(x0,y0,z0)
    vx0,vy0,vz0 = particle.vel0
    print(vx0,vy0,vz0)
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])

    y1 = fun(0,Y0,particle,planet)
    print('dY0/dt')
    print(y1)
    
