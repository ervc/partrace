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

def unstack(a, axis=-1):
    """Helper function to unpack arrays. Gradient arrays have shape
    (nz,ny,nx,3) for example. This function will return 3 arrays each
    with shape (nz,ny,nx).
    """
    return np.moveaxis(a, axis, 0)

def make_megagrid(mesh,a,rho_s):
    """Create one grid with rho_g, v_g, cs, grad.rho_g, D, grad.D
    Shape will be (nz,ny,nx,12)
    """
    # collect the grids
    rhogas = mesh.state['gasdens']
    velgasx, velgasy, velgasz = mesh.get_cartvel_grid()
    cs = mesh.state['gasenergy']
    gradrhox, gradrhoy, gradrhoz = unstack(mesh.state['gradrho'],axis=-1)
    D = mesh.create_partdiff_grid(a,rho_s)
    gradDx, gradDy, gradDz = unstack(mesh.create_diff_grad(),axis=-1)

    arrs = (rhogas,   velgasx,  velgasy, velgasz, cs,     gradrhox,
            gradrhoy, gradrhoz, D,       gradDx,  gradDy, gradDz)

    return np.stack(arrs,axis=-1)

def unpack_megagrid(megagrid):
    arrs = unstack(megagrid,axis=-1)
    return arrs

def create_mega_interpolator(mesh,a,rho_s):
    """Create an interpolator to find the 12 needed variables at any 
    any point in the MegaGrid(TM). Follows from mesh.create_interpolator,
    see that function for more in depth methodology.
    """
    from scipy.interpolate import RegularGridInterpolator
    from .constants import PI,TWOPI

    MegaGrid = make_megagrid(mesh,a,rho_s)
    megashape = MegaGrid.shape
    nargs = megashape[-1]

    # r stays the same
    Y = mesh.ycenters

    # complete the disk across midplane
    lenz = len(mesh.zcenters)
    Z = np.zeros(2*lenz)
    Z[:lenz] = mesh.zcenters
    Z[lenz:] = PI - mesh.zcenters[::-1]

    # periodic in phi direction
    # add on one extra ghost cell for interpolation
    # to allow for interpolation in full TWOPI
    lenx = len(mesh.xcenters)
    X = np.zeros(lenx+2)
    X[1:-1] = mesh.xcenters
    X[0]  = mesh.xcenters[-1] - TWOPI
    X[-1] = mesh.xcenters[0]  + TWOPI

    # copy values into the array
    arr = np.zeros((len(Z),len(Y),len(X),nargs))
    arr[0:lenz,:,1:-1] = MegaGrid
    arr[0:lenz,:,0]  = MegaGrid[:,:,-1]
    arr[0:lenz,:,-1] = MegaGrid[:,:,0]

    zvecargs = [3,7,11] # args of velgasz, gradrhoz, gradDz
    args = [0,1,2,4,5,6,8,9,10] # args of things not flipped
    # copy scalars over the midplane
    for i in args:
        arr[-1:lenz-1:-1,:,:,i] = arr[0:lenz,:,:,i]
    # flip and reverse z vectors
    for i in zvecargs:
        arr[-1:lenz-1:-1,:,:,i] = -arr[0:lenz,:,:,i]

    # create the interpolator
    interp = RegularGridInterpolator(
        (Z,Y,X),arr,method='linear',bounds_error=False)

    ### optional include regrid here if things take too long ###

    return interp

def use_interp(interp,x,y,z):
    """Take in cartesian values and interpolate in spherical coords"""
    phi = np.arctan2(y,x)
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    return interp(np.stack([theta,r,phi],axis=-1))


def new_solve_ode(t0,y0,tf,particle,planet,MegaInterp,
        savefile=False,diffusion=True,partnum=0,**kwargs):
    """New ODE solver include random motion using MegaGrid
    interpolation(TM). Forward solve integration using adaptive timestep
    and include diffusion.

    Parameters
    ----------
    t0 : float
        initial time of integration
    y0 : float or (n,) array
        initial conditions
    tf : float
        end of integration. sets direction of integral
    particle : Particle
        particle to be integrated through the mesh
    planet : Planet
        Planet in the mesh (if it exists)
    MegaInterp : interpolator
        Mega interpolator from create_mega_interpolator. Required for
        integration
    savefile : str or bool
        If str, save output to savefile. Otherwise if False do not save
        output to file, but do return final position
    diffusion : bool
        Include diffusion in integration?
    partnum : int
        particle number for keeping track during integration
    kwargs : optional keyword arguments for solving. 
        This includes:
            max_step : float

    Return
    -------
    Solver object
    """
    from .constants import YR,TWOPI
    mesh = particle.mesh

    # setup
    maxdt = np.inf
    if 'max_step' in kwargs:
        maxdt = kwargs['max_step']

    ys = [y0]
    Yim = y0
    ts = [t0]
    time = t0
    status = None
    statii = {0 : 'finished',
        -1 : 'failed',
        1 : 'inner edge',
        2 : 'outer edge',
        3 : 'accreted',
        -2 : 'other'
    }

    # constant particle parameters
    a = particle.a
    rho_s = particle.rho_s

    # constant planet parameters
    Mpl = planet.mass
    Xpl = planet.pos

    # constant star parameters
    Mstar = float(mesh.variables['MSTAR'])
    Xstar = -Xpl * Mpl/Mstar # CoM at origin

    # various constants
    G = float(mesh.variables['G'])
    omegaframe = float(mesh.variables['OMEGAFRAME'])
    Omegaf = np.array([0,0,omegaframe])

    # dYdt
    def f(t,Y):
        """Return the derivative of dYdt and the maximum d/dt"""
        X = Y[:3] # current position of particle
        V = Y[3:] # current velocity of particle
        r = np.linalg.norm(X)

        omega = np.sqrt(G*Mstar/np.linalg.norm(X)**3)

        # megagrid reference
        # arrs = (rhogas,   velgasx,  velgasy, velgasz, cs,     gradrhox,
        #         gradrhoy, gradrhoz, D,       gradDx,  gradDy, gradDz)
        rho,vgx,vgy,vgz,cs,gradrhox,gradrhoy,gradrhoz, \
            D,gradDx,gradDy,gradDz = use_interp(MegaInterp,*X)[0]

        tstop = (rho_s*a)/(rho*cs)
        St = tstop*omega

        Vg = np.array([vgx,vgy,vgz])
        Gradrho = np.array([gradrhox,gradrhoy,gradrhoz])
        GradD = np.array([gradDx,gradDy,gradDz])

        ### get veff
        if diffusion:
            Vrho = D/rho*Gradrho
            Vdiff = GradD
            Veff = V + Vrho + Vdiff
        else:
            Veff = V

        ### get accelerations
        # Epstein gas drag
        U = V - Vg
        Adrag = -(rho*cs)/(rho_s*a)*U
        # grav acceleration
        Agravstar = -G*Mstar/np.linalg.norm(X-Xstar)**3 * (X-Xstar)
        Agravpl = -G*Mpl/np.linalg.norm(X-Xpl)**3 * (X-Xpl)
        Agrav = Agravstar + Agravpl
        # cent acceleration
        Acent = -2*np.cross(Omegaf,V) - np.cross(Omegaf,np.cross(Omegaf,X))
        # total
        Atot = Adrag + Agrav + Acent

        # 6D vector = [veffx,veffy,veffx,atotx,atoty,atotz]
        dYdt = np.array([*Veff,*Atot])

        ### find what dt should be
        dtlims = [
            1/50 * TWOPI/omega, # orbital limit
            1/2 * tstop, # limit on stopping time
        ]
        if diffusion:
            dtlims += [
                TWOPI*rim/np.linalg.norm(Vdiff) * 1.e-6, # limit on grad.D
                TWOPI*rim/np.linalg.norm(Vrho)  * 1.e-5, # limit on grad.rho
            ]

        dt = min(dtlims)

        return dYdt,dt

    def rkstep(f,t,y,h):
        """Runge Kutta 4 step"""
        k1,_ = f(t,y)
        k2,_ = f(t+h/2,y+h*k1/2)
        k3,_ = f(t+h/2,y+h*k2/2)
        k4,_ = f(t+h,y+h*k3)
        return y + 1/6*(k1+2*k2+2*k3+k4)*h


    def ramp_in(n):
        n_ramp = 128
        # slowly ramp in dt at the start
        if n >= n_ramp:
            return np.inf
        return np.logspace(-7,0,n_ramp)[n]*YR

    ##### MAIN LOOP #####
    print('starting loop')
    checkpoints = np.linspace(t0,tf,101)
    ncheck = 0
    MAXN = np.inf
    nloop = 0
    dt = 0
    while status is None:
        # print(f'{time/YR:.4e}/{tf/YR:.4e} : last_step = {dt/YR:.2e}\r',
                # end='',flush=True)
        if time > checkpoints[ncheck]:
            print(f'{partnum = }, {time/YR = :.3e}, {time/tf*100:.2f}% done',flush=True)
            ncheck += 1

        Xim = Yim[:3] # current position of particle
        Vim = Yim[3:] # current velocity of particle
        rim = np.linalg.norm(Xim)
        omega = np.sqrt(G*Mstar/np.linalg.norm(Xim)**3)

        # megagrid reference
        # arrs = (rhogas,   velgasx,  velgasy, velgasz, cs,     gradrhox,
        #         gradrhoy, gradrhoz, D,       gradDx,  gradDy, gradDz)
        rho,vgx,vgy,vgz,cs,gradrhox,gradrhoy,gradrhoz, \
            D,gradDx,gradDy,gradDz = use_interp(MegaInterp,*Xim)[0]

        # need gradD for xprime calc
        GradD = np.array([gradDx,gradDy,gradDz])

        # get time derivative and max timestep possible
        dYdt,dt = f(time,Yim)
        # further limit dt with constant time limits (if need be)
        dt = min([
            dt,
            ramp_in(nloop), # slowly ramp up dt at start
            tf-time, # check if at the end
            maxdt, # hard limit
            ])

        ### integrate
        Yi = rkstep(f,time,Yim,dt)
        if diffusion:
            ### find what X' is
            Xprime = Xim + 0.5*GradD*dt
            Dxprime = mesh.get_partdiff_at(*Xprime)[0]
            ### random numbers
            rng = np.random.default_rng()
            R = np.zeros(6)
            R[:3] = rng.random(3) * 2 - 1 # random number between -1 and 1
            xi = 1/3
            ### add random diffusion
            Yi += R*np.sqrt(2/xi*Dxprime*dt)
        time += dt

        ys.append(Yi)
        ts.append(time)

        # check status of integration
        if time >= tf:
            status = 0 # finished
        ri = np.linalg.norm(Yi[:3])
        if ri <= np.nanmin(particle.mesh.ycenters):
            status = 1 # inner edge
        elif ri >= np.nanmax(particle.mesh.ycenters):
            status = 2 # outer edge
        if planet is not None:
            xp,yp,zp = Yi[:3]-Xpl
            rp = np.sqrt(xp*xp + yp*yp + zp*zp)
            if rp<=planet.envelope:
                status = 3 # accreted
                # print(f'ACCRETED! {time/YR = }')

        # make sure the integrator is not failing
        if status is None and any([math.isnan(i) or math.isinf(i) for i in Yi]):
            status = -1 # failed
        if nloop > MAXN:
            status = -2 # other

        # setup for the next timestep
        Yim = Yi

        nloop+=1
    print('\n')

    # convert outputs to arrays
    times = np.array(ts)
    history = np.stack(ys)

    # save output and return
    ret = (status,history[-1],times[-1])
    print('done! status = ',status, statii[status])
    if savefile:
        np.savez(savefile,times=times,history=history)
        print(f'saved to {savefile}')
    del(times)
    del(history)
    return ret

def integrate(t0,tf,particle,planet=None,MegaInterp=None,savefile=None,
        diffusion=True,partnum=0,**kwargs):
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
    MegaInterp : Interpolator
        MegaInterpolator to make interpolations go much faster. See
        create_mega_interpolator() for details
    savefile : str
        if not None, output integration results as compressed npz file
        with name savefile
    diffusion : bool
        Include diffusion in integration?
    partnum : int
        particle number for keeping track during integration
    kwargs : optional keyword arguments to pass to solve_ode()
        This includes:
            max_step : float

    Returns
    -------
    tuple
        contains the ending status, final position, and
        final time of integration

    """
    x0,y0,z0 = particle.pos0
    vx0,vy0,vz0 = particle.vel0
    Y0 = np.array([x0,y0,z0,vx0,vy0,vz0])
    print('integrating')
    ret = new_solve_ode(t0,Y0,tf,particle,planet,MegaInterp,
            savefile=savefile,diffusion=diffusion,partnum=partnum,**kwargs)
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
    
def old_solve_ode(fun,t0,y0,tf,args=None,savefile=False,diffusion=True,partnum=0,**kwargs):
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
        args to pass to fun(). If only one arg, format like args=(arg,).
        This should be (particle,planet) for particle default fun()
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
    nloop = 0
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
        if touts == 'ALL':
            ys.append(rk.y)
            ts.append(rk.t)
            times = np.array(ts)
            history = np.stack(ys)
            if savefile:
                flush = False
                if nloop%100 == 0:
                    flush = True
                print(f'{partnum}: t = {rk.t/3.15e7:.2e}, '+
                      f'dt = {rk.step_size/3.15e7:.2e}',
                      flush=flush)
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