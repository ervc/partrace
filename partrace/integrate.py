import numpy as np
from . import constants as const


def dYdt(t,Y,part,planet,diffusion=True):
    """Return the time derivative of Y
    Parameters
    ----------
    t : float
        time of integration
    Y : ndarray (6,)
        6d position-velocity vector of particle
    part : Particle
        particle with submesh for integration
    planet : Planet
        planet in submesh
    diffusion : bool
        calculate effective velocity using diffusion

    Returns
    -------
    ndarray (6,)
        dYdt, 6d velocity-acceleration vector
    """
    X = Y[:3]
    V = Y[3:]
    r = np.linalg.norm(X)

    # analytic variables from mesh
    G = float(part.mesh.variables['G'])
    MSTAR = float(part.mesh.variables['MSTAR'])
    omega = np.sqrt(G*MSTAR/np.linalg.norm(X)**3)
    omegaframe = float(part.mesh.variables['OMEGAFRAME'])
    cs = part.mesh.get_soundspeed(*X)

    # get submesh variables we will need
    rho = part.get_rho_at(*X)
    vgas = part.get_gasvel_at(*X)
    gradrho = part.get_gradrho_at(*X)
    diff = part.get_partdiff_at(*X)
    graddiff = part.get_gradpartdiff_at(*X)

    # derive some quantities
    tstop = (part.rho_s*part.a)/(rho*cs)
    St = tstop*omega

    
    # position derivative is the effective velocity
    if not diffusion:
        dXdt = V
    else:
        Vrho = diff/rho * gradrho
        Vdiff = graddiff
        dXdt = V + Vrho + Vdiff

    if St < const.COUPLING:
        # if stopping is low, then just integrate particle using
        # gas velocity.
        # dVdt = 0, velocity will be set later, 
        # don't worry in integration
        dXdt = vgas
        if diffusion:
            dXdt = vgas + Vrho + Vdiff
        dVdt = np.zeros(3)
        return np.array([*dXdt, *dVdt])
    
    # velocity derivative
    # drag accel
    u = V-vgas
    adrag = -u/tstop
    # grav accel
    if planet is not None:
        if planet.mass == 0:
            Xs = np.zeros(3)
        else:
            Xp = planet.pos
            Mp = planet.mass
            Ms = float(part.mesh.variables['MSTAR'])
            # 0 = (XpMp + XsMs)/(Ms+Mp)
            # Xs = -XpMp/Ms
            Xs = -Xp * Mp/Ms
    else:
        Xs = np.zeros(3)
    GM = G*MSTAR
    dstar = np.linalg.norm(X-Xs)
    astar = -GM/dstar**3 * (X-Xs)
    if planet is None:
        aplan = 0
    elif planet.mass == 0:
        aplan = 0
    else:
        Xp = planet.pos
        Mp = planet.mass
        GM = G*Mp
        dplanet = np.linalg.norm(X-Xp)
        aplan = -GM/dplanet**3 * (X-Xp)
    # cent accel
    x,y,z = X
    vx,vy,vz = V
    ax = 2*omegaframe*vy + x*omegaframe**2
    ay = -2*omegaframe*vx + y*omegaframe**2
    az = 0
    acent = np.array([ax,ay,az])
    
    dVdt = adrag + astar + aplan + acent
    
    return np.array([*dXdt, *dVdt])

def rk4(f,t,y,h,*args,**kwargs):
    """Runge-kutta 4 integration"""
    k1 = f(t,y,*args,**kwargs)
    k2 = f(t+h/2,y+h*k1/2,*args,**kwargs)
    k3 = f(t+h/2,y+h*k2/2,*args,**kwargs)
    k4 = f(t+h,y+h*k3,*args,**kwargs)
    return y + 1/6*(k1+2*k2+2*k3+k4)*h

def sigmoid(x,scale=1,center=0):
    """Return sigmoid function at x
    Parameters
    ----------
    x : float
        value to return S(x)
    scale : float
        steepness of sigmoid
    center : float
        center of sigmoid function, S(center) =  0.5

    Returns
    -------
    float
        Value of S(x)
    """
    x = (x-center)*scale
    return 1/(1+np.exp(-x))

def inv_sigmoid(x,*args,**kwargs):
    """ Return the inverse of the sigmoid, helpful for tstop calc, as
    1/S will blow up towards negative infinity
    Parameters
    ----------
    args, kwargs
        passed to sigmoid()

    Returns
    -------
    float
        value of 1/S(x)
    """
    return 1/sigmoid(x,*args,**kwargs)

def get_max_dt(part,maxdt=np.inf,tstop_scale=1,diffusion=True):
    """Get maximum stable dt
    Parameters
    ----------
    part : Particle
        particle that will be integrated
    maxdt : float
        hard max on the stepsize
    tstop_scale : float
        scaling for maximum stepsize relative to stopping time
    diffusion : boolean
        If diffusion is true, set limit on diffusion step as well

    Returns
    -------
    float
        recommended stepsize
    """
    omega = part.mesh.get_Omega(*part.pos)
    torb = 1/20 * 1/omega
    St = part.get_stokes()
    logSt = np.log10(St)

    tstop = tstop_scale*St/omega
    # scale stopping time max dt with stokes number
    # small stokes = coupled to gas = can take larger step sizes
    dtstop = inv_sigmoid(logSt,scale=10,center=np.log10(const.COUPLING))*tstop_scale*St/omega

    dt = min(torb,dtstop,maxdt)
    if diffusion:
        X = part.pos

        graddiff = part.get_gradpartdiff_at(*X)
        dtdiff = 1.e-6*np.linalg.norm(X)/np.linalg.norm(graddiff)

        rhog = part.get_rho_at(*X)
        diff = part.get_partdiff_at(*X)
        gradrhog = part.get_gradrho_at(*X)
        dtrho = 1e-5*rhog/diff*np.linalg.norm(X)/np.linalg.norm(gradrhog)

        dt = min(dt,dtdiff,dtrho)
    return dt

def rkstep_particle(part,planet,t,maxdt=np.inf,tstop_scale=1,diffusion=True):
    """Step the particle one timestep using RK4 integration
    Parameters
    ----------
    part : Particle
        Particle that will be integrated
    planet : Planet
        Planet present in the mesh
    t : float
        current time
    maxdt : float
        maximum stepsize, default = inf
    tstop_scale : float
        scaling for maximum stepsize relative to stopping time
    diffusion : bool
        Include diffusion in step of particle

    Returns
    -------
    float
        time after step, t + dt
    """
    dt = get_max_dt(part,maxdt,tstop_scale,diffusion)
    X = part.pos
    V = part.vel
    Y = np.array([*X,*V])
    args = (part,planet,diffusion)
    Yi = rk4(dYdt,t,Y,dt,*args)
    Xi = Yi[:3]
    St = part.get_stokes()
    # if coupled set velocity to gas velocity, otherwise get velocity
    # from integration.
    if St < const.COUPLING:
        Vi = part.get_gasvel_at(*Xi)
    else:
        Vi = Yi[3:]
    if diffusion:
        ### find X'
        graddiff = part.get_gradpartdiff_at(*X)
        Xprime = X + 0.5*graddiff*dt
        Dxprime = part.get_partdiff_at(*Xprime)
        ### random numbers
        rng = np.random.default_rng()
        R = rng.random(3) * 2 - 1 # random number between -1 and 1
        xi = 1/3
        Xi += R*np.sqrt(2/xi*Dxprime*dt)
    part.update_position(*Xi)
    part.update_velocity(*Vi)
    return t+dt

def integrate(part,planet,tf,savefile,tstop_scale=1,diffusion=True):
    """
    Integrate the particle from time 0 to time tf and save output as npz
    file
    Parameters
    ----------
    part : Particle
        particle to be integrated
    planet : Planet
        planet present in the mesh
    tf : float
        end time of integration
    savefile : str
        name of file to save output to
    tstop_scale : float
        scaling of max step with stopping time, default = 1.0
    diffusion : bool
        Integrate particle using random diffusion, default = True

    Returns
    -------
    str
        final status of integration
    """
    traj = [[*part.pos0,*part.vel0]]
    times = [0]
    Stokes = [part.get_stokes()]

    def save_output():
        np.savez(savefile,history=traj,times=times,Stokes=Stokes)
        print('\nSaved output to: ',savefile)
    
    maxN = int(1e6)
    time = 0
    status = 'running'
    i=0
    dt = 0
    nout = 100
    touts = np.linspace(0,tf,nout)
    iout = 0
    while status=='running':
        if time > touts[iout]:
            print(f'{time:.4e}/{tf:.2e} \t {i}/{maxN} \t {dt/const.YR = :.3e}',flush=True)
            iout += 1
        maxdt = min(tf-time,0.1*const.YR)
        try:
            time = rkstep_particle(
                part,planet,time,maxdt=maxdt,
                tstop_scale=tstop_scale,diffusion=diffusion)
        except ValueError as e:
            if savefile:
                save_output()
            status = 'failed'
            print('Solver failed with error message:')
            print(e)
#             raise ValueError
            return status
        traj.append([*part.pos,*part.vel])
        times.append(time)
        Stokes.append(part.get_stokes())
        dt = time-times[-2]
        
        partr = np.linalg.norm(part.pos)
        planr = np.linalg.norm(part.pos-planet.pos)
        if (partr < part.mesh.ycenters.min() 
                or partr > part.mesh.ycenters.max()):
            status = 'OoB'
        if planr < planet.envelope:
            status = 'accreted'
        if i > maxN:
            status = 'MAXSTEP'
        if time >= tf:
            status = 'finished'
    
        i+=1
    
    if savefile:
        save_output()
    
    return status, traj[-1], times[-1]