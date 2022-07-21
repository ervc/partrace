import matplotlib.pyplot as plt
import numpy as np

import python as pt
import python.constants as const
from python.integrate import integrate,one_step

FARGOOUT = './exampleout/fargo_rescale'

def get_radius(sol):
    rs = np.sqrt(sol.history[:,0]**2 + sol.history[:,1]**2)
    return rs

def main(fargodir):
    n = -1
    mesh = pt.create_mesh(fargodir,n=n)
    n = mesh.n['gasdens']
    def kepv(r):
        OM = mesh.get_Omega(r,0,0)
        return r*OM
    def rotating_kepv(r):
        v0 = kepv(r)
        return v0-r*float(mesh.variables['OMEGAFRAME'])
    G = float(mesh.variables['G'])
    print(G)
    print(const.G)
    MSTAR = float(mesh.variables['MSTAR'])
    print(MSTAR)
    print(const.MSUN)
    from scipy.optimize import root
    from scipy.interpolate import interp1d
    X = mesh.ycenters
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    vx,vy,_ = mesh.get_gas_vel(X,Y,Z)
    # i1 = interp1d(X,vy)
    # rt = root(i1,5*const.AU)
    minr = mesh.yedges.min()
    maxr = mesh.yedges.max()
    planet = pt.create_planet(mesh,0,'Jupiter')
    plx = planet.pos[0]
    pl_offset = np.array([1.5*const.AU,0,0])
    # x0,y0,z0 = planet.pos + pl_offset
    # x0,y0,z0 = plx*np.cos(const.PI/4),plx*np.sin(const.PI/4),0
    # x0,y0,z0 = pl_offset
    x0,y0,z0 = 10*const.AU,0,0
    print(f'{(x0,y0,z0) = }')
    print('mesh state from cart:')
    print(mesh.get_state_from_cart('gasvx',x0,y0,z0))
    print('r*vphi')
    print(mesh.get_state_from_cart('gasvx',x0,y0,z0))
    
    print('rotating_kepv')
    print(rotating_kepv(x0))

    gskw = {'height_ratios' : [0.75,0.25]}
    fig,axs = plt.subplots(2,1,gridspec_kw=gskw,sharex=True)
    ax=axs[0]
    ax.axhline(0,c='grey',ls=':')
    ax.plot(X/const.AU,vy,label='interp vy',c='k')
    # ax.plot(X/const.AU,X*mesh.get_state_from_cart('gasvx',X,Y,Z),
        # ls='--',label='r*vphi')
    
    ax.plot(X/const.AU,rotating_kepv(X),label='analytic',ls=':')
    ax.set(ylabel='vy(x,y=0)')
    ax.legend()
    ax=axs[1]
    ax.axhline(0,c='k',ls='-')
    # ax.plot(X/const.AU,vy-X*mesh.get_state_from_cart('gasvx',X,Y,Z),ls='--')
    ax.plot(X/const.AU,vy-rotating_kepv(X),ls=':')
    ax.set(xlabel='X [au]',ylabel='difference')
    plt.show()

    a = 10 # cm
    for a in [1,10,100]:
        savefile = fargodir+f'/history_n{n}_a{a}_diff.npz'
        print(savefile)
        rho_s = 2 #g/cm
        particle = pt.create_particle(mesh,x0,y0,z0,a=a,rho_s=rho_s)

        t0 = 0
        tf = 5e3*const.YR
        maxdt = 1/50*const.TWOPI/mesh.get_Omega(minr,0,0)
        firstdt = 1e-5*const.TWOPI/mesh.get_Omega(minr,0,0)
        minvk = np.nanmin(np.abs(mesh.state['gasvx']))
        maxvk = np.nanmax(np.abs(mesh.state['gasvx']))
        atol = np.zeros(6)
        atol[:3] += 1e-3*minr # xtolerance is 1e-6 of min r value
        atol[3:] += 1e-3*maxvk # vtol within 1e-6 of slowest vkep
        rtol = 1e-6
        print(f'{atol = }')
        print(f'{np.mean(np.abs(mesh.state["gasvx"])) = }')
        sol = integrate(t0,tf,particle,planet,savefile=savefile,
            max_step=maxdt,
            atol = atol,rtol=rtol,
            # first_step=firstdt,
            )
    # one_step(particle,planet)

    print('Omega calc')
    omcalc = np.sqrt(G*MSTAR/x0/x0/x0)
    print(omcalc)

    print('omega frame')
    print(mesh.variables['OMEGAFRAME'])

    print('ax calc')
    axcalc = G*MSTAR/x0/x0
    print(axcalc)

    print('vkep calc')
    vkep = np.sqrt(G*MSTAR/x0)
    print(vkep)

    fig,axs = plt.subplots(2,3)
    labels = ['x','y','z','vx','vy','vz']
    for i in range(3):
        ax = axs[0,i]
        ax.plot(sol.times,sol.history[:,i])
        ax.set_title(labels[i])
        ax = axs[1,i]
        ax.plot(sol.times,sol.history[:,i+3])
        ax.set_title(labels[i+3])

    plt.show()
    fig,axs = plt.subplots(1,2)
    x,y,z = sol.history[:,:3].T
    r = np.sqrt(x*x+y*y)
    print(f'{x.shape = }')
    vx,vy,vz = mesh.get_gas_vel(x,y,z)
    ax=axs[0]
    ax.plot(sol.times,sol.history[:,3],label='particle')
    ax.plot(sol.times,vx,label='gas')
    # ax.plot(sol.times,np.abs(vx)/r,label='gas/R0')
    ax.set(title='vx')
    ax=axs[1]
    ax.plot(sol.times,sol.history[:,4])
    ax.plot(sol.times,vy)
    # ax.plot(sol.times,np.abs(vy)/r,label='gas/R0')
    ax.set(title='vy')
    # for ax in axs:
        # ax.set(yscale='log')
    plt.show()

    fig,ax = plt.subplots()
    vpx,vpy,vpz = sol.history[:,3:].T
    vptot = np.sqrt(vpx*vpx + vpy*vpy)
    ax.plot(r,vptot,label='particle speed')
    R = mesh.ycenters
    omegaframe = float(mesh.variables['OMEGAFRAME'])
    ax.plot(R,np.sqrt(G*MSTAR/R)-R*omegaframe,ls='--',label='keplerian')
    ax.legend()
    plt.show()


    fig,ax = plt.subplots()

    ax.plot(sol.times,get_radius(sol))

    plt.show()

    x = np.linspace(-maxr,maxr,200)
    y = np.linspace(-maxr,maxr,200)
    xx,yy = np.meshgrid(x,y)
    vx,vy,_ = mesh.get_gas_vel(xx,yy)
    fig,axs = plt.subplots(1,2)
    ax = axs[0]
    ax.pcolormesh(xx,yy,np.log10(mesh.get_rho(xx,yy)))
    ax.plot(sol.history[:,0],sol.history[:,1],marker=',',c='k')
    # ax.set(aspect='equal') #,xlim=(-maxr,maxr),ylim=(-maxr,maxr))
    ax=axs[1]
    vtot = np.sqrt(vx*vx+vy*vy)
    vmin = -np.nanmax(np.abs(vtot))
    vmax = -vmin
    ax.pcolormesh(xx,yy,vtot,vmin=0,vmax=vmax,cmap='inferno')
    ax.plot(sol.history[:,0],sol.history[:,1],marker=',',c='white')
    for ax in axs:
        ax.plot(*planet.pos[:2],ls='',marker='x',color='red')
        ax.set(aspect='equal')

    plt.show()


if __name__ == '__main__':
    main(FARGOOUT)


"""
c = TWOPI*r    cm
v = r*omega    cm/s

time = c/v = TWOPI/omega s

"""
