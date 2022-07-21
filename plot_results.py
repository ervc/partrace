import numpy as np
import matplotlib.pyplot as plt

import python as pt
import python.constants as const

FARGODIR = './exampleout/fargo_rescale'

def main(fargodir):
    def npzfilename(n,a):
        return f'/history_n{n}_a{a}_diff.npz'
    fig,axs = plt.subplots(1,2)
    n=50
    alist = [1,10,100]
    for a in alist:
        npzfile = fargodir+npzfilename(n,a)
        sol = np.load(npzfile)
        x,y,z = sol['history'][:,:3].T
        vx,vy,vz = sol['history'][:,3:].T
        times = sol['times']

        r = np.sqrt(x*x + y*y)
        axs[0].plot(times,r,label=f'{a = }cm')
        axs[0].set(xlabel='times [s]',ylabel='r [cm]')

        drdt = np.zeros(len(r)-1)
        tdivs = np.zeros(len(r)-1)
        for i in range(len(r)-1):
            dr = r[i+1] - r[i]
            dt = times[i+1] - times[i]
            drdt[i] = dr/dt
            tdivs[i] = (times[i+1]+times[i])/2
        axs[1].plot(tdivs,drdt)
        # axs2 = axs[1].twinx()
        # axs2.plot(tdivs,drdt/const.AU*const.YR)
        # axs2.set(ylabel='dr/dt [ay/yr]')
        axs[1].set(xlabel='time [s]',ylabel='dr/dt [cm/s]')

    axs[0].legend()
    plt.show()

    mesh = pt.create_mesh(fargodir,n=n,quiet=True)
    planet = pt.create_planet(mesh)
    X = np.linspace(-mesh.yedges.max(),mesh.yedges.max(),200)
    Y = np.linspace(-mesh.yedges.max(),mesh.yedges.max(),200)
    xx,yy = np.meshgrid(X,Y)
    gasrho = mesh.get_rho(xx,yy)
    fig,axs = plt.subplots(1,3,sharey=True)
    for i,a in enumerate(alist):
        npzfile = fargodir+npzfilename(n,a)
        sol = np.load(npzfile)
        x,y,z = sol['history'][:,:3].T
        vx,vy,vz = sol['history'][:,3:].T
        times = sol['times']
        ax=axs[i]
        ax.pcolormesh(xx,yy,np.log10(gasrho))
        ax.plot(x,y,c='k')
        ax.set_title(f'{a = }cm')
        ax.scatter(*planet.pos[:2],marker='x',c='red')
    for ax in axs.flatten():
        ax.set(aspect='equal')
    plt.show()

    fig,axs = plt.subplots(2,1,sharex=True,
        gridspec_kw={'height_ratios' : [3,1],'hspace':0.05})
    sigmean = np.mean(mesh.state['gasdens'][-1],axis=-1)
    H = mesh.get_scaleheight(mesh.ycenters,0,0)
    rhomean = sigmean/np.sqrt(2*const.PI)/H
    drhodr = np.zeros(len(rhomean)-1)
    rdivs = np.zeros(len(rhomean)-1)
    for i in range(len(rhomean)-1):
        drho = rhomean[i+1] - rhomean[i]
        dr = mesh.ycenters[i+1] - mesh.ycenters[i]
        drhodr[i] = drho/dr
        rdivs[i] = (mesh.ycenters[i+1]+mesh.ycenters[i])/2
    from scipy.interpolate import CubicSpline
    from scipy.optimize import root
    cs = CubicSpline(rdivs,drhodr)
    rt = root(cs,7*const.AU)
    if rt.success and not all(drhodr==0):
        print('YES')
        for ax in axs:
            # continue
            ax.axvline(rt.x/const.AU,ls=':',c='grey')
        axs[0].text(rt.x/const.AU,0,'drho/dr = 0',
            rotation='vertical',ha='right',va='bottom',c='grey')
    xp,yp,zp = planet.pos
    for ax in axs:
        ax.axvline(np.sqrt(xp*xp+yp*yp)/const.AU,c='k',ls='--')
    axs[0].text(np.sqrt(xp*xp+yp*yp)/const.AU,0,'planet',
        rotation='vertical',ha='right',va='bottom')

    for i,a in enumerate(alist):
        npzfile = fargodir+npzfilename(n,a)
        sol = np.load(npzfile)
        x,y,z = sol['history'][:,:3].T
        vx,vy,vz = sol['history'][:,3:].T
        times = sol['times']

        r = np.sqrt(x*x + y*y)
        axs[0].plot(r/const.AU,times/const.YR/1e3,label=f'{a = }cm')
    axs[0].legend()
    axs[0].set(ylabel='time [kyr]',yscale='linear')
    
    
    axs[1].plot(mesh.ycenters/const.AU,
        np.log10(rhomean))
    axs[1].set(xlabel='r [au]',ylabel='log avg density')

    plt.show()



if __name__ == '__main__':
    main(FARGODIR)

