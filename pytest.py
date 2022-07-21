import matplotlib.pyplot as plt
import numpy as np

import python as pt
import python.constants as const

FARGOOUT = './exampleout/fargo_rescale'

def get_minmaxabs(arr):
    maxabs = np.nanmax(np.abs(arr))
    return -maxabs,maxabs
def get_minmax(arr):
    return np.nanmin(arr), np.nanmax(arr)

def get_scale(mesh):
    lenscale = 1
    vscale = 1
    if mesh.variables['UNITS'] == 'CGS':
        lenscale = 1/const.AU
        vscale = const.YR/const.AU
    return lenscale,vscale

def get_norbit(mesh):
    dt_out = float(mesh.variables['DT'])*float(mesh.variables['NINTERM'])
    GM = float(mesh.variables['G'])*float(mesh.variables['MSTAR'])
    R = float(mesh.variables['R0'])
    R3 = R*R*R
    torbit = const.TWOPI * np.sqrt(R3/GM)
    norbit = mesh.n['gasdens']*dt_out/torbit
    return norbit

def plot_slice(mesh):
    kthet = -1

    fig,axs = plt.subplots(1,3,sharey=True)
    

    lenscale,vscale = get_scale(mesh)
    ### plotting
    arr = np.log10(mesh.state['gasdens'][kthet])
    imrho = axs[0].pcolormesh(mesh.edges['x'][kthet],
        mesh.edges['y'][kthet]*lenscale,
        arr,cmap='inferno',
    )
    label = 'log rho [cm-2]'
    if mesh.ndim == 3:
        label = 'log rho [cm-3]'
    cbrho = plt.colorbar(imrho,ax=axs[0],label=label,location='top')
    axs[0].set(ylabel='R [au]',xlabel='azimuth [rad]')

    arr = mesh.state['gasvx'][kthet]*vscale
    vmin,vmax = get_minmaxabs(arr)
    imvx = axs[1].pcolormesh(mesh.edges['x'][kthet],
        mesh.edges['y'][kthet]*lenscale,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvx = plt.colorbar(imvx,ax=axs[1],label='vx-v0 [au/yr]',location='top')
    axs[1].set(xlabel='azimuth [rad]')

    arr = mesh.state['gasvy'][kthet]*vscale
    vmin,vmax = get_minmaxabs(arr)
    imvy = axs[2].pcolormesh(mesh.edges['x'][kthet],
        mesh.edges['y'][kthet]*lenscale,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvy = plt.colorbar(imvy,ax=axs[2],label='vy [au/yr]',location='top')
    axs[2].set(xlabel='azimuth [rad]')

    norbit = get_norbit(mesh)
    title = f'Norbit = {norbit:g}'
    if mesh.ndim == 3:
        title+=f'\ntheta = {mesh.zcenters[kthet]:.3f}'
    fig.suptitle(title)

    plt.show()

def plot_cart_slice(mesh):
    kthet = -1

    lenscale,vscale = get_scale(mesh)
    ### cartesian
    print(mesh.edges['x'].shape)
    print(mesh.cartedges['x'].shape)
    fig,axs = plt.subplots(1,3,sharey=True)

    ax = axs[0]
    arr = np.log10(mesh.state['gasdens'][kthet])
    imrho = ax.pcolormesh(mesh.cartedges['x'][kthet]*lenscale,
        mesh.cartedges['y'][kthet]*lenscale,
        arr,cmap='inferno',
    )
    label = 'log rho [cm-2]'
    if mesh.ndim == 3:
        label = 'log rho [cm-3]'
    cbrho = plt.colorbar(imrho,ax=ax,label=label,location='top')
    ax.set(ylabel='y [au]',xlabel='x [au]')

    ax = axs[1]
    arr = mesh.state['gasvx'][kthet]*vscale
    vmin,vmax = get_minmaxabs(arr)
    imvx = ax.pcolormesh(mesh.cartedges['x'][kthet]*lenscale,
        mesh.cartedges['y'][kthet]*lenscale,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvx = plt.colorbar(imvx,ax=ax,label='vx-v0 [au/yr]',location='top')
    ax.set(xlabel='x [au]')

    ax = axs[2]
    arr = mesh.state['gasvy'][kthet]*vscale
    vmin,vmax = get_minmaxabs(arr)
    imvy = ax.pcolormesh(mesh.cartedges['x'][kthet]*lenscale,
        mesh.cartedges['y'][kthet]*lenscale,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvy = plt.colorbar(imvy,ax=ax,label='vy [au/yr]',location='top')
    ax.set(xlabel='x [au]')

    for ax in axs:
        ax.set(aspect='equal')

    norbit = get_norbit(mesh)
    title = f'Norbit = {norbit:g}'
    if mesh.ndim == 3:
        title+=f'\ntheta = {mesh.zcenters[kthet]:.3f}'
    fig.suptitle(title)

    plt.show()

def plot_side(mesh):
    zge = mesh.cartedges['z']
    zgc = mesh.cartcenters['z']
    rge = mesh.edges['y']
    rgc = mesh.centers['y']
    azge = mesh.edges['x']
    azgc = mesh.centers['x']

    s = np.s_[:,:,0] # get az slice
    def avg(arr): return np.mean(arr,axis=-1) # function to avg along azimuth

    # edges
    fig,axs = plt.subplots(2,2)
    arr = np.log10(mesh.state['gasdens'])

    ax = axs[0,0]
    im = ax.pcolormesh(rge[s],zge[s],arr[s])
    cb = plt.colorbar(im,ax=ax,label='rho')

    ax = axs[1,0]
    im = ax.pcolormesh(avg(rge),avg(zge),avg(arr))
    cb = plt.colorbar(im,ax=ax,label='avg rho')

    # centers
    arr = np.log10(mesh.state['gasdens'])

    ax = axs[0,1]
    im = ax.contourf(rgc[s],zgc[s],arr[s])
    cb = plt.colorbar(im,ax=ax,label='rho')

    ax = axs[1,1]
    im = ax.contourf(avg(rgc),avg(zgc),avg(arr))
    cb = plt.colorbar(im,ax=ax,label='avg rho')

    plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def get_rho(r,z,mesh):
    """
rho[l] = SIGMA0/sqrt(2.0*M_PI)/(R0*ASPECTRATIO)*pow(r/R0,-xi)* \
        pow(sin(Zmed(k)),-beta-xi+1./(h*h));

real xi = SIGMASLOPE+1.+FLARINGINDEX;
real beta = 1.-2*FLARINGINDEX;
real h = ASPECTRATIO*pow(r/R0,FLARINGINDEX);
    """
    def pow(a,b):
        return a**b

    sigma0 = float(mesh.variables['SIGMA0'])
    R0 = float(mesh.variables['R0'])
    ar = float(mesh.variables['ASPECTRATIO'])
    ss = float(mesh.variables['SIGMASLOPE'])
    fi = float(mesh.variables['FLARINGINDEX'])

    xi = ss+1.+fi
    beta = 1.-2.*fi
    h = ar*pow(r/R0,fi)

    rho0 = sigma0/np.sqrt(2*const.PI)/(R0*ar)*pow(r/R0,-xi)*1. # zmed = pi/2
    s = np.sqrt(r**2 + z**2)
    pol = np.arccos(z/s)

    return rho0*pow(np.sin(pol),-beta-xi+1./(h*h))



def main(fargodir):
    mesh = pt.create_mesh(fargodir,n=0)
    minr = mesh.yedges.min()
    maxr = mesh.yedges.max()

    k = -1

    '''use mesh to find the density, vaz, vr, or vz at any cartesian
    location
    '''

    lenscale,vscale = get_scale(mesh)

    # set up cartesian grid at z=0 (midplane)
    nx = 250
    ny = 250
    nz = 1
    X = np.linspace(-maxr,maxr,nx)
    Y = np.linspace(-maxr,maxr,ny)
    Z = np.array([0.])
    s = np.s_[0,:,:]
    # nx = 1
    # ny = 250
    # nz = 20
    # X = np.array([0.])/lenscale
    # Y = np.linspace(-2,2,ny)/lenscale
    # Z = np.linspace(-0.025,0.025,nz)/lenscale
    # s = np.s_[:,:,0]

    zz,yy,xx = np.meshgrid(Z,Y,X,indexing='ij')

    fig,ax = plt.subplots()
    rho = mesh.get_state_from_cart('gasdens',xx,yy,zz)
    im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,np.log10(rho[s]))
    ax.set(title=f'Norbit = {get_norbit(mesh):.0f}\nx=0',
        ylabel='y-coordinate [au]',xlabel='x-coordinate [au]')
    # ax.set_aspect('equal')

    # the outline
    # ax.plot(mesh.ycenters,mesh.ycenters*np.cos(mesh.zcenters[0]),ls='-',c='k')
    # ax.plot(-mesh.ycenters,mesh.ycenters*np.cos(-mesh.zcenters[0]),ls='-',c='k')
    # ax.plot(mesh.ycenters,mesh.ycenters*np.cos(mesh.zcenters[-1]),ls='-',c='k')
    # ax.plot(-mesh.ycenters,mesh.ycenters*np.cos(-mesh.zcenters[-1]),ls='-',c='k')


    # for k in range(mesh.nz+1):
    #     ax.plot(mesh.yedges,mesh.yedges*np.cos(mesh.zedges[k]),
    #         c='k',ls='--',alpha=0.5)
    #     ax.plot(-mesh.yedges,mesh.yedges*np.cos(-mesh.zedges[k]),
    #         c='k',ls='--',alpha=0.5)

    # ax.set(ylim=(Z.min(),Z.max()))

    plt.show()

    fig,axs = plt.subplots(2,2,sharey=True,sharex=True)
    ax = axs[0,0]
    rho = mesh.get_rho(xx,yy,zz)
    im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,np.log10(rho[s]))
    cb = plt.colorbar(im,ax=ax,location='top',label=r'$\rho_g$')

    ax = axs[1,0]
    diff = mesh.get_diffusivity(xx,yy,zz)
    im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,diff[s])
    cb = plt.colorbar(im,ax=ax,location='top',label='D')

    vx,vy,_ = mesh.get_gas_vel(xx,yy,zz)
    ax = axs[0,1]
    vmin,vmax = get_minmaxabs(vx[s])
    print(vmin,vmax)
    im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,vx[s],
        vmin=vmin,vmax=vmax,cmap='coolwarm')
    ct = ax.contour(xx[s]*lenscale,yy[s]*lenscale,vx[s],[0])
    cb = plt.colorbar(im,ax=ax,location='top',label='vx')
    ax = axs[1,1]
    im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,vy[s],
        vmin=vmin,vmax=vmax,cmap='coolwarm')
    ct = ax.contour(xx[s]*lenscale,yy[s]*lenscale,vy[s],[0])
    cb = plt.colorbar(im,ax=ax,location='top',label='vy')

    for ax in axs.flatten():
        ax.set_aspect('equal')

    plt.show()

    fig,ax = plt.subplots()
    X = mesh.ycenters
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    vx,vy,_ = mesh.get_gas_vel(X,Y,Z)
    ax.plot(X,vy)
    from scipy.optimize import root
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(X,vy)
    rt = root(cs,5*const.AU)
    print(rt)


    plt.show()

    # fig,axs = plt.subplots(2,2,sharey=True,sharex=True)
    # drdx,drdy,_ = mesh.get_rho_grad(xx,yy,zz)
    # ax = axs[0,0]
    # im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,drdx[s])
    # cb = plt.colorbar(im,ax=ax,location='top',label=r'd $\rho_g$/d $x$')

    # ax = axs[0,1]
    # im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,drdy[s])
    # cb = plt.colorbar(im,ax=ax,location='top',label=r'd $\rho_g$/d $y$')

    # drdx,drdy,_ = mesh.get_diff_grad(xx,yy,zz)
    # ax = axs[1,0]
    # im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,drdx[s])
    # cb = plt.colorbar(im,ax=ax,location='top',label=r'd $D$/d $x$')

    # ax = axs[1,1]
    # im = ax.pcolormesh(xx[s]*lenscale,yy[s]*lenscale,drdy[s])
    # cb = plt.colorbar(im,ax=ax,location='top',label=r'd $D$/d $y$')


    # for ax in axs.flatten():
    #     ax.set_aspect('equal')

    # plt.show()


    # i = 0

    # ytarget = 1.2
    # j,y = find_nearest(Y,ytarget)
    # H = float(mesh.variables['ASPECTRATIO'])*y
    # zmin = Z.min()
    # zmax = Z.max()

    # fig,ax = plt.subplots()

    # ax.plot(np.log10(rho[:,j,i]),Z/H,ls='-',label='interpolated')

    # Zp = y*np.cos(mesh.zcenters)
    # Z = np.zeros(Zp.size*2)
    # Z[0:Zp.size] = Zp
    # Z[Zp.size:] = -Zp[::-1]
    # zz,yy,xx = np.meshgrid(Z,Y,X,indexing='ij')
    # rho = mesh.get_state_from_cart('gasdens',xx,yy,zz)
    # ax.plot(np.log10(rho[:,j,i]),Z/H,ls='',marker='o',label='model centers')

    # zmin = min(Z.min(),zmin)
    # zmax = max(Z.max(),zmin)

    
    # zall = np.linspace(zmin,zmax,150)
    # ax.plot(np.log10(get_rho(y,zall,mesh)),zall/H,ls='--',c='k',label='analytic')

    # print('zmin:')
    # kmax,polmax = find_nearest(mesh.zcenters,const.PI/2)
    # print(y*np.cos(polmax))

    # ax.axhline( y*np.cos(polmax)/H,c='grey',ls=':',label='lowest cell')
    # ax.axhline(-y*np.cos(polmax)/H,c='grey',ls=':')

    # ax.legend()

    # ax.set(title=f'rho(z, y={y:.2f}, x=0, t={mesh.n["gasdens"]:.0f})',
    #     ylabel='z/H',xlabel='log rho',
    #     ylim=(-0.5,0.5),xlim=(-2.74,-2.66))

    # plt.show()

    '''or mesh also contains the output data from fargo
    in its original coordinates!
    '''
    # fig,ax = plt.subplots()
    # ax.plot(mesh.ycenters,np.mean(mesh.state['gasdens'][k],axis=-1))
    # for i in range(mesh.nx):
    #     ax.plot(mesh.ycenters,mesh.state['gasdens'][k,:,i],alpha=0.5)
    # ax.set(yscale='log')
    # plt.show()


    # the Mesh also has the fargo output variables compiled at runtime
    print(f"{mesh.variables['COORDINATES'] = }")

def submain(fargodir):
    fig,ax = plt.subplots()
    cmap = plt.get_cmap('magma',12)
    for n in reversed(range(10)):
        mesh = pt.create_mesh(fargodir,n=n)
        ax.plot(mesh.ycenters,np.mean(np.log10(mesh.state['gasdens']),
            axis=0),c=cmap(n),alpha=0.3)
    plt.show()

if __name__ == '__main__':
    # submain(FARGOOUT)
    main(FARGOOUT)

    


        