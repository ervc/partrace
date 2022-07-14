import matplotlib.pyplot as plt
import numpy as np

import python as pt
import python.constants as const

FARGOOUT = './exampleout/fargo3d'

def get_minmaxabs(arr):
    maxabs = np.max(np.abs(arr))
    return -maxabs,maxabs
def get_minmax(arr):
    return np.min(arr), np.max(arr)

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


def main(fargodir):
    mesh = pt.create_mesh(fargodir)

    k = -1

    '''use mesh to find the density, vaz, vr, or vz at any cartesian
    location
    '''

    lenscale,vscale = get_scale(mesh)

    # set up cartesian grid at z=0 (midplane)
    # nx = 250
    # ny = 250
    # nz = 1
    # X = np.linspace(-10,10,nx)*lenscale
    # Y = np.linspace(-10,10,ny)*lenscale
    # Z = np.array([0.])*lenscale
    nx = 1
    ny = 250
    nz = 50
    X = np.array([0.])*lenscale
    Y = np.linspace(-2,2,ny)*lenscale
    Z = np.linspace(-0.2,0.2,nz)*lenscale

    zz,yy,xx = np.meshgrid(Z,Y,X,indexing='ij')

    fig,ax = plt.subplots()
    rho = mesh.get_state_from_cart('gasdens',xx,yy,zz)
    print(rho.shape)
    im = ax.pcolormesh(yy[:,:,0],zz[:,:,0],np.log10(rho[:,:,0]))
    ax.set(title=f'Norbit = {get_norbit(mesh):.0f}\nx=0',
        ylabel='z-coordinate',xlabel='y-coordinate')
    # ax.set_aspect('equal')

    # the outline
    ax.plot(mesh.ycenters,mesh.ycenters*np.cos(mesh.zcenters[0]),ls='-',c='k')
    ax.plot(-mesh.ycenters,mesh.ycenters*np.cos(-mesh.zcenters[0]),ls='-',c='k')
    ax.plot(mesh.ycenters,mesh.ycenters*np.cos(mesh.zcenters[-1]),ls='-',c='k')
    ax.plot(-mesh.ycenters,mesh.ycenters*np.cos(-mesh.zcenters[-1]),ls='-',c='k')


    for k in range(mesh.nz+1):
        ax.plot(mesh.yedges,mesh.yedges*np.cos(mesh.zedges[k]),
            c='k',ls='--',alpha=0.5)
        ax.plot(-mesh.yedges,mesh.yedges*np.cos(-mesh.zedges[k]),
            c='k',ls='--',alpha=0.5)

    ax.set(ylim=(0,Z.max()))

    plt.show()


    i = 0

    ytarget = 1.2
    j,y = find_nearest(Y,ytarget)

    fig,ax = plt.subplots()
    ax.plot(np.log10(rho[:,j,i]),Z,ls='-',marker='o')
    ax.set(title=f'rho(z, y={y:.2f}, x=0)',ylabel='z',xlabel='rho')

    plt.show()

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

if __name__ == '__main__':
    main(FARGOOUT)

    


        