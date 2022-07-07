import python.constants as const
from python.mesh import Mesh
import matplotlib.pyplot as plt
import numpy as np

FARGOOUT = './exampleout/fargo_rescale'

def get_minmaxabs(arr):
    maxabs = np.max(np.abs(arr))
    return -maxabs,maxabs
def get_minmax(arr):
    return np.min(arr), np.max(arr)

def outline_cell(i,j,mesh,ax):
    from matplotlib.patches import Rectangle
    xlo = mesh.xedges[i]
    xhi = mesh.xedges[i+1]
    ylo = mesh.yedges[j]
    yhi = mesh.yedges[j+1]
    width = xhi-xlo
    height = yhi-ylo
    rect = Rectangle((xlo,ylo),width,height,ec='k',fc=None,fill=False)
    ax.add_patch(rect)

if __name__ == '__main__':
    mesh = Mesh(FARGOOUT)

    for state in ['gasdens','gasvx','gasvy']:
        mesh.read_state(state,-1)

    kthet = -1

    fig,axs = plt.subplots(1,3,sharey=True)
    auperyr = const.YR/const.AU
    # imrho = mesh.plot_state('gasdens',ax=axs[0],yunits='au')
    # imvx = mesh.plot_state('gasvx',log=False,ax=axs[1],yunits='au')
    # imvy = mesh.plot_state('gasvy',log=False,ax=axs[2],yunits='au')


    ### plotting
    lenscale = 1
    vscale = 1
    if mesh.variables['UNITS'] == 'CGS':
        lenscale = 1/const.AU
        vscale = auperyr
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

    dt_out = float(mesh.variables['DT'])*float(mesh.variables['NINTERM'])
    GM = float(mesh.variables['G'])*float(mesh.variables['MSTAR'])
    R = float(mesh.variables['R0'])
    R3 = R*R*R
    torbit = const.TWOPI * np.sqrt(R3/GM)
    norbit = mesh.n[state]*dt_out/torbit
    title = f'Norbit = {norbit:g}'
    if mesh.ndim == 3:
        title+=f'\ntheta = {mesh.zcenters[kthet]:.3f}'
    fig.suptitle(title)

    plt.show()


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

    fig.suptitle(title)

    plt.show()


    x,y,z = 5*const.AU,5*const.AU,0
    i,j,k = mesh.get_cell_from_cart(x,y,z)
    print(f'x,y,z = {x,y,z}')
    print('closest cell center:')
    print(mesh.cartcenters['x'][k,j,i],
          mesh.cartcenters['y'][k,j,i],
          mesh.cartcenters['z'][k,j,i])


        