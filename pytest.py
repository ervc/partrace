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
    zgc = mesh.cartedges['z']
    rge = mesh.edges['y']
    rgc = mesh.edges['y']
    azge = mesh.edges['x']
    azgc = mesh.centers['x']

    s = np.s_[:,:,0] # get az slice
    def avg(arr): return np.mean(arr,axis=-1) # function to avg along azimuth

    fig,axs = plt.subplots(1,2)
    arr = np.log10(mesh.state['gasdens'])

    ax = axs[0]
    im = ax.pcolormesh(rge[s],zge[s],arr[s])
    cb = plt.colorbar(im,ax=ax,label='rho')

    ax = axs[1]
    im = ax.pcolormesh(avg(rge),avg(zge),avg(arr))
    cb = plt.colorbar(im,ax=ax,label='avg rho')

    plt.show()

def plot_generic(mesh):
    nx = 100
    ny = 100
    x = np.linspace(-5,5,nx)*const.AU
    y = np.linspace(-5,5,ny)*const.AU
    z = 0

    xx,yy = np.meshgrid(x,y)

    arr = np.ones_like(xx)
    for j in range(ny):
        for i in range(nx):
            arr[j,i] = mesh.get_state_from_cart('gasdens',x[i],y[j],z)

    fig,ax = plt.subplots()
    ax.pcolormesh(xx,yy,np.log10(arr))
    plt.show()



def main(fargodir):
    mesh = Mesh(fargodir)

    for state in ['gasdens','gasvx','gasvy']:
        mesh.read_state(state,-1)

    # plot_slice(mesh)
    # plot_cart_slice(mesh)
    # plot_side(mesh)
    plot_generic(mesh)


    # x,y,z = 1,1,0
    # i,j,k = mesh.get_cell_from_cart(x,y,z)
    # print(f'x,y,z = {x,y,z}')
    # print('closest cell center:')
    # print(mesh.cartcenters['x'][k,j,i],
    #       mesh.cartcenters['y'][k,j,i],
    #       mesh.cartcenters['z'][k,j,i])

if __name__ == '__main__':
    main(FARGOOUT)

    


        