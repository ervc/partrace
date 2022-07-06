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

    # x = mesh.xedges[0] + np.random.random()*(mesh.xedges[-1]-mesh.xedges[0])
    # y = mesh.yedges[0] + np.random.random()*(mesh.yedges[-1]-mesh.yedges[0])

    # if mesh.ndim == 2:
    #   i,j = mesh.get_cell_from_polar(x,y)
    # elif mesh.ndim == 3:
    #   z = np.pi/2
    #   i,j,k = mesh.get_cell_from_polar(x,y,z)

    # print(mesh.variables)

    for state in ['gasdens','gasvx','gasvy']:
        mesh.read_state(state,-1)

    kthet = -1

    fig,axs = plt.subplots(1,3,sharey=True)
    auperyr = const.YR/const.AU
    # imrho = mesh.plot_state('gasdens',ax=axs[0],yunits='au')
    # imvx = mesh.plot_state('gasvx',log=False,ax=axs[1],yunits='au')
    # imvy = mesh.plot_state('gasvy',log=False,ax=axs[2],yunits='au')

    imrho = axs[0].pcolormesh(mesh.xedges,mesh.yedges/const.AU,
        np.log10(mesh.state['gasdens'][kthet]),cmap='inferno',
    )
    label = 'log rho [cm-2]'
    if mesh.ndim == 3:
        label = 'log rho [cm-3]'
    cbrho = plt.colorbar(imrho,ax=axs[0],label=label,location='top')

    arr = mesh.state['gasvx'][kthet]*auperyr
    vmin,vmax = get_minmaxabs(arr)
    imvx = axs[1].pcolormesh(mesh.xedges,mesh.yedges/const.AU,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvx = plt.colorbar(imvx,ax=axs[1],label='vx-v0 [au/yr]',location='top')

    arr = mesh.state['gasvy'][kthet]*auperyr
    vmin,vmax = get_minmaxabs(arr)
    imvy = axs[2].pcolormesh(mesh.xedges,mesh.yedges/const.AU,
        arr, vmin=vmin,vmax=vmax,cmap='coolwarm',
    )
    cbvy = plt.colorbar(imvy,ax=axs[2],label='vy [au/yr]',location='top')

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

    for i in [1,2]:
        axs[i].set_ylabel('')

    # ax.plot(x,y,marker='x',c='k')
    # outline_cell(i,j,mesh,ax)
    # print(x,y)
    # print(mesh.xcenters[i],mesh.ycenters[j])

    # ax.set(
    #   xlim = (mesh.xcenters[i-5],mesh.xcenters[i+5]),
    #   ylim = (mesh.ycenters[j-5],mesh.ycenters[j+5])
    # )

    plt.show()



        