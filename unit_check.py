import numpy as np
import matplotlib.pyplot as plt

import python as pt
import python.constants as const

fargodir = './exampleout/fargo_rescale'



mesh = pt.create_mesh(fargodir,quiet=True,n=0)
rs = mesh.ycenters
X = mesh.ycenters
vphi = mesh.state['gasvx'][0,:,mesh.nz//2] # z=0, all rs, phi = 0

omegaframe = float(mesh.variables['OMEGAFRAME'])
GM = float(mesh.variables['G'])*float(mesh.variables['MSTAR'])
R0 = float(mesh.variables['R0'])
R3 = R0*R0*R0
vk0 = R0*omegaframe

print(f'{omegaframe = }')
print(f'{vk0 = }')

def calcvphi(r):
    return np.sqrt(GM/r)

def calcvy(x):
    return x*calcvphi(x)

def rotvy(x):
    vy = calcvy(x)
    return vy - x*omegaframe*x

def rotvphi(r):
    vp = calcvphi(r)
    return vp-r*omegaframe

fig,axs = plt.subplots(2,1,gridspec_kw={'height_ratios':[3,1]})
ax=axs[0]
ax.plot(rs,rs*vphi,c='k',label='from mesh')
ax.plot(rs,rs*rotvphi(rs),ls='--',label="r*vphi'")
ax.plot(rs,rotvy(rs),ls=':',label='rotating vy')
# ax.plot(rs,rs*calcvphi(rs)-rs*rs*omegaframe,ls='-.')
ax.legend()
ax=axs[1]
ax.plot(rs,rs*vphi-rs*rotvphi(rs),ls='--')
ax.plot(rs,rs*vphi-rotvy(rs),ls=':')
plt.show()