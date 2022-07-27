import numpy as np
import matplotlib.pyplot as plt

import python as pt
import python.constants as const

fargodir = './exampleout/fargo_rescale'



mesh = pt.create_mesh(fargodir,quiet=True,n=0)
rs = mesh.ycenters
X = mesh.ycenters
vphi = np.mean(mesh.state['gasvx'][0],axis=-1) #,:,mesh.nx//2] # z=0, all rs, phi = 0

omegaframe = float(mesh.variables['OMEGAFRAME'])
GM = float(mesh.variables['G'])*float(mesh.variables['MSTAR'])
R0 = float(mesh.variables['R0'])
R3 = R0*R0*R0
vk0 = R0*omegaframe

print(f'{omegaframe = }')
print(f'{vk0 = }')

def calcvk(r):
    return np.sqrt(GM/r)

def rotvk(r):
    vp = calcvk(r)
    return vp-r*omegaframe

fig,axs = plt.subplots(2,1,gridspec_kw={'height_ratios':[3,1]})
ax=axs[0]
ax.plot(rs,vphi,c='k',label='from mesh')
ax.plot(rs,rotvk(rs),ls='--',label="rotating vk")
ax.legend()
ax=axs[1]
ax.axhline(0,c='k')
ax.plot(rs,rotvk(rs)-vphi,ls='--')
ax.set(ylabel='difference')
plt.show()