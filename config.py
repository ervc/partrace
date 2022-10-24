import numpy as np

from partrace import constants as const

diffusion = False
fargodir = '../fargo/outputs/alpha3d_moreaz_cont'


t0 = 0
tf = 1e4*const.YR
partsize = 1 # cm
rho_s = 2 # g cm-3

noutput = 200

r = 7 # au
z = 0 # au
nparts = 16
partlocations = np.empty(nparts,dtype=tuple)
for i in range(nparts):
    phi = i*2*np.pi/nparts
    x0 = r*const.AU*np.cos(phi)
    y0 = r*const.AU*np.sin(phi)
    z0 = z*const.AU
    partlocations[i] = (x0,y0,z0)

outputdir = f'particleout/{fargodir.split("/")[-1]}_r{r}_z{z}'

