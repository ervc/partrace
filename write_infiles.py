import numpy as np

from partrace import constants as const

fargodir = '../fargo/outputs/alpha3d_moreaz_cont'
r = 7 #au
z = 0 #au
partsize = 1
noutput = 200
nparts = 16

partfile = f'inputs/test_particles_r{r}_z{z}.in'
inputfile = f'inputs/test_params_r{r}_z{z}_a{partsize}_n{noutput}.ini'
outputdir = f'particleout/test_{fargodir.split("/")[-1]}_r{r}_z{z}_a{partsize}_n{noutput}'

params = {}

params['diffusion'] = False
params['fargodir'] = fargodir

params['t0'] = 0
params['tf'] = 1e4*const.YR
params['partsize'] = partsize # cm
params['partdens'] = 2 # g cm-3

params['nout'] = noutput

partlocations = np.empty(nparts,dtype=tuple)
for i in range(nparts):
    phi = i*2*np.pi/nparts
    x0 = r*const.AU*np.cos(phi)
    y0 = r*const.AU*np.sin(phi)
    z0 = z*const.AU
    partlocations[i] = (x0,y0,z0)


params['partfile'] = partfile
params['outputdir'] = outputdir

with open(inputfile,'w') as f:
    f.write('[params]\n')
    for key in params:
        f.write(f'{key} = {params[key]}\n')

with open(partfile,'w') as f:
    for i,loc in enumerate(partlocations):
        x,y,z = loc
        f.write(f'{i}\t{x:.8e}\t{y:.8e}\t{z:.8e}\n')
print('wrote input file: ',inputfile)

