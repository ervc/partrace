__author__ = 'Eric Van Clepper'
__version__ = '0.0.5'

# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['xtick.top'] = True
# plt.rcParams['ytick.direction'] = 'in'
# plt.rcParams['ytick.right'] = True

# plt.rcParams['xtick.major.width'] = 1.5
# plt.rcParams['xtick.minor.width'] = 1.
# plt.rcParams['ytick.major.width'] = 1.5
# plt.rcParams['ytick.minor.width'] = 1.

from . import disk
from . import constants
from . import particle
from . import mesh
from . import grid

def create_grid(fargodir,nx,ny,nz=1,domain=None,
                 nout=-1,quiet=False):
	g = grid.Grid(fargodir,nx,ny,nz,domain,nout,quiet)
	return g,g.mesh

def create_mesh(fargodir, states='all', n=-1, quiet=False):
	m = mesh.Mesh(fargodir,states,n,quiet)
	return m
