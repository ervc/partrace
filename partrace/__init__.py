__author__ = 'Eric Van Clepper'
__version__ = '0.1.0'

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

from . import constants
from . import particle
from . import mesh
from . import grid
from . import integrate
from . import planet

def create_grid(fargodir,nx,ny,nz=1,domain=None,
                 nout=-1,quiet=False):
    """
    Create a cartesian re-grid from Mesh. Deprecated
    """
    g = grid.Grid(fargodir,nx,ny,nz,domain,nout,quiet)
    return g,g.mesh

def create_mesh(fargodir, states='all', n=-1, quiet=False):
    """
    Create a :class:`Mesh <partrace.mesh.Mesh>` object from 
    FARGO3D output.

    Parameters
    ----------
    fargodir : str
        directory where fargo output data can be found
    states : str, list
        list of states to read in from fargooutput. If states=='all',
        then states of ['gasdens','gasvx','gasvy','gasvz','gasenergy']
        are read in. If states==None, then no states will be read in.
        default: 'all'      
    n : int
        which number output to read. i.e. gas density will be read in
        fargodir/gasdens{n}.dat. If n==-1, then the last output will be
        read. Values are saved in self.n dictionary for each state.
        Ideally, all n values are the same and this is redundant, but if
        n = -1 is used, this can check that all the data is read in from
        the same output.
        default: -1
    quiet : bool
        print confirmation messages to stdout
        default: False
    """
    m = mesh.Mesh(fargodir,states,n,quiet)
    return m

def create_particle(mesh,x0,y0,z0=0,a=0.01,rho_s=2.):
    """
    Create a :class:`Particle <partrace.particle.Particle>` object to 
    integrate in a :class:`Mesh <partrace.mesh.Mesh>`.

    Parameters
    ----------
    mesh : Mesh
        partrace mesh object that contains disk information for the
        particle to move through
    x, y, z : float
        starting location for the particle
    a : float
        particle size in cm
        default: 0.01 cm
    rho_s : float
        particle density in g cm-3
        default: 2.0 g cm-3
    """
    p = particle.Particle(mesh,x0,y0,z0,a,rho_s)
    return p

def create_planet(mesh,planet_no=0,name='jupiter'):
    """
    Create a :class:`Planet <partrace.planet.Planet>` object to embed 
    in a :class:`Mesh <partrace.mesh.Mesh>`.

    Parameters
    ----------
    mesh : Mesh
        Mesh object that planet is embedded in
    planet_no : int
        planet number to read in data from. planet is read in from
        mesh.fargodir/planet{planet_no}.dat. Mesh.n is used to get
        planet location data consistent with mesh.
        default: 0
    name : str
        name of planet for tracking if there is more than one planet
        in mesh
        default: 'planet'
    """
    p = planet.Planet(mesh,planet_no,name)
    return p
