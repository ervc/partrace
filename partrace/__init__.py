__author__ = 'Eric Van Clepper'
__version__ = '1.0.0'

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
from . import partraceio

class Model():
    def __init__(self,alpha,mplan):
        self.alpha = alpha
        self.mplan = mplan
        self.mesh = self.create_mesh()
        

    def get_fargodir(self):
        return f'../fargo/outputs/alpha{self.alpha}_mplan{self.mplan}'

    def create_mesh(self,nout=None,quiet=True):
        if nout is None:
            nout = 'avg' if self.mplan in [200,300] else 50
        mesh = create_mesh(self.get_fargodir(),n=nout,quiet=quiet)
        return mesh

    def get_spheregrid(self):
        P = self.mesh.xcenters
        R = self.mesh.ycenters
        T = self.mesh.zcenters
        tt,rr,pp = np.meshgrid(T,R,P,indexing='ij')
        return tt,rr,pp
    
    def get_cartgrid(self):
        tt,rr,pp = self.get_spheregrid()
        xx,yy,zz = self.mesh._sphere2cart(pp,rr,tt)
        return xx,yy,zz

    def get_cylgrid(self):
        tt,rr,pp = self.get_spheregrid()
        xx,yy,zz = self.get_cartgrid()
        rr = np.sqrt(xx*xx + yy*yy)
        return pp,rr,zz

    def get_rhogrid(self):
        return self.mesh.read_state('gasdens')

    def get_spherevelgrid(self):
        vphi = self.mesh.read_state('gasvx')
        vr = self.mesh.read_state('gasvy')
        vtheta = self.mesh.read_state('gasvz')
        return vphi,vr,vtheta

    def get_cartvelgrid(self):
        tt,rr,pp = self.get_spheregrid()
        vphi,vr,vtheta = self.get_spherevelgrid()
        vx,vy,vz = self.mesh._vel_sphere2cart(pp,rr,tt,vphi,vr,vtheta)
        return vx,vy,vz

    def get_cylvelgrid(self):
        tt,rr,pp = self.get_spheregrid()
        vphi,vr,vtheta = self.get_spherevelgrid()
        zz = rr*np.cos(tt)
        vz = vr*np.cos(tt) - vtheta*np.sin(tt)
        rr = rr*np.sin(tt)
        vr = vr*np.sin(tt) + vtheta*np.cos(tt)
        return vphi,vr,vz

    def get_Omegagrid(self):
        G = float(self.mesh.variables['G'])
        MSTAR = float(self.mesh.variables['MSTAR'])
        GM = G*MSTAR
        tt,rr,pp = self.get_spheregrid()
        return np.sqrt(GM/rr/rr/rr)

    def get_surfacedensity(self):
        rho = self.mesh.read_state('gasdens')
        rhomid = rho[-1]
        R   = self.mesh.ycenters
        PHI = self.mesh.xcenters
        pp,rr = np.meshgrid(PHI,R)
        aspect = float(self.mesh.variables['ASPECTRATIO'])
        flaring = float(self.mesh.variables['FLARINGINDEX'])
        R0 = float(self.mesh.variables['R0'])
        H = rr*aspect*(rr/R0)**flaring
        return rhomid*np.sqrt(2*np.pi)*H

    def get_real_surfacedensity(self):
        rho = self.mesh.read_state('gasdens')
        THETA = self.mesh.zcenters
        R = self.mesh.ycenters
        PHI = self.mesh.xcenters
        THETAEDGE = self.mesh.zedges
        tte,rr,pp = np.meshgrid(THETAEDGE,R,PHI,indexing='ij')
        zze = rr*np.cos(tte)
        dz = zze[:-1]-zze[1:] # take differences of z edges
        Sigma = 2*np.sum(rho*dz,axis=0) # sum along the first (theta) axis. Theta approx z near midplane
                                        # multiply by 2 because this is a half disk
        return Sigma

    def find_gapedge(self,*args,**kwargs):
        """
        Find where Sigma/Sigma0 = 0.5
        There must be a way to find model0 where mass = 0
        args and kwargs pass to scipy.optimize.root_scalar() to find 
        root of func. Useful kwargs might be:
            x0: float
                initial guess
            x1: float
                second guess
            bracket: tuple of 2 floats
                interval for bracketing a guess, must have different
                signs at each bracket value
        Returns:
            scipy.optimize.RootResults
        """
        from scipy.interpolate import CubicSpline
        from scipy.optimize import root_scalar

        try:
            model0 = Model(self.alpha,0)
        except:
            raise Exception('Cannot find mplan=0 model to compare with')
        sig0 = np.average(model0.get_real_surfacedensity(),axis=-1)
        R = self.mesh.ycenters
        sig = np.average(self.get_real_surfacedensity(),axis=-1)
        # interp to find root where sig/sig0=0.5
        sigfunc = CubicSpline(R,sig/sig0-0.5)
        return root_scalar(sigfunc,*args,**kwargs)

    def find_dv0(self,*args,**kwargs):
        """
        Find where velocity is Keplerian
        args and kwargs pass to scipy.optimize.root_scalar() to find 
        root of func. Useful kwargs might be:
            x0: float
                initial guess
            x1: float
                second guess
            bracket: tuple of 2 floats
                interval for bracketing a guess, must have different
                signs at each bracket value
        Returns:
            scipy.optimize.RootResults
        """
        from scipy.interpolate import CubicSpline
        from scipy.optimize import root_scalar

        R = self.mesh.ycenters
        # midplane average gas vphi in rotating frame
        vphirot = np.average(self.mesh.read_state('gasvx')[-1],axis=-1)
        omegaframe = float(self.mesh.variables['OMEGAFRAME'])
        vphi = vphirot + R*omegaframe
        GM = float(self.mesh.variables['G'])*float(self.mesh.variables['MSTAR'])
        vkep = np.sqrt(GM/R)
        dv = (vphi - vkep)/vkep
        # interp to find root where dv=0
        dvfunc = CubicSpline(R,dv)
        return root_scalar(dvfunc,*args,**kwargs)

def create_grid(fargodir,nx,ny,nz=1,domain=None,
                 nout=-1,quiet=False):
    """
    Create a cartesian re-grid from Mesh. Deprecated
    """
    g = grid.Grid(fargodir,nx,ny,nz,domain,nout,quiet)
    return g,g.mesh

def create_mesh(fargodir, n=0, quiet=False):
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

    Returns
    -------
    Mesh
    """
    m = mesh.Mesh(fargodir,n,quiet)
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

    Returns
    -------
    Particle
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

    Returns
    -------
    Planet
    """
    p = planet.Planet(mesh,planet_no,name)
    return p

def read_params(paramfile):
    """Helper function for partracio.read_input().

    Parameters
    ----------
    paramfile : str
        Name of parameter file to readin.

    Returns
    -------
    dict
        dictionary of params from param file
    """
    return partraceio.read_input(paramfile)

def unstack(a, axis=-1):
    """Helper function to unpack arrays. Gradient arrays have shape
    (nz,ny,nx,3) for example. This function will return 3 arrays each
    with shape (nz,ny,nx).
    """
    import numpy as np
    return np.moveaxis(a, axis, 0)

def get_nparts(partfile):

    npart = 0
    with open(partfile,'r') as f:
        for line in f:
            if not line.startswith('#'):
                npart+=1
    return npart
