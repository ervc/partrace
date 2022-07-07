import numpy as np
import matplotlib.pyplot as plt
from .constants import *

class Mesh():
    """Fargo domain mesh. Contains the domain, density, 
    and velocity mesh.
    """
    def __init__(self, fargodir, quiet=False):
        self.fargodir = fargodir
        self.ndim = 3
        self.read_variables()
        self.get_domain()
        if not quiet:
            self.confirm()
        self.state = {}
        self.n = {}

    def confirm(self):
        conf_msg = ''
        conf_msg += 'Mesh created\n'
        conf_msg += f'Read in from {self.fargodir}\n'
        conf_msg += f'nx, ny, nz = {self.nx, self.ny, self.nz}\n'
        conf_msg += f'ndim = {self.ndim}\n'
        conf_msg += f'coord. system = {self.variables["COORDINATES"]}\n'
        if 'UNITS' not in self.variables:
            conf_msg += f'units = code\n'
        elif self.variables['UNITS']=='0':
            conf_msg += f'units = code\n'
        else:
            conf_msg += f'units = {self.variables["UNITS"]}\n'
        conf_msg += (f'There are {len(self.variables)} additional '
                     + f'variables stored in Mesh.variables\n')

        print(conf_msg)

    def read_variables(self):
        """Reads variables.par file and cretes dict of variables. Also
        read in from summary0.dat to get scaling laws"""
        self.variables={}
        with open(self.fargodir+'/variables.par','r') as f:
            for line in f:
                label,data = line.split()
                self.variables[label] = data

        units = 'code'
        flags = []
        with open(self.fargodir+'/summary0.dat','r') as f:
            header = False
            section = ''
            for line in f:
                if line.startswith("==="):
                    header = not header
                    continue
                if header:
                    section = line.strip()
                elif section == 'COMPILATION OPTION SECTION:':
                    if line[:2] == '-D':
                        flags = line.split()
                elif section == 'PREPROCESSOR MACROS SECTION:':
                    if len(line.split()) > 1:
                        quantity = line.split()[0]
                        val = line.split()[-1]
                        self.variables[quantity] = val

        if '-DRESCALE' in flags:
            if '-DCGS' in flags:
                units = 'CGS'
            elif '-DMKS' in flags:
                units = 'MKS'
            else:
                raise Exception('Unable to determine units from summary0.dat'
                    +' see flags\n',flags)
        if 'UNITS' not in self.variables:
            self.variables['UNITS'] = units


    def get_domain(self,ghostcells=True):
        """Reads in domain.dat files and creates arrays of cell edge 
        and cell center values. Also determines dimensionality of
        output.

        Creates 1D arrays of edges and centers in default code output.
        Shapes:
          xedges = (nx+1,)
          yedges = (ny+1,)
          zedges = (nz+1,)
          xcenters = (nx,)
          ycenters = (ny,)
          zcenters = (nz,)
        Note: nz = 1 if ndim == 2

        Also creates grids of edges and centers in output coordinates
        and converts to cartesian coordinates.

        Default
          edges = dict; contains 'x','y','z'
          centers = dict; contains 'x','y','z'
        where:
          if coords == cylindrical:
            "x" = azimuth
            "y" = radius
            "z" = z
          if coords == spherical:
            "x" = azimuth
            "y" = radius
            "z" = polar
        shapes:
          edges['x'] = (nz+1,ny+1,nx+1) <- gives lower "x" edge of cell 
                         i,j,k, where "x" is in default code output
                         (usually "x" is azimuth for cyl or spherical)
          centers['x'] = (nz,ny,nx) <- gives "x" value at center of cell

        Cartesian
          cartedges = dict; contains 'x','y','z'
          cartcenters = dict; contains 'x','y','z'
        where:
          "x","y", and "z" are cartesian coordinates centered at star
        shapes:
          cartedges['x'] = (nz+1,ny+1,nx+1) <- gives lower x edge of cell
          cartcenters['x'] = (nz,nx,ny) <- gives center x value of cell

        IMPORTANT: x[k,j,i] is "next to" x[k,j,i+1] in the azimuth 
        direction, so dx = x[k,j,i+1]-x[k,j,i] may be positive or 
        negative, and depends on the cell. Cell i,j,k is still indexed
        by the original coordinate system, either cyl or spherical.
    
        """
        self._readin_edges(ghostcells=ghostcells)

        self._get_centers()

        # create arrays
        self.edges = {}
        self.edges['z'],self.edges['y'],self.edges['x'] = np.meshgrid(
            self.zedges,self.yedges,self.xedges,indexing='ij'
        )

        self.centers = {}
        self.centers['z'],self.centers['y'],self.centers['x'] = np.meshgrid(
            self.zcenters,self.ycenters,self.xcenters,indexing='ij'
        )

        # get cartesian coordinates
        self.cartcenters = {}
        self.cartedges = {}
        if self.variables['COORDINATES'] == 'spherical':
            self._get_cartgrid_from_sphere()
        elif self.variables['COORDINATES'] == 'cylindrical':
            self._get_cartgrid_from_cyl()

    def _readin_edges(self,ghostcells=True):
        """Helper function to readin domain[x,y,z].dat files"""
        # default number of ghost cells out, check fargo distribution
        NGHOST = 3

        # x edges = azimuth, no ghost cells
        self.xedges = []
        with open(self.fargodir+'/domain_x.dat','r') as f:
            for line in f:
                self.xedges.append(float(line))
        self.nx = len(self.xedges)-1
        self.xedges = np.array(self.xedges)

        # y edges = radial, contains ghost cells
        self.yedges = []
        allyedges = []
        with open(self.fargodir+'/domain_y.dat','r') as f:
            for line in f:
                allyedges.append(float(line))
        if ghostcells:
            self.yedges = list(allyedges[NGHOST:-NGHOST])
        else:
            self.yedges = list(allyedges)
        self.ny = len(self.yedges)-1
        self.yedges = np.array(self.yedges)

        # zedges = height or polar angle, may contain ghost cells, 
        # may not exist
        self.zedges = []
        allzedges = []
        with open(self.fargodir+'/domain_z.dat','r') as f:
            for line in f:
                allzedges.append(float(line))
        if allzedges[0] == allzedges[-1] == 0.0:
            # simulation is 2d
            self.ndim = 2
            self.zedges = [-1,1]
        else:
            if ghostcells:
                self.zedges = list(allzedges[NGHOST:-NGHOST])
            else:
                self.zedges = list(allzedges)
        self.nz = len(self.zedges)-1
        self.zedges = np.array(self.zedges)

    def _get_centers(self):
        """Helper function to get centers of arrays"""
        self.xcenters = list([(self.xedges[i]
                                + self.xedges[i+1])/2 for i in range(self.nx)])
        self.ycenters = list([(self.yedges[i]
                                + self.yedges[i+1])/2 for i in range(self.ny)])
        self.zcenters = list([(self.zedges[i]
                                + self.zedges[i+1])/2 for i in range(self.nz)])
        self.xcenters = np.array(self.xcenters)
        self.ycenters = np.array(self.ycenters)
        self.zcenters = np.array(self.zcenters)

    def _sphere2cart(self,az,r,pol):
        """convert spherical to cartesian"""
        x = r*np.cos(az)*np.sin(pol)
        y = r*np.sin(az)*np.sin(pol)
        z = r*np.cos(pol)
        return x,y,z

    def _cart2sphere(self,x,y,z):
        r = np.sqrt(x*x + y*y + z*z)
        az = np.arctan2(y,x)
        pol = np.arccos(z/r)
        return az,r,pol

    def _cyl2cart(self,az,r,z):
        """convert cylindrical to cartesian"""
        x = r*np.cos(az)
        y = r*np.sin(az)
        return x,y,z

    def _cart2cyl(self,x,y,z):
        r = np.sqrt(x*x + y*y)
        az = np.arctan2(y,x)
        return az,r,z

    def _vel_sphere2cart(self,az,r,pol,azdot,rdot,poldot):
        xdot = (np.cos(az)*np.sin(pol)*rdot 
                - r*np.sin(az)*np.sin(pol)*azdot
                + r*np.cos(az)*np.cos(pol)*poldot)
        ydot = (np.sin(az)*np.sin(pol)*rdot
                + r*np.cos(az)*np.sin(pol)*azdot
                + r*np.sin(az)*np.cos(pol)*poldot)
        zdot = np.cos(pol)*rdot - r*np.sin(pol)*poldot
        return xdot,ydot,zdot

    def _vel_cyl2cart(self,az,r,z,azdot,rdot,zdot):
        xdot = np.cos(az)*rdot - r*np.sin(az)*azdot
        ydot = np.sin(az)*rdot + r*np.cos(az)*azdot
        zdot = zdot
        return xdot,ydot,zdot

    def _get_cartgrid_from_sphere(self):
        """Helper function to convert spherical grid to cartesian grid"""
        (self.cartedges['x'],
         self.cartedges['y'],
         self.cartedges['z']) = self._sphere2cart(self.edges['x'],
                                                  self.edges['y'],
                                                  self.edges['z'])
        (self.cartcenters['x'],
         self.cartcenters['y'],
         self.cartcenters['z']) = self._sphere2cart(self.centers['x'],
                                                    self.centers['y'],
                                                    self.centers['z'])

    def _get_cartgrid_from_cyl(self):
        """Helper function to convert cylindrical grid to cartesian grid"""
        (self.cartedges['x'],
         self.cartedges['y'],
         self.cartedges['z']) = self._cyl2cart(self.edges['x'],
                                               self.edges['y'],
                                               self.edges['z'])
        (self.cartcenters['x'],
         self.cartcenters['y'],
         self.cartcenters['z']) = self._cyl2cart(self.centers['x'],
                                                 self.centers['y'],
                                                 self.centers['z'])

    def get_cart_vel(self):
        """returns 3D arrays of cartesian velocities. Note: velocities
        are stored at cell edges, but we use the centers here...
        Need to check this!
        """
        if self.ndim == 2:
            self.state['gasvz'] = np.zeros_like(self.state['gasvx'])
        if self.variables['COORDINATES'] == 'spherical':
            xdot,ydot,zdot = self._vel_sphere2cart(
                self.centers['x'],self.centers['y'],self,centers['z'],
                self.state['gasvx'],self.state['gasvy'],self.state['gasvz'])
        elif self.variables['COORDINATES'] == 'cylindrical':
            xdot,ydot,zdot = self._vel_cyl2cart(
                self.centers['x'],self.centers['y'],self,centers['z'],
                self.state['gasvx'],self.state['gasvy'],self.state['gasvz'])
        else:
            raise Exception("Coordinates cannot be determined")
            return None
        return xdot,ydot,zdot

        

    def read_state(self,state,n=-1):
        """readin the grid data for output 'state' at output number n. 
        Default n=-1 gives last ouptut
        """
        MAXN = 1000

        if n < 0:
            lastn = 0
            for i in range(MAXN+1):
                try:
                    open(self.fargodir+f'/{state}{i}.dat')
                    lastn = i
                except FileNotFoundError:
                    break
            if lastn == MAXN-1:
                print(f'WARNING: Reading in state from output {lastn}=MAXN.'
                        + ' May not be the last output.')
            n = lastn+1+n

        self.n[state] = n
        statefile = self.fargodir+f'/{state}{n}.dat'

        state_arr = np.fromfile(statefile).reshape(self.nz,self.ny,self.nx)
        self.state[state] = state_arr
        return state_arr

    def plot_state(self,state,ax=None,log=True,itheta=-1,yunits=None,
                    *args,**kwargs):
        """Create 2d plot of state and return image"""
        if state not in self.state:
            arr = self.read_state(state)
        else:
            arr = self.state[state]

        if ax == None:
            fig,ax = plt.subplots()

        label = state
        if log:
            arr = np.log10(arr)
            label = 'log '+state

        yscale = 1
        if yunits == 'au':
            yscale = 1/AU

        im = ax.pcolormesh(self.xedges,self.yedges*yscale,arr[itheta],
                            *args,**kwargs)
        cb = plt.colorbar(im,ax=ax,label=label,location='bottom')

        dt_out = float(self.variables['DT'])*float(self.variables['NINTERM'])
        GM = float(self.variables['G'])*float(self.variables['MSTAR'])
        R = float(self.variables['R0'])
        R3 = R*R*R
        torbit = TWOPI * np.sqrt(R3/GM)
        norbit = self.n[state]*dt_out/torbit
        title = f'Norbit = {norbit:g}'
        if self.ndim == 3:
            title+=f'\ntheta = {self.zcenters[itheta]:.3f}'
        ax.set_title(title)

        units = 'unitless'
        if yunits:
            units = yunits

        else:
            if self.variables['UNITS'] == 'CGS':
                units = 'cm'
            elif self.variables['UNITS'] == 'MKS':
                units = 'm'
        ax.set(
            xlabel='azimuth [radians]',
            ylabel=f'radius [{units}]'
        )

        return im


    def get_cell_from_pol(self,x,y,z=0):
        """Determine the i,j,k cell index for polar coordinate x,y,z 
        in specified units
        """

        flag = 0
        which = ''
        if x < self.xedges[0] or x > self.xedges[-1]:
            flag += 1
            which += 'x'
        if y < self.yedges[0] or y > self.yedges[-1]:
            flag += 1
            which += 'y'
        if z < self.zedges[0] or z > self.zedges[-1]:
            flag += 1
            which += 'z'

        errmessage = (f'Error: coordinate {x,y,z} is not in grid\n'
                        + f'Problem in {which} direction\n'
                        + f'  xlims = {self.xedges[0], self.xedges[-1]}\n'
                        + f'  ylims = {self.yedges[0], self.yedges[-1]}\n'
                        + f'  zlims = {self.zedges[0], self.zedges[-1]}')

        if flag > 0:
            # raise Exception(errmessage)
            # print(errmessage)
            return None

        icell = 0
        for i in range(self.nx):
            icell = i
            if self.xedges[i+1] >= x:
                break
        jcell = 0
        for j in range(self.ny):
            jcell = j
            if self.yedges[j+1] >= y:
                break
        kcell = 0
        for k in range(self.nz):
            kcell = k
            if self.zedges[k+1] >= z:
                break

        return icell,jcell,kcell


    def get_cell_from_cart(self,x,y,z):
        if self.variables['COORDINATES'] == 'spherical':
            az,r,pol = self._cart2sphere(x,y,z)
            return self.get_cell_from_pol(az,r,pol)
        elif self.variables['COORDINATES'] == 'cylindrical':
            az,r,z = self._cart2cyl(x,y,z)
            return self.get_cell_from_pol(az,r,z)
        else:
            raise Exception("Coordinate system is uncertain,"
                + " cannot convert to default coordinates.")
            return None

    def get_rho_from_cart(self,x,y,z):
        '''get density at a given cartesian location from the disk'''
        cellind = self.get_cell_from_cart(x,y,z)
        if cellind is None:
            # try reflecting over z=0 at midplane
            cellind = self.get_cell_from_cart(x,y,-z)
            if cellind is None:
                # if still nothing then return NaN
                return np.nan
        icell,jcell,kcell = cellind
        return self.state['gasdens'][kcell,jcell,icell]

    def get_state_from_cart(self,state,x,y,z):
        """Get the value of 'state' at a given cartesian coordinate"""
        cellind = self.get_cell_from_cart(x,y,z)
        if cellind is None:
            # try reflecting over z=0 at midplane
            cellind = self.get_cell_from_cart(x,y,-z)
            if cellind is None:
                # if still nothing then return NaN
                return np.nan
        icell,jcell,kcell = cellind
        return self.state[state][kcell,jcell,icell]




