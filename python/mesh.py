import numpy as np
import matplotlib.pyplot as plt
from .constants import *

class Mesh():
    """Fargo domain mesh. Contains the domain, density, 
    and velocity mesh.
    """
    def __init__(self, fargodir, states='all', n=-1, quiet=False):
        self.fargodir = fargodir
        self.ndim = 3
        self.read_variables()
        self.get_domain()
        self.quiet = quiet
        if not quiet:
            self.confirm()
        self.state = {}
        self.interpolators = {}
        self.n = {}
        if states is None:
            if not quiet:
                print('Initialized, self.state and self.n are empty,'
                    + ' states can be read in using read_state()\n')
        else:
            if states == 'all':
                states = ['gasdens','gasvx','gasvy','gasvz','gasenergy']
            for state in states:
                if state == 'gasvz' and self.ndim == 2:
                    self.state['gasvz'] = np.zeros_like(self.state['gasvx'])
                else:
                    self.read_state(state,n)
                interp = self.create_interpolator(state)
                self.interpolators[state] = interp
            if not quiet:
                outstr = ('Initialized, self.states contains:\n'
                    + f'{list(self.state.keys())}\n'
                    + 'read in from outputs:\n')
                for state in self.n.keys():
                    outstr += f'n = {self.n[state]} for {state}\n'
                print(outstr)


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

    def read_variables(self,n=0):
        """Reads variables.par file and cretes dict of variables. Also
        read in from summary0.dat to get scaling laws"""
        self.variables={}
        with open(self.fargodir+'/variables.par','r') as f:
            for line in f:
                label,data = line.split()
                self.variables[label] = data

        units = 'code'
        flags = []
        with open(self.fargodir+f'/summary{n:d}.dat','r') as f:
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
                raise Exception('Unable to determine units from'
                    +f' summary{n}.dat, see flags\n',flags)
        self.variables['FLAGS'] = flags
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
                - np.sin(az)*np.sin(pol)*azdot
                + np.cos(az)*np.cos(pol)*poldot)
        ydot = (np.sin(az)*np.sin(pol)*rdot
                + np.cos(az)*np.sin(pol)*azdot
                + np.sin(az)*np.cos(pol)*poldot)
        zdot = np.cos(pol)*rdot - r*np.sin(pol)*poldot
        return xdot,ydot,zdot


    def _vel_cyl2cart(self,az,r,z,azdot,rdot,zdot):
        xdot = np.cos(az)*rdot - np.sin(az)*azdot
        ydot = np.sin(az)*rdot + np.cos(az)*azdot
        zdot = zdot
        return xdot,ydot,zdot


    def read_state(self,state,n=-1):
        """readin the grid data for output 'state' at output number n. 
        Default n=-1 gives last ouptut.

        This can be called by the user or called automatically in init()
        by specifying the 'states' keyword.
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


    def get_cell_from_cyl(self,az,r,z):
        if self.variables['COORDINATES'] == 'cylindrical':
            return self.get_cell_from_pol(az,r,z)
        elif self.variables['COORDINATES'] == 'spherical':
            s = np.sqrt(r*r + z*z) # spherical radius
            pol = np.arccos(z/s)
            return self.get_cell_from_pol(az,s,pol)
        else:
            raise Exception("Coordinate system is uncertain,"
                + " cannot convert to default coordinates.")
            return None


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


    def get_state_from_cart(self,state,x,y,z=0):
        """Get the value of 'state' at a given cartesian coordinate
        """

        # setup interpolator
        interp = self.interpolators[state]

        iterb = True
        try:
            iter(x)
        except:
            iterb = False

        # do the interpolation
        if self.variables['COORDINATES'] == 'cylindrical':
            az,r,z = self._cart2cyl(x,y,z)
            if self.ndim == 2:
                if iterb:
                    return interp(np.stack([r,az],axis=-1))
                else:
                    return interp(np.stack([r,az],axis=-1))[0]
            else:
                if iterb:
                    return interp(np.stack([z,r,az],axis=-1))
                else:
                    return interp(np.stack([z,r,az],axis=-1))[0]
        elif self.variables['COORDINATES'] == 'spherical':
            az,r,pol = self._cart2sphere(x,y,z)
            if self.ndim == 2:
                return interp(np.stack([r,az],axis=-1))
            else:
                return interp(np.stack([pol,r,az],axis=-1))

        else:
            raise Exception('uncertain on coordinates,'
                  +' cannot interpolate')
            return None


    def create_interpolator(self,state):
        """Creates the interpolator function using scipy interpolate
        If the dimension of the mesh is 2d then interpolator will be 2d,
        this avoids issues with interpolating from a single z component
        """
        from scipy.interpolate import RegularGridInterpolator

        Y = self.ycenters

        # reflective z boundary
        # z  = pi/2 - delta     delta = pi/2 - z
        # z' = pi/2 + delta
        # z' = (pi/2 - delta) + 2*delta
        # z' = z + 2*(pi/2 - z)
        #    = z + pi - 2*z
        #    = pi-z
        lenz = len(self.zcenters)
        Z = np.zeros(2*lenz)
        Z[0:lenz] = self.zcenters
        Z[lenz:] = PI - self.zcenters[::-1]
        
        # periodic boundary conditions for x axis
        X = np.zeros(len(self.xcenters)+2)

        # create the array and fill in extra X columns
        arr = np.zeros((len(Z),len(Y),len(X)))
        for i in range(self.nx):
            X[i+1] = self.xcenters[i]
            arr[0:lenz,:,i+1] = self.state[state][:,:,i]
        X[0]  = self.xcenters[-1] - TWOPI
        X[-1] = self.xcenters[0]  + TWOPI
        arr[0:lenz,:,0]  = self.state[state][:,:,-1]
        arr[0:lenz,:,-1] = self.state[state][:,:,0]
        
        # flip and repeat over the z axis
        arr[-1:lenz-1:-1,:,:] = arr[0:lenz,:,:]

        # setup interpolator
        if self.ndim == 3:
            interp = RegularGridInterpolator(
                (Z,Y,X),arr,method='linear',bounds_error=False)
        else:
            interp = RegularGridInterpolator(
                (Y,X),arr[0],method='linear',bounds_error=False)

        return interp

    def get_scaleheight(self,x,y,z=0):
        r = np.sqrt(x*x + y*y)
        ar = float(self.variables['ASPECTRATIO'])
        fi = float(self.variables['FLARINGINDEX'])
        R0 = float(self.variables['R0'])
        return r*ar*(r/R0)**fi


    def get_Omega(self,x,y,z=0):
        G = float(self.variables['G'])
        Mstar = float(self.variables['MSTAR'])
        GM = G*Mstar
        r = np.sqrt(x**2 + y**2) # <--- question: +z**2 ?
        r3 = r*r*r
        return np.sqrt(GM/r3)

    def get_soundspeed(self,x,y,z=0):
        # H = self.get_scaleheight(x,y,z)
        # Om = self.get_Omega(x,y,z)
        # return H*Om
        cs = self.get_state_from_cart('gasenergy',x,y,z)
        return cs


    def get_diffusivity(self,x,y,z=0):
        """Determine the diffusivity at a given location
        D = alpha*cs*H = alpha*H**2*omega = nu
        """
        az,r,z = self._cart2cyl(x,y,z)
        if '-DVISCOSITY' in self.variables['FLAGS']:
            return np.ones_like(x)*float(self.variables['NU'])
        elif '-DALPHAVISCOSITY' in self.variables['FLAGS']:
            H = self.get_scaleheight(x,y,z)
            cs = self.get_soundspeed(x,y,z)
            alpha = float(self.variables['ALPHA'])
            return alpha*cs*H
        else:
            if not self.quiet:
                print('Viscosity cannot be determined from variables.'
                    +' Using D=0')
            return 0

    def get_rho(self,x,y,z=0):
        """Determine the density at a given location"""
        if self.ndim == 2:
            # need to convert surface density to midplane density
            sigma = self.get_state_from_cart('gasdens',x,y,z)
            H = self.get_scaleheight(x,y,z)
            rho = sigma/np.sqrt(2*PI)/H
        else:
            rho = self.get_state_from_cart('gasdens',x,y,z)
        return rho

    """
    def get_gas_vel(self,x,y,z=0):
        omega0 = float(self.variables['OMEGAFRAME'])
        az,r,z = self._cart2cyl(x,y,z)
        azdotprime = self.get_state_from_cart('gasvx',x,y,z)
        azdot = azdotprime + r*omega0
        rdot = self.get_state_from_cart('gasvy',x,y,z)
        zdot = self.get_state_from_cart('gasvz',x,y,z)

        xdot = np.cos(az)*rdot - r*np.sin(az)*azdot
        ydot = np.sin(az)*rdot + r*np.cos(az)*azdot
        xdotp = xdot - omega0*y
        ydotp = ydot + omega0*x
        zdotp = zdot

        return xdotp,ydotp,zdotp
    """

    def get_gas_vel(self,x,y,z=0):
        if self.variables['COORDINATES'] == 'cylindrical':
            az,r,z = self._cart2cyl(x,y,z)
            azdot = self.get_state_from_cart('gasvx',x,y,z)
            rdot = self.get_state_from_cart('gasvy',x,y,z)
            zdot = self.get_state_from_cart('gasvz',x,y,z)
            xdot,ydot,zdot = self._vel_cyl2cart(az,r,z,azdot,rdot,zdot)
        elif self.variables['COORDINATES'] == 'spherical':
            az,r,pol = self._cart2sphere(x,y,z)
            azdot = self.get_state_from_cart('gasvx',x,y,z)
            rdot = self.get_state_from_cart('gasvy',x,y,z)
            poldot = self.get_state_from_cart('gasvz',x,y,z)
            xdot,ydot,zdot = self._vel_sphere2cart(az,r,pol,azdot,rdot,poldot)
        else:
            raise Exception('Coordinates cannot be determined!'
                +' Cannot calculate gas velocity.')
        return xdot,ydot,zdot

    def get_diff_grad(self,x,y,z=0):
        r = np.sqrt(x*x + y*y)
        dx = 0.01*r
        dy = 0.01*r
        diff0 = self.get_diffusivity(x,y,z)
        diffx = self.get_diffusivity(x+dx,y,z)
        dDdx = (diffx-diff0)/dx
        diffy = self.get_diffusivity(x,y+dy,z)
        dDdy = (diffy-diff0)/dy
        if self.ndim == 3:
            dz = 0.01*r*float(self.variables['ASPECTRATIO'])
            diffz = self.get_diffusivity(z,y,z+dz)
            dDdz = (diffz-diff0)/dz
        else:
            dDdz = np.zeros_like(x)
        return dDdx,dDdy,dDdz

    def get_rho_grad(self,x,y,z=0):
        r = np.sqrt(x*x + y*y)
        dx = 0.01*r
        dy = 0.01*r
        rho0 = self.get_rho(x,y,z)
        rhox = self.get_rho(x+dx,y,z)
        drhox = (rhox-rho0)/dx
        rhoy = self.get_rho(x,y+dy,z)
        drhoy = (rhoy-rho0)/dy
        if self.ndim == 3:
            dz = 0.01*r*float(self.variables['ASPECTRATIO'])
            rhoz = self.get_rho(z,y,z+dz)
            drhoz = (rhoz-rho0)/dz
        else:
            drhoz = np.zeros_like(x)
        return drhox,drhoy,drhoz




