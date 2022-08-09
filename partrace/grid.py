"""
Class for regridding mesh onto cartesian grid.

Use of this is NOT recommeded, instead use mesh.interpolators to 
get data as needed at a given location. Once interpolators are made
slow-down is negligable, plus interpolation on the cartesian grid
would have to be done anyway.
"""


import numpy as np
import matplotlib.pyplot as plt

from .mesh import Mesh
from .constants import *

class Grid():
    """Cartesian regrid from fargo mesh input"""
    def __init__(self,fargodir,nx,ny,nz=1,domain=None,
                 nout=-1,quiet=False):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.domain = domain
        self.nout = nout
        self.quiet = quiet

        # readin underlying fargo mesh
        # this also makes it accessible for later!
        self.mesh = Mesh(fargodir,states='all',n=nout)
        self.ndim = self.mesh.ndim

        # redo the grid in cartesian elements to make easier
        self.regrid()

    def regrid(self):
        print('Regridding data')

        # set up the domain of the new grid
        if self.domain is None:
            xlo = np.min(self.mesh.cartedges['x'])
            xhi = np.max(self.mesh.cartedges['x'])
            ylo = np.min(self.mesh.cartedges['y'])
            yhi = np.max(self.mesh.cartedges['y'])
            zlo = np.min(self.mesh.cartedges['z'])
            zhi = np.max(self.mesh.cartedges['z'])
        elif self.ndim == 2:
            xlo,xhi,ylo,yhi = domain
            zlo = -1.
            zhi = 1.
        elif self.ndim == 3:
            xlo,xhi,ylo,yhi,zlo,zhi = domain
        else:
            raise Exception('Problem reading in domain, does dim of'
                +' domain match dim of fargo output?')

        # create the arrays for the x, y, and z values
        X = np.linspace(xlo,xhi,self.nx)
        Y = np.linspace(ylo,yhi,self.ny)
        if self.ndim == 2:
            Z = np.array([0])
        else:
            Z = np.linspace(zlo,zhi,self.nz)

        # create the Grid.grid. Everything will be of size (nz,ny,nx)
        # in a dictionary. So grid['x'][kji] will give the x value at
        # the cell i,j,k
        self.grid = {}
        doms = ['x','y','z']
        stats = ['gasdens','gasvx','gasvy']
        labels = ['gasdens','gasvaz','gasvr']
        if self.ndim == 3:
            stats.append('gasvz')
            if self.mesh.variables['COORDINATES'] == 'spherical':
                labels.append('gasvpol')
            elif self.mesh.variables['COORDINATES'] == 'cylindrical':
                labels.append('gasvz')
        arrsize = (self.nz,self.ny,self.nx)
        for dom in doms:
            self.grid[dom] = np.zeros(arrsize)
            

        for label,stat in zip(labels,stats):
            print(f'  {stat}')
            self.grid[label] = np.zeros(arrsize)
            interp = self.mesh.create_interpolator(stat)
            for k in range(self.nz):
                z = Z[k]
                for j in range(self.ny):
                    y = Y[j]
                    for i in range(self.nx):
                        x = X[i]
                        if stat == 'gasdens':
                            self.grid['x'][k,j,i] = x
                            self.grid['y'][k,j,i] = y
                            self.grid['z'][k,j,i] = z

                        # interpolate
                        if self.mesh.variables['COORDINATES'] == 'cylindrical':
                            az,r,z = self.mesh._cart2cyl(x,y,z)
                            if self.ndim == 2:
                                val = interp([r,az])
                            else:
                                val = interp([z,r,az])
                        elif (self.mesh.variables['COORDINATES'] == 
                                                                  'spherical'):
                            az,r,pol = self.mesh._cart2sphere(x,y,z)
                            if self.ndim == 2:
                                val = interp([r,az])
                            else:
                                val = interp([pol,r,az])

                        self.grid[label][k,j,i] = val

        # create arrays of zeros for compatibility              
        if self.ndim == 2:
            if self.mesh.variables['COORDINATES'] == 'spherical':
                self.grid['gasvpol'] = np.zeros_like(self.grid['gasvaz'])
            if self.mesh.variables['COORDINATES'] == 'cylindrical':
                self.grid['gasvz'] = np.zeros_like(self.grid['gasvaz'])
        print('Calculating Velocities...')                
        self._get_cartvels()
        print('Done!\n')

    def _get_cartvels(self):
        """Helper function to calculate the cartesian velocities"""
        if self.mesh.variables['COORDINATES'] == 'spherical':
            az = np.arctan2(self.grid['y'],self.grid['x'])
            azdot = self.grid['gasvaz']
            r = np.sqrt(
                    self.grid['x']**2 + self.grid['y']**2 + self.grid['z']**2)
            rdot = self.grid['gasvr']
            pol = np.arccos(self.grid['z']/r)
            poldot = self.grid['gasvpol']

            xdot,ydot,zdot = self._vel_sphere2cart(az,r,pol,azdot,rdot,poldot) 

        elif self.mesh.variables['COORDINATES'] == 'cylindrical':
            az = np.arctan2(self.grid['y'],self.grid['x'])
            azdot = self.grid['gasvaz']
            r = np.sqrt(self.grid['x']**2 + self.grid['y']**2)
            rdot = self.grid['gasvr']
            z = self.grid['z']
            zdot = self.grid['gasvz']

            xdot,ydot,zdot = self._vel_cyl2cart(az,r,z,azdot,rdot,zdot)

        self.grid['gasvx'] = xdot
        self.grid['gasvy'] = ydot
        self.grid['gasvz'] = zdot


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