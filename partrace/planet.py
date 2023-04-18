"""
Contains planet class for planet embedded in mesh
"""

import numpy as np
import matplotlib.pyplot as plt
from .constants import *

class Planet():
    """Planet embedded in mesh, information is read in from 
    planet{planet_no}.dat in mesh.fargodir

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

    Attributes
    ----------
    mesh : Mesh
        Mesh object that planet is embedded in
    name : str
        Name of planet for tracking
    fname : str
        Name of fargo output file with planet data
    nout : int
        output number for planet data
    pos : ndarray (3,)
        Cartesian position of planet
    vel : ndarray (3,)
        Cartesian velocity of planet
    mass : float
        Mass of planet
    time : float
        Time of output in seconds
    omegaframe : float
        Rotation rate of Mesh
    hill : float
        Hill radius size
    bondi : float
        bondi radius size
    envelope : float
        size of planetary accretion envelope

    """
    def __init__(self, mesh, planet_no=0, name='planet'):
        self.mesh = mesh
        self.name = name
        self.fname = f'planet{planet_no}.dat'
        self.nout = mesh.n['gasdens']
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.mass = 0
        self.time = 0
        self.omegaframe = 0
        self.read_data()
        self.hill = self.get_hill_radius()
        self.bondi = self.get_bondi_radius()
        self.envelope = self.get_envelope_radius()
        if not mesh.quiet:
            print(f'Planet {self.name} read in from {self.fname}')
            print(f'  planet is located at {self.pos}\n')

    def read_data(self):
        """reads in data from planet file and stores as class values"""
        with open(self.mesh.fargodir+'/'+self.fname,'r') as f:
            for line in f:
                nout = int(line.split()[0])
                if nout!=self.nout:
                    continue
                nout,x,y,z,vx,vy,vz,mass,time,omegaframe = map(float,
                                                               line.split())
                self.pos = np.array([x,y,z])
                self.vel = np.array([vx,vy,vz])
                self.mass = mass
                self.time = time
                self.omegaframe = omegaframe
        if all(x==0 for x in self.pos):
            # catch case where planet output is messed up
            with open(self.mesh.fargodir+'/'+self.fname,'r') as f:
                lines = f.readlines()
                line0 = lines[0]
                nout,x,y,z,vx,vy,vz,mass,time,omegaframe = map(float,
                                                               line.split())

                self.pos = np.array([x,y,z])
                self.vel = np.array([vx,vy,vz])
                self.mass = mass
                self.time = time
                self.omegaframe = omegaframe

        if self.mesh.rescaled:
            R0    = float(self.mesh.variables['R0'])
            G     = float(self.mesh.variables['G'])
            MSTAR = float(self.mesh.variables['MSTAR'])
            LENGTH = R0
            MASS   = MSTAR
            TIME   = np.sqrt(R0*R0*R0/G/MSTAR)
            self.pos *= LENGTH
            self.vel *= LENGTH/TIME
            self.mass *= MASS
            self.time *= TIME
            self.omegaframe *= 1/TIME

    def get_hill_radius(self):
        """determine the hill radius of the planet"""
        x,y,z = self.pos
        sma = np.sqrt(x*x + y*y)
        Mpl = self.mass
        Mstar = float(self.mesh.variables['MSTAR'])
        return sma*(Mpl/3/Mstar)**(1/3)

    def get_bondi_radius(self):
        """determine the bondi radius of the planet"""
        x,y,z = self.pos
        G = float(self.mesh.variables['G'])
        Mpl = self.mass
        cs = self.mesh.get_soundspeed(x,y,z)
        return 2*G*Mpl/cs/cs

    def get_envelope_radius(self):
        """determine the envelope radius around the planet"""
        rh = self.hill
        rb = self.bondi
        return min(rh/4,rb)



