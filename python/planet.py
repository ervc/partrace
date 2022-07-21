import numpy as np
import matplotlib.pyplot as plt
from .constants import *

class Planet():
    """docstring for Planet"""
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
        print(f'Planet {self.name} read in from {self.fname}')
        print(f'  planet is located at {self.pos}')

    def read_data(self):
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

    def get_hill_radius(self):
        x,y,z = self.pos
        sma = np.sqrt(x*x + y*y)
        Mpl = self.mass
        Mstar = float(self.mesh.variables['MSTAR'])
        return sma*(Mpl/3/Mstar)**(1/3)

    def get_bondi_radius(self):
        G = float(self.mesh.variables['G'])
        Mpl = self.mass
        cs = self.mesh.get_soundspeed(*self.pos)
        return 2*G*Mpl/cs/cs

    def get_envelope_radius(self):
        rh = self.hill
        rb = self.bondi
        return min(rh/4,rb)



