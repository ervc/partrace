"""
Particle class for tracer particles that move through the Mesh.
"""


import numpy as np
from .constants import *

DEBUG = False

class Particle(object):
    """A particle that can move through the disk. Particles are
    initialized with velocity equal to the gas around them.
    INPUTS
    ------
    mesh : Mesh
        partrace mesh object that contains disk information for the
        particle to move through
    x, y, z : float
        starting location for the particle
    OPTIONAL
    a : float
        particle size in cm
        default: 0.01 cm
    rho_s : float
        particle density in g cm-3
        default: 2.0 g cm-3
    """
    def __init__(self,mesh,x,y,z,a=0.01,rho_s=2):
        self.a = a
        self.rho_s = rho_s

        self.mesh = mesh
        self.pos = np.array([x,y,z])
        self.pos0 = np.array([x,y,z])

        r = np.sqrt(x*x + y*y)

        # # initialize with keplerian velocity
        # vk = r*mesh.get_Omega(x,y,z) - r*float(mesh.variables['OMEGAFRAME'])
        # th = np.arctan2(y,x)
        # self.vel = ([vk*np.cos(th),vk*np.sin(th),0])

        # initialize velocity as same as gas
        vgx,vgy,vgz = mesh.get_gas_vel(x,y,z)
        self.vel = np.array([vgx,vgy,vgz])
        
        self.vel0 = np.array(self.vel)

    def update_position(self,x,y,z):
        """update the position of the particle"""
        self.pos = np.array([x,y,z])

    def update_velocity(self,vx,vy,vz):
        """update the velocity of the particle"""
        self.vel = np.array([vx,vy,vz])

    def get_stokes(self):
        """get the stokes number at the current location"""
        om = self.mesh.get_Omega(*self.pos)
        rho_g = self.mesh.get_rho(*self.pos)
        cs = self.mesh.get_soundspeed(*self.pos)
        return self.a*self.rho_s/(rho_g*cs)*om

    def get_drag_coeff(self):
        """drag coefficient Tanigawa et al. 2014 (Watanabe+Ida 1997)"""
        nu = self.mesh.get_diffusivity(*self.pos)
        vgx,vgy,vgz = self.mesh.get_gas_vel(*self.pos)
        vgas = np.array([vgx,vgy,vgz])
        vpar = np.array(self.vel)    
        u = np.linalg.norm(vpar-vgas)
        c = self.mesh.get_soundspeed(*self.pos)

        # Reynolds number
        R = 2*self.a*u/nu

        # Mach number
        M = u/c

        # correction factor
        w = 0.4
        if R > 2e5: w=0.2

        if u==0:
            return 1e5
        Cd = 1/(1/(24/R + 40/(10+R)) + 0.23*M) + (2.0-w)*M/(1.6 + M) + w
        return Cd

    def get_dragAccel(self):
        """find the epstein drag acceleration vector"""
        vgx,vgy,vgz = self.mesh.get_gas_vel(*self.pos)
        vgas = np.array([vgx,vgy,vgz])
        
        vpar = np.array(self.vel)
        
        vtil = vgas - vpar

        rho_g = self.mesh.get_rho(*self.pos)
        cs = self.mesh.get_soundspeed(*self.pos)
        return rho_g*cs/self.a/self.rho_s*vtil

    def get_newer_dragAccel(self):
        """Get stokes drag acceleration vector"""
        vgx,vgy,vgz = self.mesh.get_gas_vel(*self.pos)
        vgas = np.array([vgx,vgy,vgz])    
        vpar = np.array(self.vel)     
        Du = vpar - vgas
        Dumag = np.linalg.norm(Du)
        rhog = self.mesh.get_rho(*self.pos)

        CD = self.get_drag_coeff()

        ##### !!!! THIS HAS AN EXTRA FACTOR OF a BUT it WORKS !!!! #####
        adrag = -3/8 * CD * rhog/self.rho_s/self.a/self.a * Dumag * Du
        return adrag


    def get_gravAccel(self):
        """find the acceleration vector due to the star gravity"""
        G = float(self.mesh.variables['G'])
        MSTAR = float(self.mesh.variables['MSTAR'])
        GM = G*MSTAR
        X = self.pos
        dstar = np.linalg.norm(X)
        return -GM/dstar**3 * X

    def get_planetAccel(self,planet):
        """grav acceleration vector due to planet"""
        if planet is None:
            return 0
        X = self.pos
        Xp = planet.pos

        Mp = planet.mass
        GM = G*Mp
        dplanet = np.linalg.norm(X-Xp)
        return -GM/dplanet**3 * (X-Xp)

    def get_centAccel(self):
        """Centrifugal acceleration vector due to rotating frame"""
        omegaframe = float(self.mesh.variables['OMEGAFRAME'])
        x,y,z = self.pos
        vx,vy,vz = self.vel
        ax = 2*omegaframe*vy + x*omegaframe**2
        ay = -2*omegaframe*vx + y*omegaframe**2
        az = 0
        return np.array([ax,ay,az])

    def get_particleDiffusivity(self):
        """Particle diffusivity, Youdin & Lithwick 2007"""
        Dgas = self.mesh.get_diffusivity(*self.pos)
        St = self.get_stokes()
        return Dgas/(1+St**2)


    def total_accel(self,planet):
        """Get total acceleration acting on the particle"""
        tot = 0
        drag = self.get_newer_dragAccel()
        tot += drag
        star = self.get_gravAccel()
        tot += star
        if planet is not None:
            #plan = self.get_planetAccel(planet)
            #tot += plan
        cent = self.get_centAccel()
        tot += cent
        return tot

    def get_stokes_grad(self):
        """return the gradient of the stokes value at the current
        location"""
        x,y,z = self.pos
        r = np.sqrt(x*x + y*y)
        dx = 0.01*r
        dy = 0.01*r
        St0 = self.get_stokes()

        def stokes(x,y,z):
            # helper function to find stokes number
            om = self.mesh.get_Omega(x,y,z)
            rho_g = self.mesh.get_rho(x,y,z)
            cs = self.mesh.get_soundspeed(x,y,z)
            return self.a*self.rho_s/(rho_g*cs)*om

        Stx = stokes(x+dx,y,z)
        dStdx = (Stx-St0)/dx
        Sty = stokes(x,y+dy,z)
        dStdy = (Sty-St0)/dy
        if self.mesh.ndim == 3:
            dz = 0.01*r*float(self.mesh.variables['ASPECTRATIO'])
            Stz = self.get_diffusivity(z,y,z+dz)
            dStdz = (Stz-St0)/dz
        else:
            dStdz = np.zeros_like(x)
        return dStdx,dStdy,dStdz

    def get_vdiff(self):
        """Calculate the diffusive velocity based on diffusivity 
        gradient where:
        vdiff = d/dx D = d/dx Dg/(1+st^2)
        """
        Dg = self.mesh.get_diffusivity(*self.pos)
        dDgdx = np.array(self.mesh.get_diff_grad(*self.pos))
        St = self.get_stokes()
        dStdx = np.array(self.get_stokes_grad())
        p1 = dDgdx/(1+St**2)
        p2 = -Dg/((1+St**2)**(2))*2*St*dStdx
        return np.array([p1+p2]).reshape(3,)

    def get_vrho(self):
        """Calculate the diffusive velocity based on density gradient
        where:
        vrho = D/rho * d/dx[rho]
        """
        drhodx,drhody,drhodz = self.mesh.get_rho_grad(*self.pos)
        D = self.get_particleDiffusivity()
        rhog = self.mesh.get_rho(*self.pos)

        return np.array([D/rhog*drhodx, D/rhog*drhody, D/rhog*drhodz])


    def get_veff(self):
        """determine the effective velocity of the particle including
        diffusive effects from diffusivity and density gradients
        """
        veff = np.array(self.vel)
        
        # vdiff = dD/dx
        #vdiff = self.get_vdiff()
        #veff += vdiff

        # vrho = D/rho drho/dx
        #vrho = self.get_vrho()
        #veff += vrho

        return veff