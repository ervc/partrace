"""
Particle class for tracer particles that move through the Mesh.
"""


import numpy as np
from .constants import *

DEBUG = False

class Particle(object):
    """A particle that can move through the disk. Particles are
    initialized with velocity equal to the gas around them.

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

    Attributes
    ----------
    a : float
        particle size in cm
    rho_s : float
        particle density in g cm-3
    mesh : Mesh
        Mesh object the particle is embedded in
    pos : ndarray (3,)
        current cartesian position of particle
    vel : ndarray (3,)
        current cartesian velocity of particle
    pos0 : ndarray (3,)
        initial cartesian position of particle
    vel0 : ndarray (3,)
        initial cartesian velocity of particle
    """
    def __init__(self,mesh,x,y,z,a=0.01,rho_s=2):
        self.a = a
        self.rho_s = rho_s

        self.mesh = mesh
        self.pos = np.array([x,y,z])
        self.pos0 = np.array([x,y,z])

        r = np.sqrt(x*x + y*y)

        # initialize with keplerian velocity
        vk = r*mesh.get_Omega(x,y,z) - r*float(mesh.variables['OMEGAFRAME'])
        th = np.arctan2(y,x)
        self.vel = ([-vk*np.sin(th),vk*np.cos(th),0])

        # # initialize velocity as same as gas
        # vgx,vgy,vgz = mesh.get_gas_vel(x,y,z)
        # self.vel = np.array([vgx,vgy,vgz])
        
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
        rho_g = self.mesh.get_rho(*self.pos)[0]
        cs = self.mesh.get_soundspeed(*self.pos)[0]
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

        # avoid division by zero
        if u==0:
            return 1e5

        Cd = 1/(1/(24/R + 40/(10+R)) + 0.23*M) + (2.0-w)*M/(1.6 + M) + w
        return Cd

    def get_newer_dragAccel(self):
        """Get stokes drag acceleration vector. 
        (Tanigawa et al. 2014)"""
        vgx,vgy,vgz = self.mesh.get_gas_vel(*self.pos)
        vgas = np.array([vgx,vgy,vgz])    
        vpar = np.array(self.vel)     
        Du = vpar - vgas
        Dumag = np.linalg.norm(Du)
        rhog = self.mesh.get_rho(*self.pos)

        CD = self.get_drag_coeff()

        adrag = -3/8 * CD * rhog/self.rho_s/self.a * Dumag * Du
        return adrag

    def get_dragAccel(self):
        """find the epstein drag acceleration vector"""
        vgx,vgy,vgz = self.mesh.get_gas_vel(*self.pos)
        vgx = vgx[0]
        vgy = vgy[0]
        vgz = vgz[0]
        vgas = np.array([vgx,vgy,vgz])
        vpar = np.array(self.vel)
        
        vtil = vpar-vgas
        rho_g = self.mesh.get_rho(*self.pos)
        cs = self.mesh.get_soundspeed(*self.pos)
        return -rho_g*cs/self.a/self.rho_s*vtil


    def get_gravAccel(self,planet):
        """Find the total acceleration due to gravity, account for
        different stellar movement.
        """
        if planet is not None:
            if planet.mass == 0:
                Xs = np.zeros(3)
            else:
                Xp = planet.pos
                Mp = planet.mass
                Ms = float(self.mesh.variables['MSTAR'])
                # 0 = (XpMp + XsMs)/(Ms+Mp)
                # Xs = -XpMp/Ms
                Xs = -Xp * Mp/Ms
        else:
            Xs = np.zeros(3)

        astar = self.get_starAccel(Xs)
        aplan = self.get_planetAccel(planet)

        return astar + aplan

    def get_starAccel(self,Xs=0):
        """find the acceleration vector due to the star gravity"""
        G = float(self.mesh.variables['G'])
        MSTAR = float(self.mesh.variables['MSTAR'])
        GM = G*MSTAR
        X = self.pos
        dstar = np.linalg.norm(X-Xs)
        return -GM/dstar**3 * (X-Xs)

    def get_planetAccel(self,planet):
        """grav acceleration vector due to planet"""
        if planet is None:
            return 0
        if planet.mass == 0:
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
        drag = self.get_dragAccel()
        tot += drag
        grav = self.get_gravAccel(planet)
        tot += grav
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
            rho_g = self.mesh.get_rho(x,y,z)[0]
            cs = self.mesh.get_soundspeed(x,y,z)[0]
            return self.a*self.rho_s/(rho_g*cs)*om

        Stx = stokes(x+dx,y,z)
        dStdx = (Stx-St0)/dx
        Sty = stokes(x,y+dy,z)
        dStdy = (Sty-St0)/dy
        if self.mesh.ndim == 3:
            h = float(self.mesh.variables['ASPECTRATIO'])*(r/float(self.mesh.variables['R0']))**float(self.mesh.variables['FLARINGINDEX'])
            dz = 0.01*r*h
            Stz = stokes(x,y,z+dz)
            dStdz = (Stz-St0)/dz
        else:
            dStdz = np.zeros_like(x)
        return dStdx,dStdy,dStdz

    def get_Dp_grad(self):
        """
        Return cartesian diffusivity gradient in x, y, and z directions
        """
        x,y,z = self.pos
        r = np.sqrt(x*x + y*y)
        dx = 0.01*r
        dy = 0.01*r
        St0 = self.get_stokes()

        def stokes(x,y,z):
            # helper function to find stokes number
            om = self.mesh.get_Omega(x,y,z)
            rho_g = self.mesh.get_rho(x,y,z)[0]
            cs = self.mesh.get_soundspeed(x,y,z)[0]
            return self.a*self.rho_s/(rho_g*cs)*om
        def Dp(x,y,z):
            Dg = self.mesh.get_diffusivity(x,y,z)
            St = stokes(x,y,z)
            return Dg/(1+St**2)

        dDdx = (Dp(x+dx,y,z) - Dp(x-dx,y,z))/2/dx
        dDdy = (Dp(x,y+dy,z) - Dp(x,y-dy,z))/2/dy
        if self.mesh.ndim == 3:
            h = self.mesh.get_scaleheight(*self.pos)
            dz = 0.01*h
            dDdz = (Dp(x,y,z+dz) - Dp(x,y,z-dz))/2/dz
        else:
            dDdz = np.zeros_like(x)

        return dDdx, dDdy, dDdz
        


    def get_vdiff(self):
        """Calculate the diffusive velocity using finite differences
        v_diff = dDp/dx = [Dp(x+dx) - Dp(x-dx)]/2dx
        Dp = D/(1+St^2)
        """
        return np.array(self.get_Dp_grad()).reshape(3,)

        # Dg = self.mesh.get_diffusivity(*self.pos)
        # dDgdx = np.array(self.mesh.get_diff_grad(*self.pos)).reshape(3,)
        # # print(f'{dDgdx = }')
        # # print(f'{dDgdx.shape = }')
        # St = self.get_stokes()
        # dStdx = np.array(self.get_stokes_grad())
        # # print(f'{dStdx = }')
        # # print(f'{dStdx.shape = }')
        # p1 = dDgdx/(1+St**2)
        # p2 = -Dg/((1+St**2)**(2))*2*St*dStdx
        # return np.array([p1+p2]).reshape(3,)

    def get_vrho(self):
        """Calculate the diffusive velocity based on density gradient
        where:
        vrho = D/rho * d/dx[rho]
        """
        drhodx,drhody,drhodz = self.mesh.get_rho_grad(*self.pos)
        #print(f'{drhodx = }')
        #print(f'{drhodx.shape = }')
        drhodx = drhodx[0]
        drhody = drhody[0]
        drhodz = drhodz[0]
        D = self.get_particleDiffusivity()[0]
        rhog = self.mesh.get_rho(*self.pos)[0]

        #print(f'{D/rhog*drhodx = }')
        return np.array([D/rhog*drhodx, D/rhog*drhody, D/rhog*drhodz])


    def get_veff(self):
        """determine the effective velocity of the particle including
        diffusive effects from diffusivity and density gradients
        """
        veff = np.array(self.vel)
        #print(f'{veff = }')
        #print(f'{veff.shape = }')
        
        # vdiff = dD/dx
        vdiff = self.get_vdiff()
        veff += vdiff
        #print(f'{veff = }')
        #print(f'{veff.shape = }')

        # vrho = D/rho drho/dx
        vrho = self.get_vrho()
        veff += vrho
        #print(f'{veff = }')
        #print(f'{veff.shape = }')

        return veff
