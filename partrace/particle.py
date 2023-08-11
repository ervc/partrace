"""
Particle class for tracer particles that move through the Mesh.
"""


import numpy as np
from .constants import *
from .interpolate import interp3d

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

        self.subgrid = {}
        self.init_subgrid()
        self.update_subgrid()

        self.vel = self.get_vel0()
        self.vel0 = np.array(self.vel)

        if not self.mesh.quiet:
            print(f'particle created at {self.pos}')

    def init_subgrid(self):
        self.iwidth = self.mesh.nx+2 #512+1
        self.jwidth = 48+1
        self.kwidth = self.mesh.nz*2

        self.subgridshape = (self.kwidth,self.jwidth,self.iwidth)
        self.subgridsize = self.kwidth*self.jwidth*self.iwidth

        self.subxcenters = np.zeros(self.iwidth)
        self.subycenters = np.zeros(self.jwidth)
        self.subzcenters = np.zeros(self.kwidth)

        for state in ['gasdens','gasvx','gasvy','gasvz','partdiff']:
            self.subgrid[state] = np.zeros(self.subgridshape)

    def init_fullgrid(self):
        """This takes a huge amount of memory, do not use unless you 
        have a ton of RAM.
        """
        self.iwidth = self.mesh.nx+2 
        self.jwidth = self.mesh.ny
        self.kwidth = self.mesh.nz*2

        self.subgridshape = (self.kwidth,self.jwidth,self.iwidth)
        self.subgridsize = self.kwidth*self.jwidth*self.iwidth

        self.subxcenters = np.zeros(self.iwidth)
        self.subycenters = np.zeros(self.jwidth)
        self.subzcenters = np.zeros(self.kwidth)

        # xcenters plus the edges
        self.subxcenters = np.zeros(self.iwidth)
        self.subxcenters[1:-1] = self.mesh.xcenters
        self.subxcenters[0] = self.mesh.xcenters[-1]-2*np.pi
        self.subxcenters[-1] = self.mesh.xcenters[0]+2*np.pi

        # ycenters are the same
        self.subycenters = self.mesh.ycenters

        # zcenters are copied and flipped
        nz = self.mesh.nz
        self.subzcenters[:nz] = self.mesh.zcenters
        self.subzcenters[nz:] = np.pi-self.mesh.zcenters[::-1]

        for state in ['gasdens','gasvx','gasvy','gasvz']:
            self.subgrid[state] = np.zeros(self.subgridshape)
            fullmesh = self.mesh.read_state(state)
            # top half of the subgrid (which is now more like a super grid)
            self.subgrid[state][:nz,:,1:-1] = fullmesh
            # repeat the azimuthal edges
            self.subgrid[state][:nz,:,0] = fullmesh[:,:,-1]
            self.subgrid[state][:nz,:,-1] = fullmesh[:,:,0]
            if state == 'gasvz':
                self.subgrid[state][nz:] = -self.subgrid[state][nz-1::-1]
            else:
                self.subgrid[state][nz:] = self.subgrid[state][nz-1::-1]

        # get the gradient grids
        self.subgrid['gradrho'] = self.get_gradrho_grid()
        self.subgrid['gradpartdiff'] = self.get_gradpartdiff_grid()
        self.subgrid['gasvel'] = self.get_gasvel_grid()

        # get subgrid indices
        self.subi, self.subj, self.subk = self.get_subgrid_index(*self.pos)

    def get_vel0(self):
        x,y,z = self.pos0
        r = np.sqrt(x*x + y*y + z*z)
        phi = np.arctan2(y,x)
        omega = self.mesh.get_Omega(x,y,z)
        vkep = r*omega
        stokes = self.get_stokes()
        tstop = stokes/omega
        # get (non-rotating) gas vphi at closest cell point
        vphi_gas = self.subgrid['gasvx'][self.subk,self.subj,self.subi]
        vphi_gas = vphi_gas + r*float(self.mesh.variables['OMEGAFRAME'])  
        eta = 1-(vphi_gas/vkep)**2
        # armitage notes (140)
        vr = -eta/(stokes + 1/stokes) * vkep
        # armitage notes (136)
        vphi = vphi_gas - 0.5*tstop*vr*vkep/r
        # get rotating vphi
        vphi = vphi - r*float(self.mesh.variables['OMEGAFRAME'])

        # x = r*np.cos(phi)
        # xdot = rdot*np.cos(phi) - r*phidot*np.sin(phi)
        # y = r*np.sin(phi)
        # ydot = rdot*np.sin(phi) + r*phidot*np.cos(phi)
        # note: vphi = r*phidot
        vx = vr*np.cos(phi) - vphi*np.sin(phi)
        vy = vr*np.sin(phi) + vphi*np.cos(phi)
        vz = 0
        return np.array([vx,vy,vz])

    def update_position(self,x,y,z):
        """update the position of the particle, and 
        update subgrid if necessary"""
        self.pos = np.array([x,y,z])
        self.subi,self.subj,self.subk = self.get_subgrid_index(x,y,z)
        # if not already at the edge then check if we need to regrid
        i,j,k = self.mesh.get_cell_index(x,y,z)
        jhw = int((self.jwidth-1)/2)
        jlo,jhi = j-jhw, j+jhw+1
        if jlo == 0 or jhi == self.mesh.ny:
            return None
        elif (self.subj > 3/4*self.jwidth or self.subj < 1/4*self.jwidth):
            i,j,k = self.mesh.get_cell_index(x,y,z)
            jhw = int((self.jwidth-1)/2)
            jlo,jhi = j-jhw, j+jhw+1
            if jlo == 0 or jhi == self.mesh.ny:
                return None
            return self.update_subgrid()


    def update_velocity(self,vx,vy,vz):
        """update the velocity of the particle"""
        self.vel = np.array([vx,vy,vz])

    def get_subgrid_index(self,x,y,z):
        """return the cell with center location closest to the given
        cartesian location
        """
        phi = np.arctan2(y,x)
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.arccos(z/r)

        # # use regular az spacing to our advantage
        # dphi = (self.subxcenters[-1]-self.subxcenters[0])/(self.iwidth-1)
        # # the center of each cell is at i*dphi + phimin
        # # we want to minimize |phi - [i*dphi + phimin]| or
        # # closest i to phi = i*dphi+phimin
        # try:
        #     i = int(np.round((phi-self.subxcenters[0])/dphi))
        # except ValueError:
        #     print('\n',x,y,z)
        #     print(phi)
        #     print(self.subxcenters[0])
        #     print(dphi)
        #     raise ValueError
        i = np.argmin(np.abs(phi-self.subxcenters))


        # loop through r since this may or may not be linearly spaced
        # by default this will be log spaced
        # Loop through to until we start moving away from the value
        j = np.argmin(np.abs(r-self.subycenters))

        # once again, spacing may not be regular. Assume that z>0 and
        # theta < PI/2. Exceptions should be handled before the call
        # here. For example a particle below the midplane will need the
        # same gas density as +z, but gasvz should be negative.
        k = np.argmin(np.abs(theta-self.subzcenters))

        return i,j,k

    def update_subgrid(self):
        # read in new values for a subgrid centered on the particle
        x,y,z = self.pos
        ZNEG = False
        if z<0:
            z=-z
            ZNEG = True
        i,j,k = self.mesh.get_cell_index(x,y,z)
        # ihw = int((self.iwidth-1)/2)
        jhw = int((self.jwidth-1)/2)
        # ilo,ihi = i-ihw, i+ihw+1
        jlo,jhi = j-jhw, j+jhw+1

        # if gradrho is already defined then this is not the first time
        # update_subgrid is being called.
        # if jlo is exactly 0 or jhi is exactly ny, then no need to
        # regrid, as it has probably already been regridded here.
        if 'gradrho' in self.subgrid:
            if jlo == 0 or jhi == self.mesh.ny:
                return None

        # # get x centers first
        # # check bounds
        # if ilo < 0:
        #     # periodic at lower edge
        #     l = list(range(self.mesh.nx+ilo,self.mesh.nx))+list(range(0,ihi))
        #     islice = np.array(l)
        # elif ihi >= self.mesh.nx:
        #     l = list(range(ilo,self.mesh.nx))+list(range(0,ihi-self.mesh.nx))
        #     islice = np.array(l)
        # else:
        #     islice = np.arange(ilo,ihi)
        # # use array slicing to get periodic azimuthal values
        # self.subxcenters = self.mesh.xcenters[islice]
        # # if islice is less than ilo, this means some of the indices
        # # have wrapped around from the upper edge (ihi >= self.mesh.nx)
        # self.subxcenters[islice<ilo] += 2*np.pi
        # self.subxcenters[islice>ihi] -= 2*np.pi
        self.subxcenters = np.zeros(self.iwidth)
        self.subxcenters[1:-1] = self.mesh.xcenters
        self.subxcenters[0] = self.mesh.xcenters[-1]-2*np.pi
        self.subxcenters[-1] = self.mesh.xcenters[0]+2*np.pi
        # islice is indexing array, where we index the last cell,
        # then all the cells in order, then the first cell to make the
        # boundaries periodic
        islice = np.array([-1] + list(range(self.mesh.nx)) + [0])

        # # get y centers next
        # if jlo < 0:
        #     # just pad the lower edge of the grid with zeros
        #     # particle should be caught before it reaches this point
        #     # in the integration step
        #     l = [0]*-jlo + list(range(0,jhi))
        #     jslice = np.array(l)
        # elif jhi > self.mesh.ny:
        #     l = list(range(jlo,self.mesh.ny)) + [self.mesh.ny-1]*(jhi-self.mesh.ny)
        #     jslice = np.array(l)
        # else:
        #     jslice = np.arange(jlo,jhi)
        # self.subycenters = self.mesh.ycenters[jslice]
        # new get y centers
        ny = self.mesh.ny
        if jlo < 0:
            # if lower edge of mesh would be inside of inner edge
            # then just set it to be at the inner edge instead
            jslice = np.arange(0,self.jwidth)
        elif jhi > ny:
            # at the upper edge, set upper limit to be at mesh.ny
            jslice = np.arange(ny-self.jwidth,ny)
        else:
            # otherwise just a normal slice centered on the particle
            jslice = np.arange(jlo,jhi)
        self.subycenters = self.mesh.ycenters[jslice]

        # get z centers last
        nz = self.mesh.nz
        self.subzcenters[:nz] = self.mesh.zcenters
        self.subzcenters[nz:] = np.pi-self.mesh.zcenters[::-1]

        # use [:,None] and [None,:] to make jslice and islice 2D so the
        # slice can be broadcast
        for state in ['gasdens','gasvx','gasvy','gasvz']:
            self.subgrid[state][:nz] = self.mesh.read_state(state)[:,jslice[:,None],islice[None,:]]
            if state == 'gasvz':
                self.subgrid[state][nz:] = -self.subgrid[state][nz-1::-1]
            else:
                self.subgrid[state][nz:] = self.subgrid[state][nz-1::-1]

        # get the gradient grids
        self.subgrid['gradrho'] = self.get_gradrho_grid()
        self.subgrid['gradpartdiff'] = self.get_gradpartdiff_grid()
        self.subgrid['gasvel'] = self.get_gasvel_grid()

        # cell index in subgrid
        self.subi = i
        self.subj = jhw
        self.subk = k
        if ZNEG:
            self.subk = 2*nz-k-1

        return None


    # RHO FUNCTIONS
    def get_rho(self):
        """Return the gas density at the location of the particle"""
        return self.get_rho_at(*self.pos)

    def get_rho_at(self,x,y,z):
        """return the gas density at a location near the particle
        (within the submesh)
        """
        return interp3d(self.subgrid['gasdens'],
            (self.subxcenters,self.subycenters,self.subzcenters),(x,y,z))

    def get_rho_grid(self):
        return np.array(self.subgrid['gasdens'])

    def get_gradrho_grid(self):
        rho = self.subgrid['gasdens']
        p = self.subxcenters
        r = self.subycenters
        t = self.subzcenters

        # need phi,r,theta on a grid
        theta,r,phi = np.meshgrid(t,r,p,indexing='ij')

        # drhodphi
        drhodphi = np.zeros_like(rho)
        dphi = (self.subxcenters[-1]-self.subxcenters[0])/(self.iwidth-1)
        drhodphi[:,:,1:-1] = (rho[:,:,2:]-rho[:,:,:-2])/(2*dphi)
        drhodphi[:,:,0] = (rho[:,:,1]-rho[:,:,0])/(dphi)
        drhodphi[:,:,-1] = (rho[:,:,-1]-rho[:,:,-2])/(dphi)

        # drhodr
        drhodr = np.zeros_like(rho)
        drhodr[:,1:-1,:] = (rho[:,2:,:]-rho[:,:-2,:])/(r[:,2:,:]-r[:,:-2,:])
        drhodr[:,0,:] = (rho[:,1,:]-rho[:,0,:])/(r[:,1,:]-r[:,0,:])
        drhodr[:,-1,:] = (rho[:,-1,:]-rho[:,-2,:])/(r[:,-1,:]-r[:,-2,:])

        # drhodtheta
        drhodtheta = np.zeros_like(rho)
        drhodtheta[1:-1] = (rho[2:]-rho[:-2])/(theta[2:]-theta[:-2])
        drhodtheta[0] = (rho[1]-rho[0])/(theta[1]-theta[0])
        drhodtheta[-1] = (rho[-1]-rho[-2])/(theta[-1]-theta[-2])

        # chain rule
        dphidx = -np.sin(phi)/r/np.sin(theta)
        dphidy =  np.cos(phi)/r/np.sin(theta)
        dphidz =  0
        drdx = np.cos(phi)*np.sin(theta)
        drdy = np.sin(phi)*np.sin(theta)
        drdz = np.cos(theta)
        dthetadx =  np.cos(phi)*np.cos(theta)/r
        dthetady =  np.sin(phi)*np.cos(theta)/r
        dthetadz = -np.sin(theta)/r

        drhodx = drhodphi*dphidx + drhodr*drdx + drhodtheta*dthetadx
        drhody = drhodphi*dphidy + drhodr*drdy + drhodtheta*dthetady
        drhodz = drhodphi*dphidz + drhodr*drdz + drhodtheta*dthetadz

        arrs = (drhodx,drhody,drhodz)
        return np.stack(arrs,axis=-1)

    def get_gradrho(self):
        return self.get_gradrho_at(*self.pos)

    def get_gradrho_at(self,x,y,z):
        return interp3d(self.subgrid['gradrho'],
            (self.subxcenters,self.subycenters,self.subzcenters),(x,y,z))



    # STOKES FUNCTIONS
    def get_stokes(self):
        """get the stokes number at the current location"""
        return self.get_stokes_at(*self.pos)

    def get_stokes_at(self,x,y,z):
        """Return the stokes number at a location near the particle"""
        rho_g = self.get_rho_at(x,y,z)
        cs = self.mesh.get_soundspeed(x,y,z)
        om = self.mesh.get_Omega(x,y,z)
        return self.a*self.rho_s/(rho_g*cs)*om

    def get_stokes_grid(self):
        phi   = self.subxcenters
        r     = self.subycenters
        theta = self.subzcenters
        tt,rr,pp = np.meshgrid(theta,r,phi,indexing='ij')
        xx = rr*np.cos(pp)*np.sin(tt)
        yy = rr*np.sin(pp)*np.sin(tt)
        zz = rr*np.cos(tt)
        rho_g = self.subgrid['gasdens']
        cs = self.mesh.get_soundspeed(xx,yy,zz)
        om = self.mesh.get_Omega(xx,yy,zz)
        return self.a*self.rho_s/(rho_g*cs)*om

    def get_stopping_time(self):
        om = self.mesh.get_Omega(*self.pos)
        return self.get_stokes()/om

    def get_stopping_time_at(self,x,y,z):
        om = self.mesh.get_Omega(x,y,z)
        return self.get_stokes_at(x,y,z)/om


    # DIFFUSIVITY FUNCTIONS
    def get_partdiff(self):
        """Return the particle diffusivity at current location"""
        return self.get_partdiff_at(*self.pos)

    def get_partdiff_at(self,x,y,z):
        gasdiff = self.mesh.get_diffusivity(x,y,z)
        St = self.get_stokes_at(x,y,z)
        return gasdiff/(1+St*St)

    def get_partdiff_grid(self):
        phi   = self.subxcenters
        r     = self.subycenters
        theta = self.subzcenters
        tt,rr,pp = np.meshgrid(theta,r,phi,indexing='ij')
        xx = rr*np.cos(pp)*np.sin(tt)
        yy = rr*np.sin(pp)*np.sin(tt)
        zz = rr*np.cos(tt)
        gasdiff = self.mesh.get_diffusivity(xx,yy,zz)
        St = self.get_stokes_grid()
        return gasdiff/(1+St*St)

    def get_gradpartdiff_grid(self):
        D = self.get_partdiff_grid()
        p = self.subxcenters
        r = self.subycenters
        t = self.subzcenters

        # need phi,r,theta on a grid
        theta,r,phi = np.meshgrid(t,r,p,indexing='ij')

        # dDdphi
        dDdphi = np.zeros_like(D)
        dphi = (self.subxcenters[-1]-self.subxcenters[0])/(self.iwidth-1)
        dDdphi[:,:,1:-1] = (D[:,:,2:]-D[:,:,:-2])/(2*dphi)
        dDdphi[:,:,0] = (D[:,:,1]-D[:,:,0])/(dphi)
        dDdphi[:,:,-1] = (D[:,:,-1]-D[:,:,-2])/(dphi)

        # dDdr
        dDdr = np.zeros_like(D)
        dDdr[:,1:-1,:] = (D[:,2:,:]-D[:,:-2,:])/(r[:,2:,:]-r[:,:-2,:])
        dDdr[:,0,:] = (D[:,1,:]-D[:,0,:])/(r[:,1,:]-r[:,0,:])
        dDdr[:,-1,:] = (D[:,-1,:]-D[:,-2,:])/(r[:,-1,:]-r[:,-2,:])

        # dDdtheta
        dDdtheta = np.zeros_like(D)
        dDdtheta[1:-1] = (D[2:]-D[:-2])/(theta[2:]-theta[:-2])
        dDdtheta[0] = (D[1]-D[0])/(theta[1]-theta[0])
        dDdtheta[-1] = (D[-1]-D[-2])/(theta[-1]-theta[-2])

        # chain rule
        dphidx = -np.sin(phi)/r/np.sin(theta)
        dphidy =  np.cos(phi)/r/np.sin(theta)
        dphidz =  0
        drdx = np.cos(phi)*np.sin(theta)
        drdy = np.sin(phi)*np.sin(theta)
        drdz = np.cos(theta)
        dthetadx =  np.cos(phi)*np.cos(theta)/r
        dthetady =  np.sin(phi)*np.cos(theta)/r
        dthetadz = -np.sin(theta)/r

        dDdx = dDdphi*dphidx + dDdr*drdx + dDdtheta*dthetadx
        dDdy = dDdphi*dphidy + dDdr*drdy + dDdtheta*dthetady
        dDdz = dDdphi*dphidz + dDdr*drdz + dDdtheta*dthetadz

        arrs = (dDdx,dDdy,dDdz)
        return np.stack(arrs,axis=-1)

    def get_gradpartdiff(self):
        return self.get_gradpartdiff_at(*self.pos)

    def get_gradpartdiff_at(self,x,y,z):
        return interp3d(self.subgrid['gradpartdiff'],
            (self.subxcenters,self.subycenters,self.subzcenters),(x,y,z))


    # GASVEL FUNCTIONS
    def get_gasvel(self):
        return self.get_gasvel_at(*self.pos)

    def get_gasvel_at(self,x,y,z):
        return interp3d(self.subgrid['gasvel'],
            (self.subxcenters,self.subycenters,self.subzcenters),(x,y,z))

    def get_gasvel_grid(self):
        vphi   = self.subgrid['gasvx']
        vr     = self.subgrid['gasvy']
        vtheta = self.subgrid['gasvz']
        phi   = self.subxcenters
        r     = self.subycenters
        theta = self.subzcenters
        tt,rr,pp = np.meshgrid(theta,r,phi,indexing='ij')
        # use helper function already written in Mesh
        vx,vy,vz = self.mesh._vel_sphere2cart(pp,rr,tt,vphi,vr,vtheta)
        arrs = (vx,vy,vz)
        return np.stack(arrs,axis=-1)



        

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
