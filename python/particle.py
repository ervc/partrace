import numpy as np
from .constants import *
from . import disk

INCLUDE_DRAG = True
INCLUDE_DIFFUSION = True
DKD = True
DEBUG = False

class Particle(object):
	"""a particle that can move through the disk"""
	def __init__(self,x,y,z,a=0.01,rho_s=2):
		self.a = a
		self.rho_s = rho_s

		self.pos = np.array([x,y,z])
		self.history = [np.array(self.pos)]
		r = np.hypot(x,y)
		phi = np.arctan2(y,x)
		vk = r*disk.get_Omegak(r,z)
		vg = disk.get_velocity(r,z)
		self.vel = np.array([-vg*np.sin(phi),vg*np.cos(phi),0])

	def get_stokes(self):
		"""get the stokes number at the current location"""
		x,y,z = self.pos
		r = np.hypot(x,y)
		om = disk.get_Omegak(r,z)
		rho_g = disk.get_rho(r,z)
		cs = disk.get_soundspeed(r)
		return self.a*self.rho_s/(rho_g*cs)*om

	def get_dragAccel(self):
		"""find the drag acceleration vector"""
		x,y,z = self.pos
		r = np.hypot(x,y)
		phi = np.arctan2(y,x)


		vgphi = disk.get_velocity(r,z)
		vgas = np.array([-vgphi*np.sin(phi),vgphi*np.cos(phi),0])
		
		vpar = np.array(self.vel)
		
		vtil = vgas - vpar
		
		if DEBUG:
			print('keplarian velocity')
			print(r*disk.get_Omegak(r,z))
			print('gas velocity')
			print(vgphi)
			print('gas vel')
			print(vgas)
			print('particle vel')
			print(vpar)
			print('difference')
			print(vtil)

		# om = disk.get_Omegak(r,z)
		# st = self.get_stokes()
		rho_g = disk.get_rho(r,z)
		cs = disk.get_soundspeed(r)
		if DEBUG:
			print('acceleration')
			print(rho_g*cs/self.a/self.rho_s*vtil)
		return rho_g*cs/self.a/self.rho_s*vtil

	def get_gravAccel(self):
		"""find the acceleration due to the star gravity"""
		x,y,z = self.pos
		dstar = np.sqrt(x*x+y*y+z*z) # distance to star
		dstar_vec = np.array([x,y,z])
		dstar_hat = dstar_vec/dstar
		# print('grav accel')
		# print(-GM/dstar**2 * dstar_hat)
		return -GM/dstar**2 * dstar_hat

	def update_velocity(self,dt):
		"""update the velocity vector due to (drag) and grav"""
		oldvel = np.array(self.vel)
		oldvel += self.get_gravAccel()*dt
		if INCLUDE_DRAG:
			oldvel += self.get_dragAccel()*dt
		self.vel = oldvel

	def get_vdiff(self):
		return 0

	def update_position(self,dt):
		"""update position based on current velocity"""
		x,y,z = self.pos
		r = np.hypot(x,y)
		st = self.get_stokes()

		veff = np.array(self.vel)
		if INCLUDE_DIFFUSION:
			vdiff = self.get_vdiff()
			veff += vdiff
		self.pos += veff*dt
		if INCLUDE_DIFFUSION:
			R = np.random.uniform(-1.,1.)
			xi = 1./3.
			D = disk.get_diffusivity(r,z)/(1+st**2)
			self.pos += R*(2/xi*D*dt)**(1/2)

	def determine_dt(self):
		"""determine dt based on 1/50th of period"""
		x,y,z = self.pos
		r = np.hypot(x,y)
		om = disk.get_Omegak(r,z)
		return (1/5000)*(TWOPI/om)

	def take_step(self):
		dt = self.determine_dt()
		if DKD:
			self.update_position(0.5*dt)
			self.update_velocity(dt)
			self.update_position(0.5*dt)
		elif KDK:
			self.update_velocity(0.5*dt)
			self.update_position(dt)
			self.update_velocity(0.5*dt)
		self.history.append(np.array(self.pos))
