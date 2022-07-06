import python.constants as const
from python.mesh import Mesh
import matplotlib.pyplot as plt
import numpy as np

FARGOOUT = 'exampleout/fargo'


if __name__ == '__main__':
	mesh = Mesh(FARGOOUT)

	mesh.read_state('gasdens',-1)
	im = mesh.plot_state('gasdens')

	plt.show()



		