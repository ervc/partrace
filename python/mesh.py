import numpy as np
import matplotlib.pyplot as plt

class Mesh():
    """Fargo domain mesh. Contains the domain, density, 
    and velocity mesh
    """
    def __init__(self, fargodir):
        self.fargodir = fargodir
        self.ndim = 3
        self.get_domain()
        self.get_centers()
        self.state = {}

    def get_domain(self,ghostcells = True):
        # default number of ghost cells out, check fargo distribution
        NGHOST = 3 

        # x edges = azimuth, no ghost cells
        self.xedges = []
        with open(self.fargodir+'/domain_x.dat','r') as f:
            for line in f:
                self.xedges.append(float(line))
        self.nx = len(self.xedges)-1

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
            self.nz = 0
        else:
            if ghostcells:
                self.zedges = list(allzedges[NGHOST:-NGHOST])
            else:
                self.zedges = list(allzedges)
            self.nz = len(self.zedges)-1

    def get_centers(self):
        self.xcenters = list([(self.xedges[i]
                                + self.xedges[i+1])/2 for i in range(self.nx)])
        self.ycenters = list([(self.yedges[i]
                                + self.yedges[i+1])/2 for i in range(self.ny)])
        self.zcenters = list([(self.zedges[i]
                                + self.zedges[i+1])/2 for i in range(self.nz)])

    def read_state(self,state,n=-1):
        # readin the grid data for output 'state' at output number n. 
        # Default n=-1 gives last ouptut
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

        statefile = self.fargodir+f'/{state}{n}.dat'

        if self.ndim == 3:
            state_arr = np.fromfile(statefile).reshape(self.nx,self.ny,self.nz)
        else:
            state_arr = np.fromfile(statefile).reshape(self.ny,self.nx)
        self.state[state] = state_arr
        return state_arr

    def plot_state(self,state,ax=None,log=True,*args,**kwargs):
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

        if self.ndim == 3:
            im = ax.pcolormesh(self.xedges,self.yedges,arr[:,:,-1],
                                *args,**kwargs)
        else:
            im = ax.pcolormesh(self.xedges,self.yedges,arr,*args,**kwargs)
        cb = plt.colorbar(im,ax=ax,label=label)

        return im


