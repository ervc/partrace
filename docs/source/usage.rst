Usage
=====

This is Partrace! See the :doc:`modules` for a quick overview of the different modules and classes available to use.

Installation
------------

It is recommended to first create a virtual enviornment using python

.. code-block:: console

    $ python -m venv ptenv
    $ source ptenv/bin/activate

Or anaconda enviornment with dependencies

.. code-block:: console

    $ conda create --name ptenv scipy
    $ conda activate ptenv

Copy the base partrace directory from github. Change directories to the base directory and install locally.

.. code-block:: console

    (ptenv) $ pip install .


Quick Start
-----------

In python, import standard libraries and partrace and define a path variable to your FARGO3D output.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    import partrace as pt
    fargodir = 'exampleout/fargo_rescale' #for example

Next, create a :py:class:`Mesh <partrace.mesh.Mesh>` from the fargo output, then create a :py:class:`Planet <partrace.planet.Planet>` and :py:class:`Particle <partrace.particle.Particle>` within the :py:class:`Mesh <partrace.mesh.Mesh>`. Helpful constants are stored in the :py:mod:`partrace.constants` module

.. code-block:: python

    mesh = pt.create_mesh(fargodir)

    planet = pt.create_planet(mesh)
    
    x0 = 10*pt.const.AU
    y0 = 0
    z0 = 0
    particle = pt.create_particle(mesh,x0,y0,z0,a=100)

Finally, set up the solver parameters and integrate the particle!

.. code-block:: python

    t0 = 0
    tf = 1e2*const.YR
    max_step = 1/50*pt.const.TWOPI/mesh.get_Omega(minr,0,0)

    # get the inner edge of mesh and maximum velocity for atol and rtol
    minr = mesh.yedges.min()
    maxv = np.nanmax(np.abs(mesh.state['gasvx']))
    # here we define atol to be a ndarray to define different atols for
    # the position and velocity
    atol = np.zeros(6)
    atol[:3] += 1e-3*minr  # xtol is within 1e-3 of smallest r
    atol[3:] += 1e-3*maxv  # vtol is within 1e-3 of largest velocity
    rtol = 1e-6

    sol = integrate(t0,tf,p,planet,savefile='quickstart_out.npz',
                    max_step=max_step,atol=atol,rtol=rtol)

You can plot the results by getting the :py:attr:`history <partrace.integrate.Solver.history>` attribute from the :py:class:`solver <partrace.integrate.Solver>`.

.. code-block:: python

    # default output is (nout,6), use .T to unpack into more useful variables
    x,y,z,vx,vy,vz = sol.history.T
    times = sol.times

Or if the results are saved, extract the ndarrays from the npz file

.. code-block:: python

    ptout = np.load('quickstart_out.npz')
    x,y,z,vx,vy,vz = ptout['history'].T
    times = ptout['times']

If everything ran correctly, you should see the particle drifting inward!

.. code-block:: python

    r = np.sqrt(x*x + y*y)
    fig,ax = plt.subplots()
    ax.plot(times,r)
    ax.set(xlabel='time [sec]',ylabel='radius [r]')
    plt.show()


    

