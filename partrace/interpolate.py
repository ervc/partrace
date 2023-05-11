import numpy as np

def interp1d(arr,coords,x,axis=0,linear=False):
    """Linearly interpolate an axis in 1D along one of the axes.
    Parameters
    ----------
    arr : ndarray
        array to interpolate over
    coords : ndarray
        1D array of coordinates for the axis to be interpolated
        over
    x : float
        value at which to interpolate
    axis : int
        axis of array to interpolate
    linear : bool
        if the array is linearly spaced
    Returns
    -------
    float or ndarray
        interpolated value of arr at coordinate x
    """
    # if interpolating at a grid point, just return the value
    i, = np.where(np.isclose(coords,x))
    if i.size>0:
        return arr[i[0]]
    
    from math import floor,ceil,nan
    
    if coords[0] < coords[-1]:
        # increaseing coordinates
        if x < coords[0] or x > coords[-1]:
            return np.nan
    elif coords[0] > coords[-1]:
        # decreasing coordinates
        if x > coords[0] or x < coords[-1]:
            return np.nan
    
    nx = len(coords)
    if linear:
        assert np.all(np.diff(coords) > 0), "coords must be monotonically increasing"
        xi = coords[0]
        xf = coords[-1]
        dx = (xf-xi)/(nx-1)
        assert np.allclose(np.diff(coords),dx), "coords do not seem to be linearly spaced"
        # find the exact "index" where coords = x
        i = (x-xi)/dx
        ilo = int(floor(i))
        ihi = int(ceil(i))
    else:
        # find where coords is closest
        i = np.argmin(np.abs(coords-x))
        # this assumes increasing bounds... fine for now but I should 
        # fix this later
        if x > coords[i]:
            ilo = i
            ihi = min(nx-1,i+1) # catch edge case, if x>max(x), then ilo=ihi=i
        else:
            ilo = max(0,i-1) # catch edge case
            ihi = i
    xlo = coords[ilo]
    xhi = coords[ihi]
    ylo = arr[ilo]
    yhi = arr[ihi]
    if xlo == xhi:
        # in the case that xlo is the same as xhi then ylo should equal
        # yhi, so just return either value
        assert np.all(ylo==yhi)
        return ylo
    return ylo + (yhi-ylo)*(x-xlo)/(xhi-xlo)

def interp3d(arr,coords,pos):
    """Interpolate in 3D at x,y,z coordinate
    Parameters
    ----------
    arr : ndarray
        3D array to interpolate over, must have shape (ntheta,nr,nphi)
    coords : tuple
        tuple of 1D arrays of coords in phi, r, theta directions
    pos : tuple
        cartesian coordinates at place you want to interpolate
    Returns
    -------
    float
        interpolated value of arr at pos
    """
    x,y,z = pos
    phi = np.arctan2(y,x)
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    PHI,R,THETA = coords
    # interpolate in theta, then r, then phi
    # check for nans between interpolations
    arr2 = interp1d(arr,THETA,theta)
    if np.isnan(np.sum(arr2)):
        raise ValueError("interpolation in theta is out of bounds")

    arr1 = interp1d(arr2,R,r)
    if np.isnan(np.sum(arr1)):
        print('\n')
        print(f'{r:e}')
        print(R)
        raise ValueError("interpolation in r is out of bounds")

    # phi is out of PHI bounds, try adding to subtracting 2pi
    if phi < np.min(PHI):
        phi = phi + 2*np.pi
    elif phi > np.max(PHI):
        phi = phi - 2*np.pi
    arr0 =  interp1d(arr1,PHI,phi)
    if np.isnan(np.sum(arr0)):
        print(phi)
        print(PHI)
        raise ValueError("interpolation in phi is out of bounds")
    return arr0

def check_result(arr):
    if type(arr) == np.ndarray:
        return arr
    elif np.isnan(arr):
        return np.nan
    else:
        return 0