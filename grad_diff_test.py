import matplotlib.pyplot as plt
import numpy as np

import python as pt
import python.constants as const

FARGOOUT = './exampleout/fargo_rescale'

def stokes_number(mesh,x,y,z=0,a=0.01,rho_s=2.):
    om = mesh.get_Omega(x,y,z)
    rho_g = mesh.get_rho(x,y,z)
    cs = mesh.get_soundspeed(x,y,z)
    return a*rho_s/(rho_g*cs)*om

def get_dStdx(mesh,x,y,z=0,a=0.01,rho_s=2.):
    r = np.sqrt(x*x + y*y)
    dx = 0.01*r
    St0 = stokes_number(mesh,x,y,z,a,rho_s)
    Stx = stokes_number(mesh,x+dx,y,z,a,rho_s)
    dStdx = (Stx-St0)/dx
    return dStdx

def get_dDdx(mesh,x,y,z=0,a=0.01,rho_s=2.):
    St = stokes_number(mesh,x,y,z,a,rho_s)
    Dg = mesh.get_diffusivity(x,y,z)
    #only worry about dx for now
    dDgdx,*_ = mesh.get_diff_grad(x,y,z)
    dStdx = get_dStdx(mesh,x,y,z,a,rho_s)
    # d/dx Dg/(1+St^2) = dDg/dx / (1+St^2) + -Dg/(1+St^2)^2*2St*dSt/dx
    p1 = dDgdx/(1+St**2)
    p2 = -Dg/((1+St**2)**(2))*2*St*dStdx
    return p1+p2

def plot_arr(ax,x,y,arr,label,vmin=None,vmax=None):
    if vmin is None:
        vmin = -np.nanmax(np.abs(arr))
    if vmax is None:
        vmax = np.nanmax(np.abs(arr))
    im = ax.pcolormesh(x,y,arr,vmin=vmin,vmax=vmax,cmap='coolwarm')
    cb = plt.colorbar(im,ax=ax,location='top',label=label)

def main(fargodir):
    mesh = pt.create_mesh(fargodir)
    
    minr = mesh.yedges.min()
    maxr = mesh.yedges.max()

    X = np.linspace(-maxr,maxr,250)
    Y = np.linspace(-maxr,maxr,250)

    xx,yy = np.meshgrid(X,Y)

    dDgas,*_ = mesh.get_diff_grad(xx,yy)

    St = stokes_number(mesh,xx,yy)
    dD_approx = dDgas/(1+St**2)

    dStdx = get_dStdx(mesh,xx,yy)

    dD_exact = get_dDdx(mesh,xx,yy)

    vmin = min(dDgas.min(),dD_approx.min())
    vmax = max(dDgas.max(),dD_approx.max())
    cmap = 'coolwarm'
    fig,axs = plt.subplots(2,2,sharey=True,sharex=True)
    ax = axs[0,0]
    plot_arr(ax,xx,yy,dD_approx,'dD/dx = dDgas/dx / (1+St^2)')
    ax = axs[0,1]
    plot_arr(ax,xx,yy,dStdx,'dSt/dx')
    ax = axs[1,0]
    plot_arr(ax,xx,yy,dD_exact,'dD/dx = full derivative')
    ax = axs[1,1]
    arr = (dD_approx-dD_exact)/dD_exact
    plot_arr(ax,xx,yy,arr,'abs err')
    for ax in axs.flatten():
        ax.set(aspect='equal')

    plt.show()

    fig,axs = plt.subplots(1,3)
    ax = axs[0]
    im = ax.pcolormesh(xx,yy,np.log10(mesh.get_rho(xx,yy)))
    cb = plt.colorbar(im,ax=ax)
    ax.set_title('log rho')
    ax = axs[1]
    im = ax.pcolormesh(xx,yy,
        np.log10(mesh.get_state_from_cart('gasenergy',xx,yy)))
    cb = plt.colorbar(im,ax=ax)
    ax.set_title('log cs')
    ax=axs[2]
    H = mesh.get_scaleheight(xx,yy)
    Om = mesh.get_Omega(xx,yy)
    cs = np.where(np.isnan(mesh.get_rho(xx,yy)),np.nan,H*Om)
    im = ax.pcolormesh(xx,yy,np.log10(cs))
    cb = plt.colorbar(im,ax=ax)
    ax.set_title('log H/Omega')
    for ax in axs:
        ax.set(aspect='equal')

    plt.show()

if __name__ == "__main__":
    main(FARGOOUT)

