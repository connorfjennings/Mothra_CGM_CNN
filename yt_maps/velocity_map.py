import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from matplotlib.colors import TwoSlopeNorm

# ------------------ user inputs ------------------
basepath = "/gpfs/gibbs/pi/nagai/IllustrisTNG_data/TNG50-1/output"   # e.g. ".../TNG50-1/output"
outpath = "./TNG50_2D_maps/"
snap     = 99                           # z = 0
halo_id  = 20                            # FoF halo id
fov_kpc  = 200.0                        # field of view (square) in kpc
npix     = 64                           # pixel grid for the vectors (keep modest for speed)
mass_floor = 1e6                        # Msun: ignore pixels with less total gas mass than this
h = 0.6774                              # TNG Planck15 h
# -------------------------------------------------

# ---------- helpers ----------
def periodic_shift(arr, box):
    return (arr + 0.5*box) % box - 0.5*box


for halo_id in np.arange(1,500):


    # ---------- load header/box for units ----------
    head = il.groupcat.loadHeader(basepath, snap)
    box_kpc = head["BoxSize"] / h  # ckpc/h -> kpc (z=0)
    
    # ---------- FoF center & bulk velocity ----------
    grp = il.groupcat.loadSingle(basepath, snap, haloID=halo_id)
    center_kpc = grp["GroupPos"] / h            # kpc
    v_bulk     = grp["GroupVel"]                # km/s (peculiar)
    
    # ---------- load gas bound to this FoF ----------
    gas = il.snapshot.loadHalo(
        basepath, snap, halo_id, partType='gas',
        fields=['Coordinates','Masses','Velocities']
    )
    
    coords = gas['Coordinates'] / h             # kpc (z=0)
    masses = gas['Masses'] * (1e10 / h)         # Msun
    vel    = gas['Velocities'] - v_bulk         # km/s, subtract bulk to center the field
    
    # recenter coords w/ periodic fix so halo is centered
    xyz = coords - center_kpc
    xyz = periodic_shift(xyz, box_kpc)
    
    # project plane-of-sky axes
    x, y = xyz[:,0], xyz[:,1]
    vx, vy, vz = vel[:,0], vel[:,1], vel[:,2]
    
    # ---------- bin to a pixel grid (CIC-style) ----------
    half = 0.5 * fov_kpc
    edges = np.linspace(-half, half, npix+1)
    
    # mass in each pixel
    M = np.histogram2d(x, y, bins=[edges, edges], weights=masses)[0]
    
    # mass-weighted velocity components in each pixel
    Mx  = np.histogram2d(x, y, bins=[edges, edges], weights=masses*vx)[0]
    My  = np.histogram2d(x, y, bins=[edges, edges], weights=masses*vy)[0]
    Mz  = np.histogram2d(x, y, bins=[edges, edges], weights=masses*vz)[0]
    
    # safe division (ignore empty pixels)
    with np.errstate(invalid='ignore', divide='ignore'):
        vx_mean = np.divide(Mx, M, where=M>0)
        vy_mean = np.divide(My, M, where=M>0)
        vz_mean = np.divide(Mz, M, where=M>0)
    
    # mask low-mass pixels (optional)
    mask = (M > mass_floor)
    vx_mean[~mask] = 0.0
    vy_mean[~mask] = 0.0
    vz_mean[~mask] = 0.0
    
    v_mean = np.stack((vx_mean,vy_mean,vz_mean),axis=-1)
    np.save(outpath+f'snap{snap}halo{halo_id}_vmap.npy',v_mean)
    
    
    # grid of pixel centers for quiver
    xc = 0.5*(edges[:-1]+edges[1:])
    yc = xc.copy()
    XX, YY = np.meshgrid(xc, yc, indexing='xy')
    
    # ---------- build direction-only vectors ----------
    speed_xy = np.hypot(vx_mean, vy_mean)
    # avoid divide-by-zero
    Ux = np.zeros_like(vx_mean)
    Uy = np.zeros_like(vy_mean)
    nz = speed_xy > 0
    Ux[nz] = vx_mean[nz] / speed_xy[nz]
    Uy[nz] = vy_mean[nz] / speed_xy[nz]
    
    # make short line segments (direction only). Scale controls visual length in kpc.
    seg_len = (edges[1]-edges[0]) * 0.7   # ~70% of a pixel
    Ux_plot = Ux * seg_len
    Uy_plot = Uy * seg_len
    
    # color by mass-weighted LOS velocity, center colormap at 0
    vabs = np.nanmax(np.abs(vz_mean))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=+vabs)
    
    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(7,6))
    # optional: faint surface-density underlay for context
    Sigma = M / ((edges[1]-edges[0])**2)  # Msun/kpc^2
    ax.imshow(
        np.log10(Sigma + 1e-6).T, origin='lower',
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        alpha=0.3, cmap='gray'
    )
    
    Q = ax.quiver(
        XX, YY, Ux_plot.T, Uy_plot.T,
        vz_mean.T,                       # color by <v_z>_mass
        cmap='RdBu_r', norm=norm,
        angles='xy', scale_units='xy', scale=1.0,
        minlength=0, headlength=0, headwidth=0, headaxislength=0,  # look like lines, not arrows
        width=0.005
    )
    
    cb = fig.colorbar(Q, ax=ax, label=r"mass-weighted $v_{\rm LOS}$  [km s$^{-1}$]")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_title(f"TNG50-1 snap {snap} — FoF halo {halo_id}\nDirection: ⟨vₓ, vᵧ⟩ (lines); Color: ⟨v_z⟩ (RdBu)")
    ax.set_xlim([-half, half])
    ax.set_ylim([-half, half])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(outpath+f'snap{snap}halo{halo_id}_vmap.pdf',bbox_inches='tight')
