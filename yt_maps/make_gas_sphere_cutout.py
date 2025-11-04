#!/usr/bin/env python3
import os, sys, h5py, numpy as np
import illustris_python as il

try:
    # Optional MPI acceleration: use if you launch with mpirun -np N ...
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1

# ---------------- user params ----------------
basepath   = "/gpfs/gibbs/pi/nagai/IllustrisTNG_data/TNG50-1/output"
snapnum    = 99
fof_id     = 20
radius_kpc = 500.0                        # sphere radius in *proper* kpc
outdir     = "./cutouts"
outfile    = f"TNG50_snap{snapnum:03d}_fof{fof_id:06d}_gas_sphere_1500kpc.hdf5"
chunk_rows = 2_000_000                    # I/O chunk size
pad_ckpch  = 5.0                          # pad local box by this many ckpc/h
# ---------------------------------------------

Msun_to_g   = 1.98847e33
kpc_to_cm   = 3.085677581491367e21

os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, outfile)

# Header + FoF center
hdr = il.groupcat.loadHeader(basepath, snapnum)
Time   = float(hdr["Time"])                 # scale factor
HubbleParam   = float(hdr["HubbleParam"])          # hubble param
Omega0      = float(hdr.get("Omega0", 0.3089))
OmegaLambda = float(hdr.get("OmegaLambda", 0.6911))

grp = il.groupcat.loadSingle(basepath, snapnum, haloID=fof_id)
center_ckpch = grp["GroupPos"].astype(np.float64)   # (3,), in ckpc/h

# Convert selection radius: proper kpc -> ckpc/h
# proper_kpc * (h/a) = ckpc/h   (because comoving kpc = proper/a, then *h)
R_sel_ckpch = (radius_kpc) * (HubbleParam / Time)

# Resolve snapshot subfiles and BoxSize
snapdir = os.path.join(basepath, f"snapdir_{snapnum:03d}")
files = sorted([os.path.join(snapdir, f) for f in os.listdir(snapdir) if f.endswith(".hdf5")])
if not files:
    raise RuntimeError(f"No snapshot files found in {snapdir}")
with h5py.File(files[0], "r") as f0:
    BoxSize_ckpch = float(f0["Header"].attrs["BoxSize"])

# Helper: periodic distances (squared), vectorized
def periodic_d2(pos_ckpch, ctr_ckpch, box):
    d = pos_ckpch - ctr_ckpch[None, :]
    # wrap to [-box/2, box/2]
    d -= np.round(d / box) * box
    return np.einsum("ij,ij->i", d, d)

# Distribute files across ranks (round-robin)
my_files = files[RANK::SIZE]

# Fields we will try to read (skip if missing)
want_fields = ["Coordinates","Velocities","Masses","Density",
               "InternalEnergy","ElectronAbundance",
               "GFM_Metallicity","GFM_Metals","ParticleIDs"]

# Collect locally
local = {k: [] for k in want_fields}

for path in my_files:
    with h5py.File(path, "r") as f:
        if "PartType0" not in f:  # no gas in this subfile
            continue
        g = f["PartType0"]
        N = g["Coordinates"].shape[0]
        # determine which fields exist in this subfile
        have = [k for k in want_fields if k in g.keys()]

        for start in range(0, N, chunk_rows):
            stop = min(start + chunk_rows, N)
            pos = g["Coordinates"][start:stop].astype(np.float64)  # ckpc/h
            msk = periodic_d2(pos, center_ckpch, BoxSize_ckpch) <= (R_sel_ckpch**2)
            if not msk.any():
                continue
            # store pos first (always exists)
            local["Coordinates"].append(pos[msk].astype(np.float32))
            # other fields if present
            for k in have:
                if k == "Coordinates":
                    continue
                local[k].append(g[k][start:stop][msk])

# Concatenate local shards
for k in list(local.keys()):
    if len(local[k]) == 0:
        local[k] = None
    else:
        local[k] = np.concatenate(local[k], axis=0)

# Gather to rank 0
if SIZE > 1 and COMM is not None:
    gathered = COMM.gather(local, root=0)
else:
    gathered = [local]

if RANK != 0:
    sys.exit(0)

# Merge on rank 0
global_data = {k: [] for k in want_fields}
for d in gathered:
    for k, v in d.items():
        if v is not None:
            global_data[k].append(v)

for k in list(global_data.keys()):
    if len(global_data[k]) == 0:
        global_data[k] = None
    else:
        global_data[k] = np.concatenate(global_data[k], axis=0)

count = 0 if global_data["Coordinates"] is None else global_data["Coordinates"].shape[0]
print(f"Selected gas cells within {radius_kpc:.1f} kpc proper: {count:,}")

if count == 0:
    raise SystemExit("No gas found in the requested sphere; check center/radius.")

# Build a *local* box around the selected points (still in ckpc/h)
mins = global_data["Coordinates"].min(axis=0)
maxs = global_data["Coordinates"].max(axis=0)
lo   = mins - pad_ckpch
hi   = maxs + pad_ckpch
local_box = hi - lo
BoxSize_local = float(local_box.max())  # single scalar BoxSize (ckpc/h)

# Shift coordinates into [0, BoxSize_local)
coords_shift = (global_data["Coordinates"] - lo[None, :]).astype(np.float32)

# Write mini-snapshot (Gadget-like)
with h5py.File(outpath, "w") as f:
    # Header
    h = f.create_group("Header")
    counts = np.zeros(6, dtype=np.uint32)
    counts[0] = coords_shift.shape[0]  # only PartType0 here
    h.attrs["NumPart_ThisFile"]         = counts
    h.attrs["NumPart_Total"]            = counts
    h.attrs["NumPart_Total_HighWord"]   = np.zeros(6, dtype=np.uint32)
    h.attrs["MassTable"]                = np.zeros(6, dtype=np.float64)  # variable masses
    h.attrs["Time"]                     = Time
    h.attrs["Redshift"]                 = 1.0/Time - 1.0
    h.attrs["BoxSize"]                  = BoxSize_local
    h.attrs["NumFilesPerSnapshot"]      = 1
    h.attrs["Omega0"]                   = Omega0
    h.attrs["OmegaLambda"]              = OmegaLambda
    h.attrs["HubbleParam"]              = HubbleParam
    # provenance
    h.attrs["SelectionCenter_ckpch"]    = center_ckpch
    h.attrs["SelectionRadius_kpc_proper"]= radius_kpc
    h.attrs["LocalBoxLo_ckpch"]         = lo
    h.attrs["LocalBoxHi_ckpch"]         = hi
    h.attrs["OriginalBoxSize_ckpch"]    = BoxSize_ckpch
    
    h.attrs["UnitLength_in_cm"]           = kpc_to_cm #/ HubbleParam     # (kpc/h) in cm
    h.attrs["UnitMass_in_g"]              = (1e10 * Msun_to_g) #/ HubbleParam  # (1e10 Msun/h) in g
    h.attrs["UnitVelocity_in_cm_per_s"]   = 1.0e5                       # km/s -> cm/s
    h.attrs["ComovingIntegrationOn"]      = 1                           # cosmological
    h.attrs["Flag_DoublePrecision"]       = 0                           # if you wrote float32

    # PartType0 group
    g = f.create_group("PartType0")
    g.create_dataset("Coordinates", data=coords_shift, dtype=np.float32)
    # copy any available fields (verbatim native units)
    for k in ["Velocities","Masses","Density","InternalEnergy",
              "ElectronAbundance","GFM_Metallicity","GFM_Metals","ParticleIDs"]:
        if global_data[k] is not None:
            g.create_dataset(k, data=global_data[k])

print(f"Wrote cutout: {outpath}")
