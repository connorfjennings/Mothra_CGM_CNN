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
radius_kpc = 1000.0                        # sphere radius in *proper* kpc
outdir     = "./cutouts"
outfile    = f"TNG50_snap{snapnum:03d}_fof{fof_id:06d}_gas_sphere_1000kpc.hdf5"
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

COMM.Barrier(); t0 = MPI.Wtime()

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

if COMM is None:
    raise RuntimeError("Run this script with mpirun/mpiexec to use the parallel writer.")

# Concatenate each field locally
for k in list(local.keys()):
    if len(local[k]) == 0:
        local[k] = None
    else:
        local[k] = np.concatenate(local[k], axis=0)

# Local counts
localN = 0 if local["Coordinates"] is None else local["Coordinates"].shape[0]
globalN = COMM.allreduce(localN, op=MPI.SUM)

if globalN == 0:
    if RANK == 0:
        print("No gas found in the requested sphere; check center/radius.")
    sys.exit(0)

# --- compute global box (mins/maxs) and shift ---

if localN > 0:
    mins_loc = local["Coordinates"].min(axis=0).astype(np.float64)
    maxs_loc = local["Coordinates"].max(axis=0).astype(np.float64)
else:
    mins_loc = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs_loc = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

mins_glb = np.empty(3, dtype=np.float64)
maxs_glb = np.empty(3, dtype=np.float64)
COMM.Allreduce(mins_loc, mins_glb, op=MPI.MIN)
COMM.Allreduce(maxs_loc, maxs_glb, op=MPI.MAX)

lo = mins_glb - pad_ckpch
hi = maxs_glb + pad_ckpch
local_box = hi - lo
BoxSize_local = float(local_box.max())

# Shift coordinates locally
if localN > 0:
    coords_shift_local = (local["Coordinates"] - lo[None, :]).astype(np.float32)
else:
    coords_shift_local = np.empty((0, 3), dtype=np.float32)

COMM.Barrier(); t1 = MPI.Wtime()
if RANK == 0: print(f"Dataset creation took {t1-t0:.1f}s")

# Helper: discover which fields exist globally and their trailing shape/dtype
def global_field_info(name, prefer_dtype=None):
    has_local = int(local[name] is not None)
    has_global = COMM.allreduce(has_local, op=MPI.SUM) > 0
    if not has_global:
        return False, None, None
    # get a representative shape[1:] and dtype from any rank
    shp_tail = None
    dt = None
    if has_local:
        shp_tail = local[name].shape[1:]
        dt = local[name].dtype
    shp_list = COMM.allgather(shp_tail)
    dt_list  = COMM.allgather(dt)
    for st, dti in zip(shp_list, dt_list):
        if st is not None:
            shp_tail = st
        if dti is not None:
            dt = dti
    if prefer_dtype is not None:
        dt = prefer_dtype
    return True, shp_tail, dt

def _as_tail_tuple(tail):
    """Normalize tail to a tuple (possibly empty)."""
    if tail is None:
        return ()
    if isinstance(tail, tuple):
        return tail
    if isinstance(tail, (list, np.ndarray)):
        return tuple(tail)
    # scalar tail (e.g., 3)
    return (int(tail),)

def _make_shape(n_glb, tail):
    """Return (n_glb, *tail) as a proper tuple without star-expansions."""
    return (n_glb,) + _as_tail_tuple(tail)

def _choose_chunks(n_glb, tail, size):
    # Aim for ~O(10–50) chunks per rank; tune as desired
    # Ensure at least 1e5 rows per chunk to keep metadata moderate
    t = _as_tail_tuple(tail)
    target_chunks_per_rank = 32
    base = max(100_000, n_glb // max(1, size * target_chunks_per_rank))
    return (max(1, base),) + t

# Compute offsets for each dataset
def offsets_for(name):
    n_loc = 0 if local[name] is None else local[name].shape[0]
    n_glb = COMM.allreduce(n_loc, op=MPI.SUM)
    # exclusive scan gives the starting index for each rank
    off = COMM.exscan(n_loc)
    if RANK == 0:
        off = 0
    return n_loc, n_glb, off

def tmsg(where):
    COMM.Barrier()
    if RANK == 0:
        print(where, flush=True)
    COMM.Barrier()

tmpdir = os.path.join(outdir, "tmp_shards")
if RANK == 0:
    os.makedirs(tmpdir, exist_ok=True)
if COMM: COMM.Barrier()

rank_path = os.path.join(tmpdir, f"shard_rank{RANK:05d}_"+outfile)
with h5py.File(rank_path, "w") as f:
    h = f.create_group("Header")
    if RANK == 0:
        # write globals once (we’ll copy to final later)
        h.attrs["Time"] = Time; h.attrs["HubbleParam"] = HubbleParam
        h.attrs["Omega0"] = Omega0; h.attrs["OmegaLambda"] = OmegaLambda
        h.attrs["SelectionCenter_ckpch"] = center_ckpch
        h.attrs["SelectionRadius_kpc_proper"] = radius_kpc
        h.attrs["LocalBoxLo_ckpch"] = lo; h.attrs["LocalBoxHi_ckpch"] = hi
        h.attrs["OriginalBoxSize_ckpch"] = BoxSize_ckpch
        h.attrs["UnitLength_in_cm"] = kpc_to_cm
        h.attrs["UnitMass_in_g"] = (1e10*Msun_to_g)
        h.attrs["UnitVelocity_in_cm_per_s"] = 1e5
        h.attrs["ComovingIntegrationOn"] = 1
        h.attrs["Flag_DoublePrecision"] = 0
        h.attrs["BoxSize"] = BoxSize_local
    g = f.create_group("PartType0")
    # write only local rows; use chunked smallish datasets to keep writes fast
    def _w(name, arr):
        if arr is None:
            return
        tail = arr.shape[1:]
        g.create_dataset(name, data=arr, chunks=(min(arr.shape[0], 500_000),)+tail)
    _w("Coordinates", coords_shift_local)
    for fld in ["Velocities","Masses","Density","InternalEnergy",
                "ElectronAbundance","GFM_Metallicity","GFM_Metals","ParticleIDs"]:
        _w(fld, local[fld])

if COMM: COMM.Barrier()
if RANK == 0: print("Wrote per-rank shards. Ready to merge.")

shards = sorted([os.path.join(tmpdir,f) for f in os.listdir(tmpdir) if f.endswith(outfile)])

# count rows per shard
counts = []
for p in shards:
    with h5py.File(p, "r") as f:
        g = f["PartType0"]
        counts.append(g["Coordinates"].shape[0])
offsets = np.cumsum([0]+counts[:-1])
globalN = int(sum(counts))


with h5py.File(outpath, "w") as fout:
    H = fout.create_group("Header")
    # copy attrs from first shard
    with h5py.File(shards[0], "r") as f0:
        for k, v in f0["Header"].attrs.items():
            H.attrs[k] = v
        H.attrs["NumPart_ThisFile"]       = np.array([globalN,0,0,0,0,0], dtype=np.uint32)
        H.attrs["NumPart_Total"]          = np.array([globalN,0,0,0,0,0], dtype=np.uint32)
        H.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        H.attrs["MassTable"]              = np.zeros(6, dtype=np.float64)
        H.attrs["NumFilesPerSnapshot"]    = 1
        # Redshift & Time should already be set, but ensure Redshift:
        if "Time" in H.attrs:
            H.attrs["Redshift"] = 1.0/float(H.attrs["Time"]) - 1.0

    G = fout.create_group("PartType0")

    # figure out which fields exist globally
    fieldnames = set()
    for p in shards:
        with h5py.File(p, "r") as f:
            fieldnames |= set(f["PartType0"].keys())
    fieldnames = sorted(fieldnames)

    # create final datasets
    tails = {}
    dsets = {}
    for name in fieldnames:
        # discover tail/dtype
        tail, dt = None, None
        for p in shards:
            with h5py.File(p, "r") as f:
                g = f["PartType0"]
                if name in g and g[name].shape[0] > 0:
                    tail = g[name].shape[1:]
                    dt = g[name].dtype
                    break
        if tail is None:  # should not happen
            continue
        dsets[name] = G.create_dataset(name, shape=(globalN,)+tail, dtype=dt, chunks=True)
        tails[name] = tail

    # copy data slab-wise
    for p, off, cnt in zip(shards, offsets, counts):
        if cnt == 0:
            continue
        with h5py.File(p, "r") as f:
            g = f["PartType0"]
            for name in fieldnames:
                if name in g:
                    dsets[name][off:off+cnt, ...] = g[name][...]

print("Merged shards ->", outpath)

