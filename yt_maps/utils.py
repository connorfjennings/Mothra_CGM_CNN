import os, h5py, numpy as np
from mpi4py import MPI
import yt

# --------------------------- helpers ---------------------------

def _axis_intervals(lo, hi, box):
    # normalize to [0, box)
    lo %= box; hi %= box
    return [(0, hi), (lo, box)] if lo > hi else [(lo, hi)]

def _mask_in_aabb_periodic(pos, intervals):
    # intervals: list of 3 lists of (lo,hi)
    m = np.zeros(pos.shape[0], dtype=bool)
    for xlo, xhi in intervals[0]:
        mx = (pos[:, 0] >= xlo) & (pos[:, 0] < xhi)
        if not mx.any():  # cheap prune
            continue
        for ylo, yhi in intervals[1]:
            my = (pos[:, 1] >= ylo) & (pos[:, 1] < yhi)
            if not my.any():
                continue
            myx = mx & my
            for zlo, zhi in intervals[2]:
                mz = (pos[:, 2] >= zlo) & (pos[:, 2] < zhi)
                if mz.any():
                    m |= (myx & mz)
    return m

def _ptype_key(partType):
    pmap = {"gas":"PartType0", "dm":"PartType1", "dm2":"PartType2", "tracer":"PartType3",
            "stars":"PartType4", "bh":"PartType5", 0:"PartType0", 1:"PartType1",
            2:"PartType2", 3:"PartType3", 4:"PartType4", 5:"PartType5"}
    return pmap[partType]

# --------------------------- main MPI loader ---------------------------

def load_bbox_mpi(
    basepath, snapnum,
    partType="gas",
    center_ckpch=None,          # length-3 array (ckpc/h)
    halfsize_ckpch=None,        # length-3 array (ckpc/h)
    fields=("Coordinates", "Masses"),
    chunk=2_000_000,
    gather=False,               # if True, gather results on rank 0 (can be heavy)
):
    """
    Load all particles of a given type within a periodic AABB using MPI.
    Units: positions in ckpc/h (native snapshot units).
    Returns:
      local_dict if gather==False
      (global_dict on rank 0, None on others) if gather==True
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    pkey = _ptype_key(partType)

    # 1) Rank 0 discovers subfiles and box size, broadcasts to everyone
    if rank == 0:
        snapdir = os.path.join(basepath, f"snapdir_{snapnum:03d}")
        files = sorted([os.path.join(snapdir, f) for f in os.listdir(snapdir) if f.endswith(".hdf5")])
        if not files:
            raise RuntimeError(f"No snapshot files in {snapdir}")
        with h5py.File(files[0], "r") as f0:
            box = float(f0["Header"].attrs["BoxSize"])  # ckpc/h
    else:
        files, box = None, None

    files = comm.bcast(files, root=0)
    box = comm.bcast(box, root=0)

    # 2) Build periodic intervals for the AABB (in ckpc/h)
    center = np.asarray(center_ckpch, dtype=float)
    half   = np.asarray(halfsize_ckpch, dtype=float)
    lo = center - half
    hi = center + half
    intervals = [
        _axis_intervals(lo[0], hi[0], box),
        _axis_intervals(lo[1], hi[1], box),
        _axis_intervals(lo[2], hi[2], box),
    ]

    # 3) Distribute files: simple round-robin by rank
    my_files = files[rank::size]

    # 4) Stream and collect local results
    out_local = {fld: [] for fld in fields}
    for path in my_files:
        with h5py.File(path, "r") as f:
            if pkey not in f:
                continue
            grp = f[pkey]
            N = grp["Coordinates"].shape[0]
            # read in chunks
            for start in range(0, N, chunk):
                stop = min(start + chunk, N)
                pos = grp["Coordinates"][start:stop]  # ckpc/h
                m = _mask_in_aabb_periodic(pos, intervals)
                if not m.any():
                    continue
                for fld in fields:
                    out_local[fld].append(grp[fld][start:stop][m])

    # Concatenate local lists -> arrays
    for k in out_local:
        if len(out_local[k]):
            out_local[k] = np.concatenate(out_local[k])
        else:
            # shape-conscious empty
            if k == "Coordinates":
                out_local[k] = np.empty((0, 3), dtype=np.float32)
            else:
                out_local[k] = np.empty((0,), dtype=np.float32)

    if not gather:
        return out_local, box

    # 5) Gather to rank 0 (pickle-based gather; simplest/robust, but memory-heavy at root)
    gathered = comm.gather(out_local, root=0)
    if rank != 0:
        return None, box

    # rank 0: concat across ranks
    out_global = {k: [] for k in fields}
    for d in gathered:
        for k in fields:
            out_global[k].append(d[k])
    for k in fields:
        out_global[k] = np.concatenate(out_global[k]) if len(out_global[k]) else (
            np.empty((0, 3), dtype=np.float32) if k == "Coordinates" else np.empty((0,), dtype=np.float32)
        )
    return out_global, box

def tng_units_to_yt_stream(g, a, h):
    # g: dict with native units; returns dict of YTArrays in physical units
    # Positions: ckpc/h -> proper kpc
    xyz_kpc = (g["Coordinates"] * (a / h)).astype(np.float64) * 1e-3  # ckpc/h * a/h -> cMpc -> kpc
    data = {
        "particle_position_x": xyz_kpc[:,0],  # kpc
        "particle_position_y": xyz_kpc[:,1],
        "particle_position_z": xyz_kpc[:,2],
        # Masses may be in Msun (Arepo gas is Msun), or 1e10 Msun/h for catalog masses; check your snapshot.
        "particle_mass": g["Masses"],  # convert to Msun if needed
        "density": g["Density"],       # g/cm^3 if from snapshot; otherwise convert accordingly
        "InternalEnergy": g["InternalEnergy"],
        "GFM_Metallicity": g["GFM_Metallicity"],
        "GFM_Metals": g["GFM_Metals"],
        "ElectronAbundance": g["ElectronAbundance"],
        "SmoothingLength": g["SmoothingLength"] * (a / h) * 1e-3,  # ckpc/h -> kpc (proper)
    }
    bbox_kpc = np.array([[xyz_kpc[:,i].min(), xyz_kpc[:,i].max()] for i in range(3)], dtype="float64")
    ds = yt.load_particles(
        data,
        length_unit="kpc", mass_unit="Msun", time_unit="s", bbox=bbox_kpc
    )
    return ds
