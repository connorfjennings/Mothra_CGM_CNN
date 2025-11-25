import os, re, sys, glob, math, json, random, numpy as np
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import segmentation_models_pytorch as smp
from architecture import UNetBasic, UNetDropout, UNetMultiHeadProfiles
import pandas as pd

# ---------------- config ----------------
#DATA_DIR   =  "/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/" + sys.argv[1]  # folder with your .npy packs
#PATTERN    = "TNG50_snap099_subid*_views10_aug8_C5_256x256.npy"
CATALOG    = "/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/catalog_pkls/" + sys.argv[1] #catalog pkl
CHECKPOINTS_DIR = "/home/cj535/palmer_scratch/CNN_checkpoints/" + sys.argv[2]
CHECKPOINTS_NAME = "UNetMultihead"
H, W = 256, 256
IN_CHANNELS = 5            # mask + bolometric + 3 vel bins
TARGET_C = 3                # u,v,w
ALEOTORIC_ERRORS = 0
OUT_CHANNELS = (ALEOTORIC_ERRORS+1) * TARGET_C
R_MASK = 8                     # pixels
BATCH_SIZE = 32
EPOCHS = 100
FREEZE_ENCODER_EPOCHS = 3
LR = 5e-4
DROPOUT_P = 0.3
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#VEL_BINS = [(-300, -100), (-100, 100), (100, 300)]  # 3 input channels
COMPRESSION = 'log10'
# weights
LAMBDA_MAPS  = torch.tensor([2.0, 2.0, 0.5], device=DEVICE) #for u,v,w
VSCALE_MAPS  = 50.0 #typical velocity of the CGM, used in loss calculation so all losses are on a similar scale, up to LAMBDA
LAMBDA_MASS  = 1.0
LAMBDA_FLOW  = 1.0

shell_midpoints = np.arange(20,205,5)
K=shell_midpoints.shape[0]
L=shell_midpoints.shape[0]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ---------- mask utilities ----------
def circular_outer_mask(H: int, W: int, R: float, center=None, device="cpu"):
    """
    Returns a mask of shape (1,H,W): 1 outside the circle of radius R, 0 inside.
    center: (yc, xc) in pixel coords; default is image center.
    """
    yc, xc = center if center is not None else (H/2.0, W/2.0)
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    rr2 = (yy - yc)**2 + (xx - xc)**2
    mask = (rr2 >= R**2).float().unsqueeze(0)  # (1,H,W)
    return mask

# ---------- masked loss ----------
def masked_mse(pred, target, mask, eps=1e-8):
    """
    pred, target: (B,2,H,W)
    mask: (B,1,H,W) with 1 where loss is computed (outside circle), 0 where ignored.
    """
    diff2 = (pred - target)**2
    wsum = mask.sum()
    if wsum < eps:
        # if everything is masked, return zero to avoid NaNs
        return diff2.new_zeros(())
    return (diff2 * mask).sum() / (wsum * pred.shape[1])  # average over valid pixels & channels

def masked_gaussian_nll(pred, target, mask, channel_weights=None, eps=1e-8, sigma_min=5.0, sigma_max=300.0):
    """
    Heteroscedastic masked Gaussian NLL.
    pred:   (B, 2C, H, W) if predicting mean and logvar per channel (here C=2 for u,v).
            First C channels = mean; next C channels = log(variance) per pixel.
            If pred has only C channels, falls back to masked MSE.
    target: (B, C, H, W)
    mask:   (B, 1, H, W) with 1 where valid, 0 where ignored.
    """
    B, C, H, W = target.shape
    mask = mask.to(dtype=target.dtype)                  # (B,1,H,W)
    valid = mask.sum()                                  # scalar

    if valid < eps:
        return target.new_zeros(())

    if channel_weights is None:
        w_c = 1.0
    else:
        # channel_weights: (C,)
        w_c = channel_weights.view(1, C, 1, 1)   # broadcast to (1,C,1,1)

    if pred.shape[1] == C:
        # plain masked MSE
        diff2 = (pred - target) ** 2 / VSCALE_MAPS**2
        diff2 = diff2 * w_c
        return (diff2 * mask).sum() / (valid * C)

    elif pred.shape[1] == 2 * C:
        mu, s = pred[:, :C], pred[:, C:]
        # bounded variance
        var = (sigma_min**2) + (sigma_max**2 - sigma_min**2) * torch.sigmoid(s)
        logvar = torch.log(var)
    
        diff2 = (mu - target)**2
        nll = 0.5 * (diff2 / var + logvar)
        nll = nll * w_c
        return (nll * mask).sum() / (valid * C)

    else:
        raise ValueError(f"pred has {pred.shape[1]} channels; expected {C} or {2*C}.")


def masked_mae(pred, target, mask, eps=1e-8):
    B, C, H, W = target.shape
    diff = (pred[:,:C] - target).abs()
    wsum = mask.sum()
    if wsum < eps:
        return diff.new_zeros(())
    return (diff * mask).sum() / (wsum * C).clamp_min(1.0)



# ---------------- dataset using in-RAM packs ----------------
class GalaxyPackDataset(Dataset):
    def __init__(self, packs, subids, mean=None, std=None, compression='log10',
                 r_mask=10, in_channels=4,
                 subid_to_Mprof=None, subid_to_Fprof=None,
                 Mprof_mode="log10", Fprof_mode=None,        # None => no log
                 M_mean=None, M_std=None, F_mean=None, F_std=None,
                 channel0mask=True):
        self.subids = list(subids)
        self.packs  = packs
        self.items  = []
        for sid in self.subids:
            N = self.packs[sid].shape[0]
            self.items.extend((sid, i) for i in range(N))
        self.mask = circular_outer_mask(H, W, r_mask, device="cpu")
        self.mean, self.std = mean, std
        self.compression = compression
        self.in_channels = in_channels
        self.channel0mask=channel0mask

        # profiles
        self.subid_to_Mprof = subid_to_Mprof or {}
        self.subid_to_Fprof = subid_to_Fprof or {}
        self.Mprof_mode = Mprof_mode
        self.Fprof_mode = Fprof_mode
        self.M_mean = None if M_mean is None else np.asarray(M_mean, np.float32)  # (K,)
        self.M_std  = None if M_std  is None else np.asarray(M_std,  np.float32)
        self.F_mean = None if F_mean is None else np.asarray(F_mean, np.float32)  # (L,)
        self.F_std  = None if F_std  is None else np.asarray(F_std,  np.float32)

    def __len__(self): return len(self.items)

    def _encode_M(self, sid):
        v = np.asarray(self.subid_to_Mprof[sid], dtype=np.float32)  # (K,)
        if self.Mprof_mode == "log10":
            v = np.log10(np.clip(v, 1e-30, None))
        return (v - self.M_mean) / self.M_std

    def _encode_F(self, sid):
        v = np.asarray(self.subid_to_Fprof[sid], dtype=np.float32)  # (L,)
        # no log if signed
        if self.F_mean is not None and self.F_std is not None:
            v = (v - self.F_mean) / self.F_std
        return v

    def __getitem__(self, idx):
        sid, i = self.items[idx]
        sample = self.packs[sid][i]
        if self.channel0mask:
            xmask = sample[0]
            x = sample[1:self.in_channels]
        else:
            x = sample[:self.in_channels]
        y = sample[self.in_channels:]

        if self.compression == 'sqrt':
            x = np.sqrt(x)
        elif self.compression == 'log10':
            x = np.log10(x + 1e-25)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-21)
        x = x * self.mask.numpy()
        x = np.concatenate([xmask[None, ...], x], axis=0)  # add mask channel back 

        Mp = self._encode_M(sid).astype(np.float32)  # (K,)
        Fp = self._encode_F(sid).astype(np.float32)  # (L,)

        if self.channel0mask:
            lossmask = (xmask[None, ...] * self.mask.numpy()).astype(np.float32)
            return {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y),
                "mask": torch.from_numpy(lossmask),
                "Mprof": torch.from_numpy(Mp),   # (K,)
                "Fprof": torch.from_numpy(Fp),   # (L,)
                "subid": sid
            }
        else:
            return {
                "x": torch.from_numpy(x),
                "y": torch.from_numpy(y),
                "mask": self.mask.clone(),
                "Mprof": torch.from_numpy(Mp),   # (K,)
                "Fprof": torch.from_numpy(Fp),   # (L,)
                "subid": sid
            }

def compute_profile_norm(mapping, train_subids, mode="log10"):
    # mapping: dict sid -> (D,) np.array
    arrs = []
    for sid in train_subids:
        v = np.asarray(mapping[sid], dtype=np.float32)
        if mode == "log10":
            v = np.log10(np.clip(v, 1e-25, None))  # guard
        arrs.append(v[None, :])  # (1,D)
    A = np.concatenate(arrs, axis=0)  # (Ntrain, D)
    mean = A.mean(axis=0)             # (D,)
    std  = A.std(axis=0) + 1e-8       # (D,)
    return mean.astype(np.float32), std.astype(np.float32)

def compute_input_norm(packs, subids, compression='log10', inC=IN_CHANNELS, channel0mask=True):
    if channel0mask:
        C_eff = inC - 1   # we do NOT normalize the mask channel
    else:
        C_eff = inC

    s = np.zeros(C_eff, dtype=np.float64)
    q = np.zeros(C_eff, dtype=np.float64)
    n = 0

    for sid in subids:
        arr = packs[sid]   # (N, C_total, H, W)
        if channel0mask:
            x = arr[:, 1:inC, :, :]   # drop mask channel; (N, C_eff, H, W)
        else:
            x = arr[:, :inC, :, :]    # (N, C_eff, H, W)

        if compression == 'sqrt':
            x = np.sqrt(x)
        elif compression == 'log10':
            x = np.log10(x + 1e-25)

        n += x.shape[0] * H * W
        xc = x.transpose(1, 0, 2, 3).reshape(C_eff, -1)  # (C_eff, N*H*W)
        s += xc.sum(axis=1)
        q += (xc**2).sum(axis=1)

    mean = s / n
    var  = (q / n) - mean**2
    std  = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------- training / eval ----------------
def vec_loss(pred, target):  # pred:(B,D) target:(B,D)
    return F.smooth_l1_loss(pred, target)
    
def train_one_epoch(model, loader, opt, scaler=None):
    model.train()
    avg = {"loss":0.,"map_mae":0.,"M_mae":0.,"F_mae":0.}
    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)
        m = batch["mask"].to(DEVICE)
        M = batch["Mprof"].to(DEVICE)  # (B,K)
        Fv= batch["Fprof"].to(DEVICE)  # (B,L)

        opt.zero_grad(set_to_none=True)
        autocast_on = ("cuda" in DEVICE)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_on):
                out = model(x)
                loss_maps = masked_gaussian_nll(out["maps"], y, m, channel_weights=LAMBDA_MAPS)
                loss_M    = vec_loss(out["mass_prof"], M)
                loss_F    = vec_loss(out["flow_prof"], Fv)
                loss = loss_maps + LAMBDA_MASS*loss_M + LAMBDA_FLOW*loss_F
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            out = model(x)
            loss_maps = masked_gaussian_nll(out["maps"], y, m, channel_weights=LAMBDA_MAPS)
            loss_M    = vec_loss(out["mass_prof"], M)
            loss_F    = vec_loss(out["flow_prof"], Fv)
            loss = loss_maps + LAMBDA_MASS*loss_M + LAMBDA_FLOW*loss_F
            loss.backward(); opt.step()

        # report MAEs in normalized space; convert later to physical units for logs
        map_mae = masked_mae(out["maps"].detach(), y, m).item()
        M_mae   = F.l1_loss(out["mass_prof"].detach(), M).item()
        F_mae   = F.l1_loss(out["flow_prof"].detach(), Fv).item()

        avg["loss"]    += loss.item()
        avg["map_mae"] += map_mae
        avg["M_mae"]   += M_mae
        avg["F_mae"]   += F_mae

    n = len(loader)
    for k in avg: avg[k] /= n
    return avg

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    avg = {"loss":0.,"map_mae":0.,"M_mae":0.,"F_mae":0.}
    for batch in loader:
        x = batch["x"].to(DEVICE)
        y = batch["y"].to(DEVICE)
        m = batch["mask"].to(DEVICE)
        M = batch["Mprof"].to(DEVICE)
        Fv= batch["Fprof"].to(DEVICE)

        out = model(x)
        loss_maps = masked_gaussian_nll(out["maps"], y, m, channel_weights=LAMBDA_MAPS)
        loss_M    = vec_loss(out["mass_prof"], M)
        loss_F    = vec_loss(out["flow_prof"], Fv)
        loss      = loss_maps + LAMBDA_MASS*loss_M + LAMBDA_FLOW*loss_F

        map_mae = masked_mae(out["maps"], y, m).item()
        M_mae   = F.l1_loss(out["mass_prof"], M).item()
        F_mae   = F.l1_loss(out["flow_prof"], Fv).item()

        avg["loss"]    += loss.item()
        avg["map_mae"] += map_mae
        avg["M_mae"]   += M_mae
        avg["F_mae"]   += F_mae
    n = len(loader)
    for k in avg: avg[k] /= n
    return avg

def unnorm_mae_profile(mae_z, std_vec):
    # mae_z is scalar averaged over bins; approx back to units by mean(std)
    return mae_z * float(np.mean(std_vec))

def main():
    # 1) load all galaxy packs into RAM
    #packs = find_packs(DATA_DIR)  # dict: subid -> (N,5,256,256)
    #all_subids = sorted(packs.keys(), key=lambda s: int(s))
    #print(f"Loaded {len(all_subids)} galaxies into RAM.")
    catalog_path = CATALOG
    df = pd.read_pickle(catalog_path)
    # ensure subids are strings (since your packs/dataset use string keys)
    df.index = df.index.astype(str)

    # 1) load all galaxy packs into RAM from catalog
    packs = {}
    subid_to_Mprof = {}
    subid_to_Fprof = {}

    for sid in df.index:
        row = df.loc[sid]
        arr = np.load(row["maps_path"])  # (N, C, H, W)
        packs[sid] = arr.astype(np.float32, copy=False)

        subid_to_Mprof[sid] = row["mass_profile"]  # (K,)
        subid_to_Fprof[sid] = row["flow_profile"]  # (L,)

    all_subids = sorted(packs.keys(), key=lambda s: int(s))
    print(f"Loaded {len(all_subids)} galaxies into RAM (with 1D profiles).")

    
    # 2) split by subid (no leakage)
    groups = np.array([int(s) for s in all_subids])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    # Split operates on indices; use subids as both samples and groups
    idx = np.arange(len(all_subids))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    train_subids = [all_subids[i] for i in train_idx]
    test_subids  = [all_subids[i] for i in test_idx]
    print(f"Train galaxies: {len(train_subids)} | Test galaxies: {len(test_subids)}")
    #

    K = len(next(iter(subid_to_Mprof.values())))
    L = len(next(iter(subid_to_Fprof.values())))

    # 3) compute input normalization on train only
    
    M_mean, M_std = compute_profile_norm(subid_to_Mprof, train_subids, mode="log10")

    def compute_z_norm(mapping, train_subids):
        arrs = [np.asarray(mapping[s], np.float32)[None, :] for s in train_subids]
        A = np.concatenate(arrs, axis=0)
        return A.mean(0).astype(np.float32), (A.std(0) + 1e-8).astype(np.float32)

    F_mean, F_std = compute_z_norm(subid_to_Fprof, train_subids)

    mean, std = compute_input_norm(packs, train_subids, compression=COMPRESSION, inC=IN_CHANNELS, channel0mask=True)
    print("Input mean:", mean, "std:", std)

    # 4) datasets / loaders
    train_ds = GalaxyPackDataset(
        packs, train_subids,
        mean=mean, std=std,
        r_mask=R_MASK, compression=COMPRESSION,
        in_channels=IN_CHANNELS,
        subid_to_Mprof=subid_to_Mprof,
        subid_to_Fprof=subid_to_Fprof,
        Mprof_mode="log10", Fprof_mode=None,
        M_mean=M_mean, M_std=M_std,
        F_mean=F_mean, F_std=F_std,
    )
    test_ds = GalaxyPackDataset(
        packs, test_subids,
        mean=mean, std=std,
        r_mask=R_MASK, compression=COMPRESSION,
        in_channels=IN_CHANNELS,
        subid_to_Mprof=subid_to_Mprof,
        subid_to_Fprof=subid_to_Fprof,
        Mprof_mode="log10", Fprof_mode=None,
        M_mean=M_mean, M_std=M_std,
        F_mean=F_mean, F_std=F_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=("cuda" in DEVICE),
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=("cuda" in DEVICE),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # 5) model/optim
    MODEL = UNetMultiHeadProfiles(in_channels=IN_CHANNELS,out_channels=OUT_CHANNELS,p=DROPOUT_P,K=K,L=L)
    model = MODEL.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler= torch.amp.GradScaler(enabled=("cuda" in DEVICE))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.3, patience=3,
        threshold=1e-4, min_lr=1e-6
    )

    # 6) train
    #os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(CHECKPOINTS_DIR,CHECKPOINTS_NAME+"_cgm_best.pt")#"checkpoints/unet_cgm_best.pt"
    for p in model.net.encoder.parameters(): p.requires_grad = False  # warmup 3–5 epochs
    
    for epoch in range(1, EPOCHS+1):
        if epoch > FREEZE_ENCODER_EPOCHS:
            for p in model.net.encoder.parameters(): p.requires_grad = True
                
        tr = train_one_epoch(model, train_loader, opt, scaler)
        va = evaluate(model, test_loader)
        scheduler.step(va['loss'])

        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"train loss {tr['loss']:.4f} | map-mae {tr['map_mae']:.4f} | "
            f"Mprof-mae(z) {tr['M_mae']:.4f} (~×{np.mean(M_std):.3f} dex) | "
            f"Fprof-mae(z) {tr['F_mae']:.4f} (~×{np.mean(F_std):.3f} units) | "
            f"val loss {va['loss']:.4f} | val map-mae {va['map_mae']:.4f}"
        )

        # ----- save BEST checkpoint -----
        if va['loss'] < best_val:
            best_val = va['loss']
            torch.save({
                "model": model.state_dict(),
                "mean": mean, "std": std,
                "M_mean": M_mean, "M_std": M_std,
                "F_mean": F_mean, "F_std": F_std,
                "epoch": epoch, "val_loss": va['loss'],
                "config": {
                    "R_MASK": R_MASK, "H": H, "W": W,
                    "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                    "LAMBDA_MAPS": LAMBDA_MAPS,
                    "LAMBDA_MASS": LAMBDA_MASS,
                    "LAMBDA_FLOW": LAMBDA_FLOW,
                }
            }, best_path)
            print(f"  ✓ saved best → {best_path}")
    
        # ----- save PERIODIC checkpoint every 10 epochs -----
        if epoch % 10 == 0:
            periodic_path = os.path.join(CHECKPOINTS_DIR, CHECKPOINTS_NAME+f"_cgm_epoch{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "mean": mean, "std": std,
                "M_mean": M_mean, "M_std": M_std,
                "F_mean": F_mean, "F_std": F_std,
                "epoch": epoch, "val_loss": va['loss'],
                "config": {
                    "R_MASK": R_MASK, "H": H, "W": W,
                    "BATCH_SIZE": BATCH_SIZE, "LR": LR,
                    "LAMBDA_MAPS": LAMBDA_MAPS,
                    "LAMBDA_MASS": LAMBDA_MASS,
                    "LAMBDA_FLOW": LAMBDA_FLOW,
                }
            }, periodic_path)
            print(f"  • saved periodic → {periodic_path}")
        print(f"finished epoch {epoch}")
    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()
