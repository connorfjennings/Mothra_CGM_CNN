import os, re, sys, glob, math, json, random, numpy as np
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import segmentation_models_pytorch as smp

# ---------------- config ----------------
DATA_DIR   =  "/home/cj535/palmer_scratch/TNG50_cutouts/MW_sample_maps/" + sys.argv[1]  # folder with your .npy packs
PATTERN    = "TNG50_snap099_subid*_views10_aug8_C5_256x256.npy"
CHECKPOINTS_DIR = "/home/cj535/palmer_scratch/CNN_checkpoints/" + sys.argv[2]
H, W = 256, 256
R_MASK = 20                     # pixels
BATCH_SIZE = 16
EPOCHS = 200
FREEZE_ENCODER_EPOCHS = 10
LR = 5e-4
NUM_WORKERS = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VEL_BINS = [(-300, -100), (-100, 100), (100, 300)]  # 3 input channels
COMPRESSION = 'log10'

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

def masked_mae(pred, target, mask, eps=1e-8):
    diff = (pred - target).abs()
    wsum = mask.sum()
    if wsum < eps:
        return diff.new_zeros(())
    return (diff * mask).sum() / (wsum * pred.shape[1]).clamp_min(1.0)

subid_re = re.compile(r".*?_subid(?P<subid>\d+)_views10_aug8_C5_256x256\.npy$")
def find_packs(data_dir: str) -> Dict[str, np.ndarray]:
    packs = {}
    for path in glob.glob(os.path.join(data_dir, PATTERN)):
        m = subid_re.match(path)
        if not m: 
            continue
        subid = m.group("subid")
        # Load fully into RAM as float32 ndarray
        arr = np.load(path)  # already float32 per your save; if not: .astype(np.float32, copy=False)
        if arr.shape[1] != 5 or arr.shape[2:] != (H, W):
            raise RuntimeError(f"Unexpected shape {arr.shape} in {path}")
        packs[subid] = arr
    if not packs:
        raise RuntimeError("No packs found. Check DATA_DIR/PATTERN.")
    return packs

# ---------------- dataset using in-RAM packs ----------------
class GalaxyPackDataset(Dataset):
    """
    Yields individual (view×aug) samples from a list of subids.
    Expects memory-resident dict: subid -> ndarray (N,5,H,W).
    Normalization (mean/std) is applied to input channels only.
    """
    def __init__(self, packs: Dict[str, np.ndarray], subids: List[str], mean=None, std=None, compression='log10',r_mask=R_MASK):
        self.subids = list(subids)
        self.packs = packs
        # Build an index: for each subid, iterate over samples
        self.items = []  # list of (subid, local_idx)
        for sid in self.subids:
            N = self.packs[sid].shape[0]
            self.items.extend((sid, i) for i in range(N))
        self.mask = circular_outer_mask(H, W, r_mask, device="cpu")  # (1,H,W)
        self.mean = mean  # (3,)
        self.std  = std   # (3,)
        self.compression = compression

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        sid, i = self.items[idx]
        sample = self.packs[sid][i]       # (5,H,W) float32
        x = sample[:3]                    # (3,H,W)
        y = sample[3:]                    # (2,H,W)
        if self.compression == 'sqrt':
            x = np.sqrt(x)
        elif self.compression == 'log10':
            x = np.log10(x+1e-25)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-21)
        # zero inputs in center
        x = x * self.mask.numpy()
        return {
            "x": torch.from_numpy(x),     # float32
            "y": torch.from_numpy(y),
            "mask": self.mask.clone(),    # torch float32 (1,H,W)
            "subid": sid
        }

def compute_input_norm(packs: Dict[str, np.ndarray], subids: List[str],compression='log10'):
    """
    Compute per-channel mean/std over the 3 brightness channels using ONLY the train subids.
    """
    s = np.zeros(3, dtype=np.float64)
    q = np.zeros(3, dtype=np.float64)
    n = 0
    for sid in subids:
        arr = packs[sid]          # (N,5,H,W)
        x = arr[:, :3, :, :]      # (N,3,H,W)
        if compression == 'sqrt':
            x = np.sqrt(x)
        elif compression == 'log10':
            x = np.log10(x+1e-25)
        n += x.shape[0]*H*W
        s += x.reshape(-1,3,H,W).transpose(1,0,2,3).reshape(3,-1).sum(axis=1)
        q += (x**2).reshape(-1,3,H,W).transpose(1,0,2,3).reshape(3,-1).sum(axis=1)
    mean = s / n
    var  = (q / n) - mean**2
    std  = np.sqrt(var)#np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)

# ---------------- training / eval ----------------
def train_one_epoch(model, loader, opt, scaler=None):
    model.train()
    tot_loss, tot_mae = 0.0, 0.0
    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        m = batch["mask"].to(DEVICE, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = masked_mse(pred, y, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            pred = model(x); loss = masked_mse(pred, y, m)
            loss.backward(); opt.step()

        mae = masked_mae(pred.detach(), y, m).item()
        tot_loss += loss.item(); tot_mae += mae
    n = len(loader)
    return tot_loss/n, tot_mae/n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_loss, tot_mae = 0.0, 0.0
    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)
        m = batch["mask"].to(DEVICE, non_blocking=True)
        pred = model(x)
        loss = masked_mse(pred, y, m)
        mae  = masked_mae(pred, y, m).item()
        tot_loss += loss.item(); tot_mae += mae
    n = len(loader)
    return tot_loss/n, tot_mae/n

def main():
    # 1) load all galaxy packs into RAM
    packs = find_packs(DATA_DIR)  # dict: subid -> (N,5,256,256)
    all_subids = sorted(packs.keys(), key=lambda s: int(s))
    print(f"Loaded {len(all_subids)} galaxies into RAM.")

    # 2) split by subid (no leakage)
    groups = np.array([int(s) for s in all_subids])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    # Split operates on indices; use subids as both samples and groups
    idx = np.arange(len(all_subids))
    train_idx, test_idx = next(splitter.split(idx, groups=groups))
    train_subids = [all_subids[i] for i in train_idx]
    test_subids  = [all_subids[i] for i in test_idx]
    print(f"Train galaxies: {len(train_subids)} | Test galaxies: {len(test_subids)}")

    # 3) compute input normalization on train only
    mean, std = compute_input_norm(packs, train_subids,compression=COMPRESSION)
    print("Input mean:", mean, "std:", std)

    # 4) datasets / loaders
    train_ds = GalaxyPackDataset(packs, train_subids, mean=mean, std=std, r_mask=R_MASK,compression=COMPRESSION)
    test_ds  = GalaxyPackDataset(packs, test_subids,  mean=mean, std=std, r_mask=R_MASK,compression=COMPRESSION)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=("cuda" in DEVICE),
                              persistent_workers=(NUM_WORKERS>0))
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=("cuda" in DEVICE),
                              persistent_workers=(NUM_WORKERS>0))

    # 5) model/optim
    model = smp.FPN(
        encoder_name="resnet50",       # good starting point; try "resnet50", "convnext_tiny", "efficientnet-b3", etc.
        encoder_weights="imagenet",    # <-- THIS loads pretrained encoder weights
        in_channels=3,                 # your three velocity-bin brightness maps
        classes=2,                     # 2 output channels (u, v)
        activation=None                # regression: keep raw logits
    ).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler= torch.amp.GradScaler(enabled=("cuda" in DEVICE))

    # 6) train
    #os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")
    best_path = os.path.join(CHECKPOINTS_DIR,"fpn_cgm_best.pt")#"checkpoints/unet_cgm_best.pt"
    for p in model.encoder.parameters(): p.requires_grad = False  # warmup 3–5 epochs
    
    for epoch in range(1, EPOCHS+1):
        if epoch > FREEZE_ENCODER_EPOCHS:
            for p in model.encoder.parameters(): p.requires_grad = True
                
        tr_loss, tr_mae = train_one_epoch(model, train_loader, opt, scaler)
        va_loss, va_mae = evaluate(model, test_loader)
    
        print(f"[{epoch:03d}/{EPOCHS}] train: loss {tr_loss:.5f}, mae {tr_mae:.5f} | "
              f"val: loss {va_loss:.5f}, mae {va_mae:.5f}")
    
        # ----- save BEST checkpoint -----
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "mean": mean, "std": std,
                "epoch": epoch, "val_loss": va_loss,
                "config": {
                    "R_MASK": R_MASK, "H": H, "W": W,
                    "BATCH_SIZE": BATCH_SIZE, "LR": LR
                }
            }, best_path)
            print(f"  ✓ saved best → {best_path}")
    
        # ----- save PERIODIC checkpoint every 10 epochs -----
        if epoch % 10 == 0:
            periodic_path = os.path.join(CHECKPOINTS_DIR,f"fpn_cgm_epoch{epoch:03d}.pt")
            torch.save({
                "model": model.state_dict(),
                "mean": mean, "std": std,
                "epoch": epoch, "val_loss": va_loss,
                "config": {
                    "R_MASK": R_MASK, "H": H, "W": W,
                    "BATCH_SIZE": BATCH_SIZE, "LR": LR
                }
            }, periodic_path)
            print(f"  • saved periodic → {periodic_path}")
        print(f"finished epoch {epoch}")
    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()
