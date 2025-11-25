import os, re, sys, glob, math, json, random, numpy as np
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import segmentation_models_pytorch as smp
from architecture import UNetBasic, UNetDropout


def enable_mc_dropout_only(model: nn.Module):
    model.eval()  # keep BN frozen
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()

@torch.no_grad()
def predict_mc(model, x, mask=None, T=100, device="cuda"):
    """
    x:     (B, inC, H, W) tensor
    mask:  (B, 1, H, W) tensor with 1=valid region (optional)
    returns dict with:
      'mu' (B,C,H,W), 'std_epistemic' (B,C,H,W),
      and if aleatoric present: 'sigma_aleatoric' (B,C,H,W), 'std_total'
    """
    x = x.to(device, non_blocking=True)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    enable_mc_dropout_only(model)
    preds = []
    for _ in range(T):
        y = model(x)                     # (B,C or 2C,H,W)
        preds.append(y)
    preds = torch.stack(preds, 0)        # (T,B,C_or_2C,H,W)

    B, _, H, W = preds.shape[1:]
    # detect aleatoric head
    outC = preds.shape[2]
    if outC % 2 == 0 and outC >= 4:
        C = outC // 2
        mu      = preds[:, :, :C]                    # (T,B,C,H,W)
        logvar  = preds[:, :, C:]                    # (T,B,C,H,W)
        mu_mean = mu.mean(0)                         # (B,C,H,W)
        std_epi = mu.std(0, unbiased=True)           # epistemic
        sigma_ale = torch.exp(0.5 * logvar).mean(0)  # average aleatoric std
        # Combine (variances add)
        std_total = torch.sqrt(std_epi**2 + sigma_ale**2)
        out = {"mu": mu_mean, "std_epistemic": std_epi,
               "sigma_aleatoric": sigma_ale, "std_total": std_total}
    else:
        # plain regression: no aleatoric
        mu_mean = preds.mean(0)                      # (B,outC,H,W)
        std_epi = preds.std(0, unbiased=True)
        out = {"mu": mu_mean, "std_epistemic": std_epi}

    # apply mask if provided
    if mask is not None:
        for k in out:
            out[k] = out[k] * mask

    return out


@torch.no_grad()
def ood_score_mc_dropout(model, x, mask, T=20):
    enable_mc_dropout_only(model)
    outs = [model(x) for _ in range(T)]             # (T,B,C,H,W or 2C)
    preds = torch.stack(outs,0).float()
    C = x.shape[1]  # not used; infer C from pred shape if needed
    if preds.shape[2] % 2 == 0 and preds.shape[2] >= 4:
        mu = preds[:,:,:preds.shape[2]//2]
    else:
        mu = preds
    sigma_epi = mu.std(0, unbiased=True)            # (B,C,H,W)
    # reduce over channels & space, masked
    m = mask.float()
    per_img = (sigma_epi.mean(1) * m[:,0]).flatten(1)  # (B,H*W)
    score = torch.quantile(per_img, 0.95, dim=1)       # (B,)
    return score  # higher => more OOD


