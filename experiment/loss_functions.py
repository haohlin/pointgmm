import numpy as np
import torch

EPS = 1.0e-15

def max_diag_dist(original):
    """
    Find maximum diagnal distance
    """
    max_coord_original = np.max(original, axis=0)
    min_coord_original = np.min(original, axis=0)
    diag_dist_original = np.sqrt(np.sum((max_coord_original - min_coord_original) ** 2))

    return diag_dist_original

def PSNR(recontruction, original, max_dist): 
    if recontruction.shape[0] == 0:
        print('Not enough points. abort')
        return 100
    closest_dist = np.zeros((recontruction.shape[0]), dtype=np.float32)

    for i in range(recontruction.shape[0]):
        dist = np.sum((recontruction[i] - original) ** 2, axis=1)
        closest_dist[i] = np.min(dist)

    mse = np.mean(closest_dist) 
    if(mse < EPS):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(max_dist / np.sqrt(mse)) 
    return psnr 

# Modified from: https://github.com/wentaoyuan/deepgmr 
def chamfer_loss(gts, preds, side="both", reduce=False):
    P = batch_pairwise_dist(gts, preds)
    if reduce:
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        if side == "both":
            return loss_1 + loss_2
        elif side == "left":
            return loss_1
        elif side == "right":
            return loss_2
    else:
        mins, _ = torch.min(P, 1)
        loss_1 = torch.mean(mins, dim=1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins, dim=1)
        if side == "both":
            return loss_1 + loss_2
        elif side == "left":
            return loss_1
        elif side == "right":
            return loss_2

def batch_pairwise_dist(x, y):
    x = x.float()
    y = y.float()
    bs, num_points_x, points_dim = x.size()
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2)[:, None]
    xy = torch.matmul(x, y.transpose(2, 1))
    P = (xx + yy - 2*xy)
    return P
