import os
import torch
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

EPS = 1.0e-15

def write_ply(verts, output_file, colors=None, indices=None):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    print('Writing ' + output_file + '...')
    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], 
                                                            int(color[0]), int(color[1]), int(color[2])))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    
    print('Done!')
    file.close()

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

def log(log_dir, chamfer_list, chamfer_list_nms, psnr_list, psnr_list_nms, n_components, n_components_nms):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename_chamfer = f'{log_dir}chamfer.txt'
    filename_chamfer_nms = f'{log_dir}chamfer_nms.txt'
    filename_psnr = f'{log_dir}psnr.txt'
    filename_psnr_nms = f'{log_dir}psnr_nms.txt'
    # filename_time = f'{log_dir}{chair}_{id:03d}_time.txt'
    f1 = open(filename_chamfer, 'a')
    f2 = open(filename_chamfer_nms, 'a')
    f3 = open(filename_psnr, 'a')
    f4 = open(filename_psnr_nms, 'a')

    for i in range(len(chamfer_list)):
        f1.write(f'{n_components[i]}:{chamfer_list[i]}\t')
        f2.write(f'{n_components_nms[i]}:{chamfer_list_nms[i]}\t')
        f3.write(f'{n_components[i]}:{psnr_list[i]}\t')
        f4.write(f'{n_components_nms[i]}:{psnr_list_nms[i]}\t')
    f1.write('\n')
    f2.write('\n')
    f3.write('\n')
    f4.write('\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()

def build_gmm(pi, mu, sigma):
    gmm = GaussianMixture(n_components=pi.shape[0])
    gmm.weights_ = pi              # K
    gmm.means_ = mu                # Kx3
    gmm.covariances_ = sigma       # Kx3x3
    return gmm

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Chamfer distance / PSNR calculator')
    parser.add_argument('--input', type=str, default='generated/table_vae/eval_points/hgmms/hgmms_001.npz')
    parser.add_argument('--obj_class', type=str, default='table')
    parser.add_argument('--id', type=str, default='01')
    parser.add_argument('--output_dir', type=str, default='logs/loss/')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    # args
    args = parse_args()
    print('Called with args:')
    print(args)
    
    # import data
    npz_data = np.load(args.input, allow_pickle=True)
    input_pts = torch.from_numpy(npz_data['input_points']).cuda().unsqueeze(0)
    # input_pts = input_pts.repeat(4,1,1)
    pts_seg = torch.from_numpy(npz_data['pts_seg']).cuda()
    pi = npz_data['pi']
    mu = npz_data['mu']
    sigma = npz_data['sigma']
    pi_nms = npz_data['pi_nms']
    mu_nms = npz_data['mu_nms']
    sigma_nms = npz_data['sigma_nms']

    # generate ply file for input point cloud
    # write_ply(verts=input_pts.squeeze(0).cpu(), colors=None, indices=None, output_file=f'./experiment/{args.obj_class}_{args.id}.ply')
    
    # sample points and calculate chamfer loss
    # log_dir = args.output_dir
    # obj_class = args.obj_class
    # obj_id = args.id
    log_dir = f'{args.output_dir}{args.obj_class}_{args.id}/'
    max_dist = max_diag_dist(npz_data['input_points'])

    for s in range(10):
        psnr_list = []
        psnr_list_nms = []
        chamfer_list = []
        chamfer_list_nms = []
        n_components = []
        n_components_nms = []
        for i in range(len(pi)):
            gmm = build_gmm(pi[i], mu[i], sigma[i])
            gmm_nms = build_gmm(pi_nms[i], mu_nms[i], sigma_nms[i])
            reconstruction, labels = gmm.sample(input_pts.shape[1])
            reconstruction_nms, labels_nms = gmm_nms.sample(input_pts.shape[1])

            psnr = PSNR(reconstruction, npz_data['input_points'], max_dist)
            psnr_nms = PSNR(reconstruction_nms, npz_data['input_points'], max_dist)
            psnr_list.append(psnr)
            psnr_list_nms.append(psnr_nms)

            chamfer = chamfer_loss(gts=input_pts, preds=torch.from_numpy(reconstruction).cuda().unsqueeze(0)).tolist()
            chamfer_nms = chamfer_loss(gts=input_pts, preds=torch.from_numpy(reconstruction_nms).cuda().unsqueeze(0)).tolist()
            chamfer_list.append(chamfer[0])
            chamfer_list_nms.append(chamfer_nms[0])
            n_components.append(pi[i].shape[0])
            n_components_nms.append(pi_nms[i].shape[0])

        # write log file
        log(log_dir, chamfer_list, chamfer_list_nms, psnr_list, psnr_list_nms, n_components, n_components_nms)