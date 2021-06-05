import os
import sys
import torch
import argparse
import kaolin as kal
import numpy as np
from loss_functions import *
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))

from models.primitives import *
from fitters.cuboid_fitter import CuboidFitter
from show.view_utils import set_axes_equal

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib



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

def log(log_dir, chamfer_list, psnr_list, n_components, iou_list):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename_chamfer = f'{log_dir}chamfer_merged.txt'
    filename_psnr = f'{log_dir}psnr_merged.txt'
    filename_iou = f'{log_dir}iou_merged.txt'
    # filename_psnr_nms = f'{log_dir}psnr_nms.txt'
    # filename_time = f'{log_dir}{chair}_{id:03d}_time.txt'
    f1 = open(filename_chamfer, 'a')
    f2 = open(filename_psnr, 'a')
    f3 = open(filename_iou, 'a')
    # f4 = open(filename_psnr_nms, 'a')

    for i in range(len(chamfer_list)):
        f1.write(f'{n_components[i]}:{chamfer_list[i]}\t')
        f2.write(f'{n_components[i]}:{psnr_list[i]}\t')
        f3.write(f'{n_components[i]}:{iou_list[i]}\t')
    f1.write('\n')
    f2.write('\n')
    f3.write('\n')
    # f4.write('\n')
    f1.close()
    f2.close()
    f3.close()
    # f4.close()

def build_gmm(pi, mu, sigma):
    gmm = GaussianMixture(n_components=pi.shape[0])
    gmm.weights_ = pi              # K
    gmm.means_ = mu                # Kx3
    gmm.covariances_ = sigma       # Kx3x3
    return gmm

def norm_verts(verts):
    # Normalize all vertices to [0,1]
    verts_temp = verts.view(-1, 3)
    min, _ = torch.min(verts_temp, dim=0)
    verts = (verts - min)
    verts_temp = verts.view(-1, 3)
    max = torch.max(verts_temp)
    verts = verts / max
    return verts

def mesh_iou(vertices_1, faces_1, vertices_2, faces_2):
    vertices_1 = norm_verts(vertices_1)
    vertices_2 = norm_verts(vertices_2)
    origin = torch.zeros((1, 3)).cuda()
    scale = torch.ones((1)).cuda()
    # point cloud to voxel grids
    vox_gt = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices_1, faces_1, 30, origin, scale)
    vox_pred = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices_2, faces_2, 30, origin, scale)
    vox_gt = kal.ops.voxelgrid.fill(vox_gt.cpu()).cuda()
    vox_pred = kal.ops.voxelgrid.fill(vox_pred.cpu()).cuda()

    vox_union = torch.logical_or(vox_gt, vox_pred)
    vox_intersection = torch.logical_and(vox_gt, vox_pred)
    area_inter = torch.sum(vox_intersection)
    area_union = torch.sum(vox_union)
    iou = area_inter / area_union

    logs_path = './logs/debug/'
    timelapse = kal.visualize.Timelapse(logs_path)
    timelapse.add_voxelgrid_batch(
        iteration=0,
        category='debug_voxel_union',
        voxelgrid_list=[vox_union.squeeze()] # K x N x N x N
    )

    timelapse.add_voxelgrid_batch(
        iteration=0,
        category='debug_voxel_gt',
        voxelgrid_list=[vox_gt.squeeze()] # K x N x N x N
    )

    timelapse.add_voxelgrid_batch(
        iteration=0,
        category='debug_voxel_pred',
        voxelgrid_list=[vox_pred.squeeze()] # K x N x N x N
    )
    # plot_box([box[bboxes[j]].cpu().numpy(), box[i].cpu().numpy()])


    return iou


def box_2_mesh(box_list):
    # logs_path = './logs/'
    # timelapse = kal.visualize.Timelapse(logs_path)

    vertices = []
    faces = []
    # voxels = []
    for k, box in enumerate(box_list):
        box_verts, box_faces = box.get_mesh(k)
        vertices.append(box_verts)
        faces.append(box_faces)

    vertices = torch.cat(vertices, dim=0).cuda() # k x 8 x 3 -> 8k x 3
    faces = torch.cat(faces, dim=0).cuda() # k x 12 x 3 -> 12k x 3
    # voxels = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices.unsqueeze(0), faces, 80).squeeze() # N x N x N
    # voxels = kal.ops.voxelgrid.fill(voxels).squeeze()
    # timelapse.add_mesh_batch(
    #     iteration=0,
    #     category=f'box_',
    #     vertices_list=[vertices],
    #     faces_list=[faces]
    # )    
    # timelapse.add_voxelgrid_batch(
    #     iteration=0,
    #     category='hgmm_NOnms_' + cls + '_L' + str(level),
    #     voxelgrid_list=[voxels] # K x N x N x N
    # ) 
    return vertices, faces

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Chamfer distance / PSNR calculator')
    parser.add_argument('--input', type=str, default='generated/chair_vae/eval_points/hgmms/')
    parser.add_argument('--obj_class', type=str, default='chair')
    parser.add_argument('--id', type=str, default='01')
    parser.add_argument('--output_dir', type=str, default='logs/primitive/')

    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    # args
    args = parse_args()
    print('Called with args:')
    print(args)
    
    # generate ply file for input point cloud
    # write_ply(verts=input_pts.squeeze(0).cpu(), colors=None, indices=None, output_file=f'./experiment/{args.obj_class}_{args.id}.ply')
    
    # sample points and calculate chamfer loss
    # log_dir = args.output_dir
    # obj_class = args.obj_class
    # obj_id = args.id

    for s in range(10):
        # import data
        npz_data = np.load(f'{args.input}hgmms_{s:03d}.npz', allow_pickle=True)
        input_pts = torch.from_numpy(npz_data['input_points']).cuda().unsqueeze(0)
        # input_pts = input_pts.repeat(4,1,1)
        pts_seg = torch.from_numpy(npz_data['pts_seg']).cuda()
        pi = npz_data['pi']
        mu = npz_data['mu']
        sigma = npz_data['sigma']
        pi_nms = npz_data['pi_nms']
        mu_nms = npz_data['mu_nms']
        sigma_nms = npz_data['sigma_nms']
        cuboid_list_merged = npz_data['cuboid_list']
        cuboid_list = npz_data['cuboid_list_merged']
        input_mesh = npz_data['input_mesh'][0]
        input_vertices = input_mesh['vertices'].cuda()
        input_faces = input_mesh['faces'].cuda()

        log_dir = f'{args.output_dir}{args.obj_class}_{args.id}/'
        max_dist = max_diag_dist(npz_data['input_points'])

        psnr_list = []
        # psnr_list_nms = []
        chamfer_list = []
        # chamfer_list_nms = []
        # n_components = []
        # n_components_nms = []
        n_components_merged = []
        iou_list = []
        for i in range(len(pi)):
            box_list = []
            box_merged_list = []

            # matplotlib.use( 'tkagg' )
            # fig = plt.figure(figsize=(20, 20))
            # ax = fig.add_subplot(121, projection='3d')
            # ax.view_init(azim=60, elev=0)
            # ax.scatter(input_pts[0, :, 0].cpu(), input_pts[0, :, 1].cpu(), input_pts[0, :, 2].cpu())
            
            # for k, (box_k, box_k_merged) in enumerate(zip(cuboid_list[i], cuboid_list_merged[i])):
            #     box_model = Cuboid(center=box_k["center"], 
            #                     x_axis=box_k["x_axis"], y_axis=box_k["y_axis"], z_axis=box_k["z_axis"], 
            #                     x_range=box_k["x_range"],y_range=box_k["y_range"],z_range=box_k["z_range"], 
            #                     obj_conf=box_k["obj_conf"])
            #     box_merged_model = Cuboid(center=box_k_merged["center"], 
            #                     x_axis=box_k_merged["x_axis"], y_axis=box_k_merged["y_axis"], z_axis=box_k_merged["z_axis"], 
            #                     x_range=box_k_merged["x_range"],y_range=box_k_merged["y_range"],z_range=box_k_merged["z_range"], 
            #                     obj_conf=box_k_merged["obj_conf"])
            #     box_list.append(box_model)
            #     box_merged_list.append(box_merged_model)
            #     # box_model.plot(ax)

            verts, faces = box_2_mesh(cuboid_list[i])
            # verts, faces = box_2_mesh(box_list)
            iou = mesh_iou(input_vertices.unsqueeze(0), input_faces, verts.unsqueeze(0), faces)
            iou_list.append(iou)
            pts = kal.ops.mesh.sample_points(verts.unsqueeze(0), faces, input_pts.shape[1])[0].squeeze(0)
            # print(f'input: {np.max(npz_data["input_points"])}:{np.min(npz_data["input_points"])}')
            # print(f'pred: {torch.max(pts)}:{torch.min(pts)}')
            # ax = fig.add_subplot(122, projection='3d')
            # ax.scatter(pts[:, 0].cpu(), pts[:, 1].cpu(), pts[:, 2].cpu())

            # gmm = build_gmm(pi[i], mu[i], sigma[i])
            # gmm_nms = build_gmm(pi_nms[i], mu_nms[i], sigma_nms[i])
            # reconstruction, labels = gmm.sample(input_pts.shape[1])
            # reconstruction_nms, labels_nms = gmm_nms.sample(input_pts.shape[1])

            chamfer = chamfer_loss(gts=input_pts, preds=pts.unsqueeze(0)).tolist()[0]
            # chamfer_nms = chamfer_loss(gts=input_pts, preds=torch.from_numpy(reconstruction_nms).cuda().unsqueeze(0)).tolist()
            chamfer_list.append(chamfer)
            # chamfer_list_nms.append(chamfer_nms[0])

            psnr = PSNR(pts.cpu().numpy(), npz_data['input_points'], max_dist)
            # psnr_nms = PSNR(reconstruction_nms, npz_data['input_points'], max_dist)
            psnr_list.append(psnr)
            # psnr_list_nms.append(psnr_nms)

            # n_components.append(pi[i].shape[0])
            # n_components_nms.append(pi_nms[i].shape[0])
            n_components_merged.append(len(cuboid_list[i]))
            print(f'Level: {i}, IoU: {iou}, chamfer: {chamfer}, psnr: {psnr}, n_primitives: {len(cuboid_list[i])}')

            # set_axes_equal(ax)
            # plt.show()

        # write log file
        log(log_dir, chamfer_list, psnr_list, n_components_merged, iou_list)
        # log(log_dir, chamfer_list, chamfer_list_nms, psnr_list, psnr_list_nms, n_components, n_components_nms)