# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

import numpy as np
import torch
from scipy.spatial import ConvexHull
from numpy import *
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import kaolin as kal
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids as mesh2voxel

def set_axes_equal(ax):
    '''
    Source: https://stackoverflow.com/a/31364297
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]                # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],                # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],                # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],                        # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)

def get_corners(x,y,z):
    x_corners = x[np.r_[0:4,5:9]]
    y_corners = y[np.r_[0:4,5:9]]
    z_corners = z[np.r_[0:4,5:9]]
    corners_3d = np.vstack([x_corners,y_corners,z_corners]).T
    return torch.tensor(corners_3d)

def cuboid_faces(k):
    bottom = [[0,1,2], [0,2,3]]
    upper = [[4,5,6], [4,6,7]]
    front = [[0,1,5], [0,5,4]]
    back = [[3,2,6], [3,6,7]]
    left = [[0,4,7], [0,7,3]]
    right = [[1,2,6], [1,6,5]]
    return torch.tensor(bottom + upper + front + back + left + right).cuda() + 8 * k 

def plot_gmm(ax, mix, mu, cov, color=None, cmap='Spectral', azim=60, elev=0, numWires=5, wireframe=True):
    if color is None:
        color = np.arange(mix.shape[0]) / (mix.shape[0] - 1)
    if cmap is not None:
        cmap = cm.get_cmap(cmap)
        color = cmap(color)

    # u = np.linspace(0.0, 2.0 * np.pi, numWires)
    # v = np.linspace(0.0, np.pi, numWires)
    # X = np.outer(np.cos(u), np.sin(v))
    # Y = np.outer(np.sin(u), np.sin(v))
    # Z = np.outer(np.ones_like(u), np.cos(v)) 
    # XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
    # print(X.shape, Y.shape, Z.shape)
    # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    alpha = mix / mix.max()
    ax.view_init(azim=azim, elev=elev)
    # numWires = 2
    for k in range(mix.shape[0]):
        #print(mix[k])
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        X, Y, Z = cuboid_data(mu[k], (3, 3, 3))
        XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        x, y, z = V.T @ (np.sqrt(s)[:, None] * XYZ) + mu[k][:, None]
        #print(x.shape)
        x = x.reshape(4, numWires)
        y = y.reshape(4, numWires)
        z = z.reshape(4, numWires)
        if wireframe:
            ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
        else:
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])#

def plot_box(points_group):#
    # points_group = points_group.numpy()
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(points_group)):
        points = points_group[i]
        #points[:,[1, 2]] = points[:,[2, 1]]
        # P = [[2.06498904e-01 , -6.30755443e-07 ,  1.07477548e-03],
        # [1.61535574e-06 ,  1.18897198e-01 ,  7.85307721e-06],
        # [7.08353661e-02 ,  4.48415767e-06 ,  2.05395893e-01]]
        #Z = np.zeros((8,3))
        #for i in range(8): Z[i,:] = np.dot(points[i,:],P)
        #Z = 10.0*Z
        #Z = points


        #r = [-1,1]

        #X, Y = np.meshgrid(r, r)
        # plot vertices
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])

        # list of sides' polygons of figure
        verts = [[points[0],points[1],points[2],points[3]],
        [points[4],points[5],points[6],points[7]], 
        [points[0],points[1],points[5],points[4]], 
        [points[2],points[3],points[7],points[6]], 
        [points[1],points[2],points[6],points[5]],
        [points[4],points[7],points[3],points[0]]]

        # plot sides
        ax.add_collection3d(Poly3DCollection(verts, 
        facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # equal axis scale in 3d: https://stackoverflow.com/a/21765085
    pts_stack = np.vstack(points_group)
    max_range = np.array([pts_stack[:, 0].max()-pts_stack[:, 0].min(), 
                        pts_stack[:, 1].max()-pts_stack[:, 1].min(), 
                        pts_stack[:, 2].max()-pts_stack[:, 2].min()]).max() / 2.0

    mid_x = (pts_stack[:, 0].max()+pts_stack[:, 0].min()) * 0.5
    mid_y = (pts_stack[:, 1].max()+pts_stack[:, 1].min()) * 0.5
    mid_z = (pts_stack[:, 2].max()+pts_stack[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #plt.axis('scaled')
    set_axes_equal(ax)
    plt.show()

def non_max_suppression_3d(gms, nms_th=0.25, conf_th=0.001):
    matplotlib.use( 'tkagg' )
    pi, mu, cov, SVD = gms
    s = SVD['s']
    V = SVD['V']
    box = []
    for k in range(pi.shape[0]):
        #print("before flatten: ", s)
        min_pc = np.argmin(s[k])
        if s[k, min_pc] < 0.005:
            s[k, min_pc] = 0.005
        #print("after flatten: ", s)
        X, Y, Z = cuboid_data([0,0,0], [1,1,1])
        XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        # x, y, z = V[k].T @ (np.sqrt(5.99*s[k])[:, None] * XYZ) + mu[k][:, None]
        x, y, z = V[k].T @ (1.2*np.sqrt(5.99*s[k])[:, None] * XYZ) + mu[k][:, None]
        box.append(get_corners(x,y,z))
    box = torch.stack(box) # k x 8 x 3

    if len(box) == 0:
        return box

    sorted_gm = np.argsort(-pi)
    box = box[sorted_gm].cuda()
    pi_sorted = pi[sorted_gm]
    bboxes = [0]

    # Normalize all vertices to [0,1]
    box_temp = box.view(-1, 3)
    min, _ = torch.min(box_temp, dim=0)
    box = (box - min)
    box_temp = box.view(-1, 3)
    max = torch.max(box_temp)
    box = box / max

    print("Generating voxels...")
    origin = torch.zeros((1, 3)).cuda()
    scale = torch.ones((1)).cuda()
    vox_list = mesh2voxel(box, cuboid_faces(0), 30, origin, scale)
    vox_list = kal.ops.voxelgrid.fill(vox_list.cpu()).cuda()
    print("Voxels generated!")

    # logs_path = './usd/'
    # timelapse = kal.visualize.Timelapse(logs_path)
    # timelapse.add_voxelgrid_batch(
    #     iteration=0,
    #     category='debug_all_voxel',
    #     voxelgrid_list=all_vox # K x N x N x N
    # )

    for i in np.arange(1, len(vox_list)):
        vox_gt = vox_list[i]
        flag = 1
        # print("Testing instance" + str(i))
        for j in range(len(bboxes)):

            vox_pred = vox_list[bboxes[j]]
            vox_union = torch.logical_or(vox_gt, vox_pred)

            vox_intersection = torch.logical_and(vox_gt, vox_pred)
            area_inter = torch.sum(vox_intersection)
            area_union = torch.sum(vox_union)
            IOU_3d = area_inter / area_union
            # print("Testing instance" + str(i) + ", IOU_3d = "+ str(IOU_3d))

            logs_path = './usd/'
            timelapse = kal.visualize.Timelapse(logs_path)
            # timelapse.add_voxelgrid_batch(
            #     iteration=0,
            #     category='debug_voxel_union',
            #     voxelgrid_list=[vox_union.squeeze()] # K x N x N x N
            # )

            # timelapse.add_voxelgrid_batch(
            #     iteration=0,
            #     category='debug_voxel_gt',
            #     voxelgrid_list=[vox_gt.squeeze()] # K x N x N x N
            # )

            # timelapse.add_voxelgrid_batch(
            #     iteration=0,
            #     category='debug_voxel_pred',
            #     voxelgrid_list=[vox_pred.squeeze()] # K x N x N x N
            # )
            # plot_box([box[bboxes[j]].cpu().numpy(), box[i].cpu().numpy()])

            if IOU_3d > nms_th or pi_sorted[bboxes[j]] < conf_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(i)

    sorted_bboxes = np.asarray(bboxes, np.int32)
    # print("######final bboxe######: ", sorted_bboxes.shape[0])
    # plot_box(box[sorted_bboxes].cpu().numpy())
    true_bboxes = sorted_gm[sorted_bboxes]
    SVD['U'] = SVD['U'][true_bboxes]
    SVD['s'] = SVD['s'][true_bboxes]
    SVD['V'] = SVD['V'][true_bboxes]

    return pi[true_bboxes], mu[true_bboxes], cov[true_bboxes], SVD

# 3D-IoU-Python: https://github.com/AlienCat-K/3D-IoU-Python.git
      