import argparse
import os
import torch
import numpy as np
import kaolin as kal
import kaolin.ops.mesh as kmesh
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

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

def create_color_palette():
    return [
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (0, 0, 0),
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (0, 0, 0),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (0, 0, 0),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (0, 0, 0),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144),
       (0, 0, 0)
    ]

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
    return torch.tensor(bottom + upper + front + back + left + right) + 8 * k 

def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return np.array(x), np.array(y), np.array(z)


def plot_gmm(ax, mix, mu, cov, color=None, cmap='Spectral', azim=60, elev=0, numWires=15, wireframe=True, plot_box=True):
    if color is None:
        color = np.arange(mix.shape[0]) / (mix.shape[0] - 1)
    if cmap is not None:
        cmap = cm.get_cmap(cmap)
        color = cmap(color)

    if not plot_box:
        u = np.linspace(0.0, 2.0 * np.pi, numWires)
        v = np.linspace(0.0, np.pi, numWires)
        X = np.outer(np.cos(u), np.sin(v))
        Y = np.outer(np.sin(u), np.sin(v))
        Z = np.outer(np.ones_like(u), np.cos(v)) 

        # HALTED TODO: visualize ellipsoids in Kaolin
        # X = sphere_mesh.vertices[:, 0]
        # Y = sphere_mesh.vertices[:, 1]
        # Z = sphere_mesh.vertices[:, 2]

        XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        numWires_x = numWires
    alpha = mix / mix.max()
    ax.view_init(azim=azim, elev=elev)
    vertices = []
    faces = []
    voxels = []
    for k in range(mix.shape[0]):
        # skip if no points in instance k
        if not mix[k]: 
            continue
        
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        if plot_box:
            X, Y, Z = cuboid_data([0,0,0], [1,1,1])#mu[k]
            XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
            numWires_x = 4
            numWires = 5
            xyz_stretch = 1.2*np.sqrt(5.99*s)[:, None] * XYZ
        else:
            # xyz_stretch = 1.2*np.sqrt(5.99*s)[:, None] * XYZ
            xyz_stretch = np.sqrt(1.5*s)[:, None] * XYZ
        xyz = V.T @ (xyz_stretch) + mu[k][:, None]#V.T
        x, y, z = xyz
        if plot_box:
            corners = get_corners(x, y, z)
            """
            HALTED TODO:Calculate corners based on the furtherst point along x, y axis
            """
            
            box_faces = cuboid_faces(k)
            vertices.append(corners)
            faces.append(box_faces)
            # vox = kal.ops.conversions.trianglemeshes_to_voxelgrids(corners.unsqueeze(0), cuboid_faces(0), 20).squeeze(0) # N x N x N
            # voxels.append(vox) # K x N x N x N
        else:
            vertices.append(torch.from_numpy(xyz.T))
            faces.append(sphere_mesh.faces + sphere_mesh.vertices.shape[0] * k)
        x = x.reshape(numWires_x, numWires)
        y = y.reshape(numWires_x, numWires)
        z = z.reshape(numWires_x, numWires)

        # ax.quiver(mu[k,0], mu[k,1], mu[k,2], V[:,0], V[:,1], V[:,2], color="r", length=0.5, normalize=True)
        if wireframe:
            # ax.scatter(X.flatten(), Y.flatten(), Z.flatten())
            ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
        else:
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
    # ax.set_axis_off()
    if plot_box:
        vertices = torch.cat(vertices, dim=0) # k x 8 x 3 -> 8k x 3
        faces = torch.cat(faces, dim=0) # k x 12 x 3 -> 12k x 3
        # voxels = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices.unsqueeze(0), faces, 80).squeeze() # N x N x N
        # voxels = kal.ops.voxelgrid.fill(voxels).squeeze()
        # timelapse.add_mesh_batch(
        #     iteration=0,
        #     category='box_table',
        #     vertices_list=[vertices],
        #     faces_list=[faces]
        # )    
        # timelapse.add_voxelgrid_batch(
        #     iteration=0,
        #     category='voxel_table',
        #     voxelgrid_list=[voxels] # K x N x N x N
        # )    
    else:
        vertices = torch.cat(vertices, dim=0) # k x n_verts x 3 -> 8k x 3
        faces = torch.cat(faces, dim=0) # k x 12 x 3 -> 12k x 3
        # timelapse.add_mesh_batch(
        #     iteration=0,
        #     category='hgmm_airplane_L3',
        #     vertices_list=[vertices],
        #     faces_list=[faces]
        # )    

def plot_pcd(ax, pcd, color=None, cmap='Spectral', size=4, alpha=0.9, azim=60, elev=0):
    if color is None:
        color = pcd[:, 0]
        vmin = -2
        vmax = 1.5
    else:
        vmin = 0
        vmax = 1
    ax.view_init(azim=azim, elev=elev)
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color, s=size, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    min_lim = min(pcd.min() * 0.9, lims.min())
    max_lim = max(pcd.max() * 0.9, lims.max())
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((min_lim, max_lim))
    ax.set_axis_off()


def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='HGMM and segmentation visualization')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--obj_class', type=str, default='chair')
    parser.add_argument('--id', type=str, default='000_03')
    parser.add_argument('--save_ply', action='store_true')
    #parser.add_argument('--num_pts', help='number of input points', default=50000, type=int)
    parser.add_argument('--output_dir', type=str, default='generated/ply/')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args
    args = parse_args()
    print('Called with args:')
    print(args)
    
    # import data
    if args.input:
        npz_data = np.load(args.input)
    else:
        npz_data = np.load("generated/" + args.obj_class + "_vae/eval_points/hgmms/hgmms_" + args.id + ".npz")
    input_pts = npz_data['input_points']
    pts_seg = npz_data['pts_seg']
    pi = npz_data['pi'][155:780]
    print(sum(pi))
    mu = npz_data['mu'][155:780]
    sigma = npz_data['sigma'][155:780]
    print("number of clusters: ", pi.shape[0])

    color_palette = create_color_palette()
    newcolors = [np.asarray(color_palette[i])/255 for i in range(45)]
    newcmp = ListedColormap(newcolors)

    #colors = np.zeros((sample_pts.shape[0],3))
    # sample_color = np.zeros(sample_pts.shape[0])
    # vert_sample_colors = np.zeros((sample_pts.shape[0], 3))
    # for i in range(splits.shape[0]-1):
    #     start = splits[i]
    #     end = splits[i+1]
    #     sample_color[start:end] = float(i)/float(splits.shape[0]-1)
    #     vert_sample_colors[start:end] = color_palette[i % 45]
        #colors[start:end] = npz_data["palette"][i]
    
    fig = plt.figure(figsize=(20, 20))

    #pts_seg = [np.asarray(create_color_palette()[pts_seg[i]%45])/255 for i in range(pts_seg.shape[0])]
    seg_color = pts_seg / (pi.shape[0]-1)

    logs_path = './logs/'
    # npz_airplane = np.load("generated/airplane_vae/eval_points/hgmms/hgmms_" + args.id + ".npz")
    # npz_table = np.load("generated/table_vae/eval_points/hgmms/hgmms_" + args.id + ".npz")
    npz_chair = np.load("generated/chair_vae/eval_points/hgmms/hgmms_" + args.id + ".npz")
    # airplane_pts = npz_airplane['input_points']
    # table_pts = npz_table['input_points']
    chair_pts = torch.from_numpy(npz_chair['input_points']) 

    timelapse = kal.visualize.Timelapse(logs_path)
    # voxels = kal.ops.conversions.pointclouds_to_voxelgrids(chair_pts.unsqueeze(0)).squeeze() # N x N x N
    # timelapse.add_voxelgrid_batch(
    #     iteration=0,
    #     category='voxel_chair',
    #     voxelgrid_list=[voxels] # K x N x N x N
    # )
    # timelapse.add_pointcloud_batch(
    #     iteration=1,
    #     pointcloud_list=[torch.from_numpy(airplane_pts * 10), 
    #                      torch.from_numpy(table_pts), 
    #                      torch.from_numpy(chair_pts)
    #                     ]
    #     # semantic_ids=[torch.from_numpy(pts_seg)]
    # )

    sphere_mesh = kal.io.obj.import_mesh('./generated/sphere.obj', with_materials=True)
    # the sphere is usually too small (this is fine-tuned for the clock)
    # vertices = mesh.vertices.cuda() * 75
    # faces = mesh.faces
    # timelapse.add_mesh_batch(
    #     iteration=0,
    #     # category='optimized_mesh',
    #     vertices_list=[vertices],
    #     faces_list=[mesh.faces]
    # )

    # ax = fig.add_subplot(121, projection='3d')
    # plot_pcd(ax, input_pts, color=seg_color, cmap=newcmp)
    # # plot_gmm(ax, pi, mu, sigma, cmap=newcmp, wireframe=True, plot_box=True)#, color=npz_data["palette"])
    # ax.set_title("point cloud segmentation")

    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, input_pts, cmap=newcmp)
    plot_gmm(ax, pi, mu, sigma, cmap=newcmp, wireframe=True, plot_box=False)#, color=npz_data["palette"])
    ax.set_title("point cloud (with GMM)")
    set_axes_equal(ax)
    plt.tight_layout()

    plt.show()

    # output ply
    if args.save_ply: 
        output_dir = args.output_dir + args.obj_class
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vert_seg_colors = [color_palette[pts_seg[i]%45] for i in range(pts_seg.shape[0])]
        seg_output_file = output_dir + '/seg_' + args.id + '.ply'
        write_ply(verts=input_pts, colors=vert_seg_colors, indices=None, output_file=seg_output_file)

        sample_output_file = output_dir + '/sample_' + args.id + '.ply'
        write_ply(verts=sample_pts, colors=vert_sample_colors, indices=None, output_file=sample_output_file)