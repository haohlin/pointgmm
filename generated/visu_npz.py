import numpy as np
import argparse
import os
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

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

def plot_gmm(ax, mix, mu, cov, color=None, cmap='Spectral', azim=60, elev=0, numWires=15, wireframe=True):
    if color is None:
        color = np.arange(mix.shape[0]) / (mix.shape[0] - 1)
    if cmap is not None:
        cmap = cm.get_cmap(cmap)
        color = cmap(color)

    u = np.linspace(0.0, 2.0 * np.pi, numWires)
    v = np.linspace(0.0, np.pi, numWires)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v)) 
    XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])

    alpha = mix / mix.max()
    ax.view_init(azim=azim, elev=elev)

    for k in range(mix.shape[0]):
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        x, y, z = V.T @ (np.sqrt(s)[:, None] * XYZ) + mu[k][:, None]
        #print(x.shape)
        x = x.reshape(numWires, numWires)
        #print(x.shape)
        y = y.reshape(numWires, numWires)
        z = z.reshape(numWires, numWires)
        if wireframe:
            ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
        else:
            ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])

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
    #parser.add_argument('--input', type=str, default='generated/chair_vae/eval_points/hgmms/hgmms_000_00.npz')
    parser.add_argument('--obj_class', type=str, default='chair')
    parser.add_argument('--id', type=str, default='1')
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
    npz_data = np.load("generated/" + args.obj_class + "_vae/eval_points/hgmms/hgmms_" + args.id + ".npz")
    input_pts = npz_data['input_points']
    sample_pts = npz_data['sample_points']
    splits = npz_data['splits']
    pts_seg = npz_data['pts_seg']
    pi = npz_data['pi']
    mu = npz_data['mu']
    sigma = npz_data['sigma']
    print("number of clusters: ", pi.shape[0])

    color_palette = create_color_palette()
    newcolors = [np.asarray(color_palette[i])/255 for i in range(45)]
    newcmp = ListedColormap(newcolors)

    #colors = np.zeros((sample_pts.shape[0],3))
    sample_color = np.zeros(sample_pts.shape[0])
    vert_sample_colors = np.zeros((sample_pts.shape[0], 3))
    for i in range(splits.shape[0]-1):
        start = splits[i]
        end = splits[i+1]
        sample_color[start:end] = float(i)/float(splits.shape[0]-1)
        vert_sample_colors[start:end] = color_palette[i % 45]
        #colors[start:end] = npz_data["palette"][i]
    

    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(121, projection='3d')
    #pts_seg = [np.asarray(create_color_palette()[pts_seg[i]%45])/255 for i in range(pts_seg.shape[0])]
    seg_color = pts_seg / (splits.shape[0]-1)

    plot_pcd(ax, input_pts, color=seg_color, cmap=newcmp)
    #plot_gmm(ax, pi, mu, sigma, cmap=newcmp)
    ax.set_title("point cloud segmentation")

    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, sample_pts, color=sample_color, cmap=newcmp)
    plot_gmm(ax, pi, mu, sigma, cmap=newcmp, wireframe=False)#, color=npz_data["palette"])
    ax.set_title("point cloud (with GMM)")

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