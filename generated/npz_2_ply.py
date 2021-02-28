import argparse
import numpy as np
from utils import *
from visualization import *

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

# color palette for nyu40 labels
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


def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--input', type=str, default='./data/scans/scannet/scene0004_00_vh_clean_2.ply')
    parser.add_argument('--id', type=str, default='scene4')
    parser.add_argument('--num_pts', help='number of input points', default=50000, type=int)
    parser.add_argument('--output_dir', type=str, default='./data/generated_data/scannet/')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args
    args = parse_args()
    print('Called with args:')
    print(args)
    
    # import data
    pts = read_ply_open3D(args.input)
    pts = random_choice(pts, args.num_pts)

    # output ply
    output_scan_dir = args.output_dir + args.id + '/'
    if not os.path.exists(output_scan_dir):
        os.makedirs(output_scan_dir)
    output_file = output_scan_dir + str(pts.shape[0]) + '.ply'
    write_ply(verts=pts, colors=None, indices=None, output_file=output_file)