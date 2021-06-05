import models.model_factory as factory
import models.gm_utils as gm_utils
import models.nms as nms
import kaolin as kal

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from fitters.plane_fitter import PlaneFitter
from fitters.sphere_fitter import SphereFitter
from fitters.cuboid_fitter import CuboidFitter
from options import TrainOptions, Options
from constants import OUT_DIR
from show.view_utils import view
from custom_types import *
from process_data.mesh_loader import get_loader, AnotherLoaderWrap
from process_data.files_utils import collect, init_folders
from show.view_utils import set_axes_equal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

class ViewMem:

    @staticmethod
    def get_palette(num_colors: int) -> list:
        if num_colors == 1:
            return [.45]
        if num_colors not in ViewMem.colors:
            ViewMem.colors[num_colors] = [ViewMem.color_map(float(idx) / (num_colors - 1)) for idx in range(num_colors)]
        return ViewMem.colors[num_colors]

    colors = {}
    color_map =  plt.cm.get_cmap('Spectral')
    device = CPU
    ds_size = 1
    max_items = 8
    points_in_sample = 4096#2048
    loader = None
    memory = None
    save_separate = {'sample', 'byid'}
    last_idx = None


def create_palettes(splits: list) -> list:
    palette = []
    for split in splits:
        num_colors = len(split) - 1
        palette.append(ViewMem.get_palette(num_colors))
    return palette


def save_pil_image(image, path, prefix: str, start_counts:int, _) -> int:
    image.save(f'{path}{prefix}_{start_counts:03d}.png')
    return 1


def save_np_points(points_group, path: str, prefix: str, start_counts:int, trace: ViewMem) -> int :
    # for i, group in enumerate(zip(*points_group)):
    #     if prefix in trace.save_separate:
    #         saving_path = f'{path}{prefix}_{start_counts + i:03d}.npz'
    #     else:
    #         saving_path = f'{path}{prefix}_{start_counts:03d}_{i:02d}.npz'
    #     np.savez_compressed(saving_path, input_points=points_group[0], sample_points=group[1], splits=group[2], 
    #                         pts_seg=group[3], palette=group[4], pi=group[5], mu=group[6], sigma=group[7])
    
    saving_path = f'{path}{prefix}_{start_counts:03d}.npz'
    np.savez_compressed(saving_path, input_points=points_group[0], pi=points_group[1], mu=points_group[2], 
                                     sigma=points_group[3], pts_seg=points_group[4], pi_nms=points_group[5], 
                                     mu_nms=points_group[6], sigma_nms=points_group[7], cuboid_list = points_group[8], 
                                     cuboid_list_merged = points_group[9], input_mesh = points_group[10])

    if prefix in trace.save_separate:
        return len(points_group[0])
    return 1


def saving_handler(args: Options, trace: ViewMem):

    def init_prefix(prefix: str, saving_index: int, saving_folder: str):
        nonlocal saving_dict, suffix
        same_type_file = collect(saving_folder, suffix[saving_index], prefix=prefix)
        if len(same_type_file) == 0:
            saving_dict[saving_index][prefix] = 0
        else:
            last_number_file_name = same_type_file[-1][1]
            saving_dict[saving_index][prefix] = int(last_number_file_name.split('_')[1]) + 1

    def get_path(prefix: str, saving_index:int) -> str:
        nonlocal saving_dict
        saving_folder = f'{saving_folders[saving_index]}{prefix}/'
        if prefix not in saving_dict[saving_index]:
            init_folders(f'{saving_folders[saving_index]}/{prefix}/')
            init_prefix(prefix, saving_index, saving_folder)
        # saving_dict[saving_index][prefix] += 1
        return saving_folder


    def handle(prefix: str, *items):
        nonlocal saving_dict
        msg = '0: continue | 1: save image | 2: save points | 3: save both '
        to_do = get_integer((0, 4), msg)
        if to_do > 0:
            to_do = to_do - 1
            for i in range(len(saving_f)):
                if to_do == i or to_do == len(saving_f):
                    path = get_path(prefix, i)
                    saving_dict[i][prefix] += saving_f[i](items[i], path, prefix, saving_dict[i][prefix], trace)

    saving_dict = [dict(), dict()]
    saving_folders = [f'{OUT_DIR}/{args.info}/eval_images/', f'{OUT_DIR}/{args.info}/eval_points/']
    saving_f = [save_pil_image, save_np_points]
    suffix = ['.png', '.npz']
    return handle


def init_loader(args, trace: ViewMem):
    if trace.loader is None:
        trace.loader = AnotherLoaderWrap(get_loader(args), trace.max_items)


def get_z_by_id(encoder, args: Options, num_items: int, idx, trace: ViewMem):
    init_loader(args, trace)
    if idx is None and trace.last_idx is None:
        inds, data = trace.loader.get_random_batch()
        trace.last_idx = [int(index) for index in inds[:num_items]]
    else:
        data, mesh = trace.loader.get_by_ids(*idx)
    input_points = data[:num_items].to(trace.device)
    input_mesh = mesh
    z, _, _ = encoder(input_points)
    return input_points, z, input_mesh


def get_integer(allowed_range: tuple, msg: str='') -> int:
    if msg == '':
        msg = f'\tPlease choose number of objects to show from {allowed_range[0]} to {allowed_range[1] -1}\n\t'
    while (True):
        try:
            integer = int(input(msg))
            if allowed_range[0] <= integer < allowed_range[1]:
                break
            else:
                raise ValueError
        except ValueError:
            print('Unexpected argument, please try again')
    return integer

def sample(_, decoder, args: Options, trace: ViewMem):
    if trace.memory is None:
        num_items = get_integer((1, 8))
    else:
        num_items = trace.memory[1]
    z = torch.randn(num_items, args.dim_z).to(trace.device)
    gms = decoder(z)
    vs, splits = gm_utils.hierarchical_gm_sample(gms, trace.points_in_sample, False)
    vs = vs.cpu().numpy()
    splits = [s for s in splits]
    palette = create_palettes(splits)
    im, points = view([vs_ for vs_ in vs], splits, palette)
    return True, (sample, num_items), im, (points, splits, palette)

def get_box_corners(x,y,z):
    x_corners = x[np.r_[0:4,5:9]]
    y_corners = y[np.r_[0:4,5:9]]
    z_corners = z[np.r_[0:4,5:9]]
    corners_3d = np.vstack([x_corners,y_corners,z_corners]).T
    return torch.tensor(corners_3d)

def plane_faces(k):
    faces = [[0,1,2], [0,1,3], [0,2,3]]
    return torch.tensor(faces) + 4 * k 

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

def plane_2_usd(plane_list, level, azim=60, elev=0):
    logs_path = './logs/'
    timelapse = kal.visualize.Timelapse(logs_path)

    vertices = []
    faces = []
    for k, plane in enumerate(plane_list):        
        x,y,z = plane.get_corners()
        corners = torch.from_numpy(np.vstack([x.flatten(), y.flatten(), z.flatten()]).T) 
        face = plane_faces(k)
        vertices.append(corners)
        faces.append(face)

    vertices = torch.cat(vertices, dim=0) # k x 4 x 3 -> 4k x 3
    faces = torch.cat(faces, dim=0) # k x 2 x 3 -> 2k x 3
    # voxels = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices.unsqueeze(0), faces, 80).squeeze() # N x N x N
    # voxels = kal.ops.voxelgrid.fill(voxels).squeeze()
    timelapse.add_mesh_batch(
        iteration=0,
        category='merged_plane_' + cls + '_L' + str(level),
        vertices_list=[vertices],
        faces_list=[faces]
    )    
    # timelapse.add_voxelgrid_batch(
    #     iteration=0,
    #     category='hgmm_NOnms_' + cls + '_L' + str(level),
    #     voxelgrid_list=[voxels] # K x N x N x N
    # ) 

def hgmm_2_box_2_usd(mix, mu, cov, level, azim=60, elev=0):
    logs_path = './logs/'
    timelapse = kal.visualize.Timelapse(logs_path)

    vertices = []
    faces = []
    voxels = []
    for k in range(mix.shape[0]):
        # skip if no points in instance k
        if not mix[k]: 
            continue
        
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        X, Y, Z = cuboid_data([0,0,0], [1,1,1])#mu[k]
        XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        numWires_x = 4
        numWires = 5
        xyz_stretch = 1.2*np.sqrt(5.99*s)[:, None] * XYZ
        # xyz_stretch = np.sqrt(5*s)[:, None] * XYZ
        xyz = V.T @ (xyz_stretch) + mu[k][:, None]#V.T
        x, y, z = xyz
        
        corners = get_box_corners(x, y, z)
        box_faces = cuboid_faces(k)
        vertices.append(corners)
        faces.append(box_faces)

    vertices = torch.cat(vertices, dim=0) # k x 8 x 3 -> 8k x 3
    faces = torch.cat(faces, dim=0) # k x 12 x 3 -> 12k x 3
    # voxels = kal.ops.conversions.trianglemeshes_to_voxelgrids(vertices.unsqueeze(0), faces, 80).squeeze() # N x N x N
    # voxels = kal.ops.voxelgrid.fill(voxels).squeeze()
    timelapse.add_mesh_batch(
        iteration=0,
        category='box_' + cls + '_L' + str(level),
        vertices_list=[vertices],
        faces_list=[faces]
    )    
    # timelapse.add_voxelgrid_batch(
    #     iteration=0,
    #     category='hgmm_NOnms_' + cls + '_L' + str(level),
    #     voxelgrid_list=[voxels] # K x N x N x N
    # ) 

def hgmm_2_usd(mix, mu, cov, level, azim=60, elev=0):
    logs_path = './logs/'
    timelapse = kal.visualize.Timelapse(logs_path)
    sphere_mesh = kal.io.obj.import_mesh('./generated/sphere.obj', with_materials=True)

    X = sphere_mesh.vertices[:, 0]
    Y = sphere_mesh.vertices[:, 1]
    Z = sphere_mesh.vertices[:, 2]

    XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
        
    vertices = []
    faces = []
    voxels = []
    for k in range(mix.shape[0]):
        # skip if no points in instance k
        if not mix[k]: 
            continue
        
        # find the rotation matrix and radii of the axes
        U, s, V = np.linalg.svd(cov[k])
        
        xyz_stretch = 1.2*np.sqrt(5.99*s)[:, None] * XYZ
        # xyz_stretch = np.sqrt(5*s)[:, None] * XYZ
        xyz = V.T @ (xyz_stretch) + mu[k][:, None]#V.T
        x, y, z = xyz
        
        vertices.append(torch.from_numpy(xyz.T))
        faces.append(sphere_mesh.faces + sphere_mesh.vertices.shape[0] * k)

    vertices = torch.cat(vertices, dim=0) # k x n_verts x 3 -> 8k x 3
    faces = torch.cat(faces, dim=0) # k x 12 x 3 -> 12k x 3
    timelapse.add_mesh_batch(
        iteration=0,
        category='hgmm_NOnms_' + cls + '_L' + str(level),
        vertices_list=[vertices],
        faces_list=[faces]
    )    

def hgmms(encoder, decoder, args: Options, trace: ViewMem):
    input_points, z ,input_mesh= get_z_by_id(encoder, args, 1, idx=[6], trace=trace)
    input_mesh = [{
        'vertices': input_mesh[0].squeeze(0),
        'faces': input_mesh[1].squeeze(0)
    }]
    gms = decoder(z)
    num_gms = len(gms)
    vs = []
    splits = []
    pts_seg = []
    pi = []
    mu = []
    sigma = []
    pi_nms_list = []
    mu_nms_list = []
    sigma_nms_list = []
    cuboid_list = []
    cuboid_list_merged = []
    for i in range(num_gms):
        gms_ = [gms[i] for i in range(i+1)]
        vs_, splits_, pi_, mu_, sigma_ = gm_utils.hierarchical_gm_sample(gms_, trace.points_in_sample,
                                                                        flatten_sigma=False)
        # pi_, mu_, sigma_ = gm_utils.get_hgmm_params(gms_, flatten_sigma=False)
        pi_ = pi_.squeeze(0).cpu().numpy()
        mu_ = mu_.squeeze(0).cpu().numpy()
        sigma_ = sigma_.squeeze(0).cpu().numpy()

        U, s, V = np.linalg.svd(sigma_)
        SVD = {
        'U': U, # kx3x3
        's': s, # kx3
        'V': V  # kx3x3
        }

        # pi_nms, mu_nms, sigma_nms = pi_, mu_, sigma_
        # hgmm_2_usd(pi_, mu_, sigma_, i)
        # TODO: substitute IoU calculation with primitive merging algo.
        pi_nms, mu_nms, sigma_nms, SVD = nms.non_max_suppression_3d((pi_, mu_, sigma_, SVD), 
                                                                    nms_th=0.35, conf_th=0.001)
        # hgmm_2_box_2_usd(pi_, mu_, sigma_, i)

        gmm = GaussianMixture(n_components=pi_nms.shape[0])
        gmm.weights_ = pi_nms              # K
        gmm.means_ = mu_nms                # Kx3
        gmm.covariances_ = sigma_nms       # Kx3x3
        gmm.precisions_cholesky_ = _compute_precision_cholesky(sigma_nms, gmm.covariance_type)
        pt_membership_ = gmm.predict(input_points.squeeze(0))
        pts_seg.append(pt_membership_)

        # Mark gaussians with no points
        for k in range(pi_nms.shape[0]):
            if k in pt_membership_:
                # print(k)
                pass
            else: 
                # print("None")
                gmm.weights_[k] = 0
        n_instances = pi_nms.shape[0]
        W = np.zeros((1, pt_membership_.shape[0], n_instances), dtype=bool)
        for idx, a_i in enumerate(pt_membership_):
            W[0, idx, a_i] = 1
        W_sum = np.sum(W, axis=1)

        fitter_feed = {
        'P': input_points,   # 1xNx3
        'W': W,            # 1xN
        'GMM': gmm,            # 1xN
        'SVD': SVD
        # 'normal_per_point': normal_per_point,
        }
        parameters = {}

        # matplotlib.use( 'tkagg' )
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(azim=60, elev=0)

        level_cuboids = CuboidFitter.compute_parameters(fitter_feed, parameters, ax=None, n_instances=n_instances, plot=False)
        
        # set_axes_equal(ax)
        # plt.show()
        # PlaneFitter.compute_parameters(fitter_feed, parameters, ax, n_instances, plot=False)
        # SphereFitter.compute_parameters(fitter_feed, parameters, ax, plot=False)
        # ax.scatter(input_points[0, :, 0], input_points[0, :, 1], input_points[0, :, 2])
        # PlaneFitter.merge_planes(parameters, ax, plot=True)

        # fig = plt.figure(figsize=(20, 20))
        # ax2 = fig.add_subplot(111, projection='3d')
        # ax2.view_init(azim=60, elev=0)
        level_cuboids_merged = CuboidFitter.merge_cuboids(level_cuboids, ax=None, plot=False)
        level_cuboids_merged = CuboidFitter.merge_cuboids(level_cuboids_merged, ax=None, plot=False)
        # set_axes_equal(ax2)
        # plt.show()
        # plane_2_usd(parameters['planes'], i)
        
        """
        TODO: build a neighboring graph to connect and merge neighboring primitives.
        Another nms algorithm: for those with center in the same plane of similar 2-axis direction,
        if edges are close, merge.
        """


        vs.append(vs_.squeeze(0).cpu().numpy())
        splits.append(splits_.squeeze(0).cpu().numpy())
        pi.append(pi_)
        mu.append(mu_)
        sigma.append(sigma_)
        pi_nms_list.append(pi_nms)
        mu_nms_list.append(mu_nms)
        sigma_nms_list.append(sigma_nms)
        cuboid_list.append(level_cuboids)
        cuboid_list_merged.append(level_cuboids_merged)
        

    # pts_seg = gm_utils.hgmm_segmentation(gms, input_points)
    palette = create_palettes(splits)
    im, sample_points = view(vs, splits, palette)
    return True, (hgmms, ), im, (input_points.squeeze(0), pi, mu, sigma, pts_seg, pi_nms_list, 
                                mu_nms_list, sigma_nms_list, cuboid_list, cuboid_list_merged, input_mesh)
    # return True, (hgmms, ), im, (input_points.squeeze(0), sample_points, splits, pts_seg, palette, pi, mu, sigma)


def interpolate(encoder, decoder, args: Options, trace: ViewMem):
    if trace.memory is None:
        msg = '\tPlease choose number of interpolation: from 8 to 20\n\t'
        num_interpulate = get_integer((7, 21), msg)
    else:
        num_interpulate = trace.memory[1]
    input_points, z = get_z_by_id(encoder, args, 2, trace.last_idx, trace)
    gms = decoder.interpulate(z, num_interpulate)
    vs, splits = gm_utils.hierarchical_gm_sample(gms, trace.points_in_sample)
    spread = [vs[i].cpu().numpy() for i in range(vs.shape[0])]
    splits = [s for s in splits]
    palette = create_palettes(splits)
    im, points = view(spread, splits, palette) #, save_path=f'{cp_folder}/interpulate_{idx[0].item()}_{idx[1].item()}.png')
    return True, (interpolate, (num_interpulate,)), im, (points, splits, palette)


def evaluate(args: Options, trace: ViewMem):

    def last_try(encdoer, decoder, args, trace):
        if trace.memory is None:
            print("Don't know what to do")
            return False, None
        return trace.memory[0](encdoer, decoder, args, trace)

    def to_exit(_, __, ___, ____):
        print(':-o Goodbye')
        return False, None, None, None

    encoder, _ = factory.model_lc(args.encoder, args, device=trace.device)
    decoder, _ = factory.model_lc(args.decoder, args, device=trace.device)
    encoder.eval(), decoder.eval()

    choices = {0: to_exit, 1: sample, 2: interpolate, 3: hgmms, 4: last_try}
    menu = ' | '.join(sorted(list(f"{key}: {str(item).split()[1].split('.')[-1]}" for key, item in choices.items())))
    eval_choice = 1
    allow_saving = saving_handler(args, trace)
    while eval_choice:
        eval_choice = get_integer((0, len(choices)), menu + '\n')
        with torch.no_grad():
            if eval_choice != len(choices) - 1:
                trace.memory = None
            check, trace.memory, image, points_group = choices[eval_choice](encoder, decoder, args, trace)
            if check:
                function_name = str(trace.memory[0]).split()[1].split('.')[-1]
                allow_saving(function_name, image, points_group)
                print(f"{function_name} done")


if __name__ == '__main__':
    cls = 'table'
    # "chair" = 1
    # "airplane" = 4
    # "table" = 6
    # logs_path = './logs/'
    # timelapse = kal.visualize.Timelapse(logs_path)
    evaluate(TrainOptions(tag=cls).load(), ViewMem())