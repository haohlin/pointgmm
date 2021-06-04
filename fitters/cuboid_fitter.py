import os, sys
# BASE_DIR = os.path.dirname(__file__)
# sys.path.append(os.path.join(BASE_DIR, '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

# from utils.tf_wrapper import batched_gather
# from utils.geometry_utils import weighted_plane_fitting
# from utils.tf_numerical_safe import acos_safe
# from fitters.adaptors import *
from models.primitives import *
from matplotlib import pyplot as plt
from show.view_utils import set_axes_equal
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from models.models_utils import *
from show.view_utils import set_axes_equal
import matplotlib
import torch
import tensorflow as tf
import numpy as np

# ''' Fitters should have no knowledge of ground truth labels,
#     and should assume all primitives are of the same type.
#     They should predict a primitive for every column.
# '''

def surface_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]   # (-1/2, -1/2, 0)
    # get the length, width, and height
    l, w, h = size                                  # (1, 1, 0)
    x = [[o[0], o[0] + l], 
         [o[0], o[0] + l]]                # x coordinate of points in bottom surface

    y = [[o[1], o[1]], 
         [o[1] + w, o[1] + w]]                # y coordinate of points in bottom surface

    z = [[o[2], o[2]], 
         [o[2], o[2]]]                        # z coordinate of points in bottom surface
    return np.array(x), np.array(y), np.array(z)

class CuboidFitter:
    def primitive_name():
        return 'cuboid'
        
    def weighted_plane_fitting(P, W):
        # P - BxNx3
        # W - BxN
        # Returns n, c, with n - Bx3, c - B
        WP = P * tf.expand_dims(W, axis=2) # BxNx3
        W_sum = tf.reduce_sum(W, axis=1) # B
        P_weighted_mean = tf.reduce_sum(WP, axis=1) / tf.maximum(tf.expand_dims(W_sum, 1), DIVISION_EPS) # Bx3
        A = P - tf.expand_dims(P_weighted_mean, axis=1) # BxNx3
        # n = solve_weighted_tls(A, W) # Bx3
        c = tf.reduce_sum(n * P_weighted_mean, axis=1)
        return n, c

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['plane_n'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['plane_c'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['plane_n'] = tf.nn.l2_normalize(parameters['plane_n'], axis=2)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        parameters_gt['plane_n'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        feed_dict[parameters_gt['plane_n']] = batch['plane_n_gt']

    # def ax_align(ax1, ax2, range1, range2, center_vector, center_dist):
    #     equal_axis = cross_norm(ax1, ax2) < 0.258 \
    #                     and cross_norm(ax1, center_vector) < 0.258 \
    #                     and cross_norm(ax2, center_vector) < 0.258 \
    #                     and (center_dist - (range1 + range2)) / center_dist < 0.4
    #     merged_range = np.zeros(2)
    #     if equal_axis:
    #         merged_range[0] = (center_dist + range1 + range2) / 2
    #         merged_range[1] = 
    #     return equal_axis, merged_range

    def same_cuboid(p1:Cuboid, p2:Cuboid):
        dist_thresh = 0.3#0.2
        ang_thresh = 0.15#0.258
        
        center_vector = p1.center-p2.center
        center_dist = np.linalg.norm(center_vector)
        center_vector = l2_norm(center_vector)
        
        # if np.dot(p1.n, p2.n) < 0:
        #     mean_norm = l2_norm(p1.n - p2.n)
        # else: 
        #     mean_norm = l2_norm(p1.n + p2.n)

        merged_range = np.zeros(2)

        # TODO: aline three axis and compare edge length
        # 1. Get center vector;
        # 2. Find the axis (in each cuboid) that aligns with the center vector;
        # 3. Align the rest two axis based on axis size
        # 4. If center distance, size and orientation of the two edges are within threshold:
        # 5. Merge.

        equal_x = cross_norm(p1.x_axis, p2.x_axis) < ang_thresh \
                    and cross_norm(p1.x_axis, center_vector) < ang_thresh \
                    and cross_norm(p2.x_axis, center_vector) < ang_thresh \
                    and (center_dist - (p1.x_range[0] + p2.x_range[0])) / center_dist < dist_thresh # Boundaries of two cuboids are close
        equal_y = cross_norm(p1.y_axis, p2.y_axis) < ang_thresh \
                    and cross_norm(p1.y_axis, center_vector) < ang_thresh \
                    and cross_norm(p2.y_axis, center_vector) < ang_thresh \
                    and (center_dist - (p1.y_range[0] + p2.y_range[0])) / center_dist < dist_thresh
        equal_xy = cross_norm(p1.x_axis, p2.y_axis) < ang_thresh \
                    and cross_norm(p1.x_axis, center_vector) < ang_thresh \
                    and cross_norm(p2.y_axis, center_vector) < ang_thresh \
                    and (center_dist - (p1.x_range[0] + p2.y_range[0])) / center_dist < dist_thresh
        equal_yx = cross_norm(p1.x_axis, p2.x_axis) < ang_thresh \
                    and cross_norm(p1.x_axis, center_vector) < ang_thresh \
                    and cross_norm(p2.x_axis, center_vector) < ang_thresh \
                    and (center_dist - (p1.y_range[0] + p2.x_range[0])) / center_dist < dist_thresh
        if equal_x:
            merged_range[0] = (center_dist + p1.x_range[0] + p2.x_range[0]) / 2
            merged_range[1] = (p1.y_range[0] + p2.y_range[0]) / 2
            # TODO: add new center generation based on edges
        elif equal_y:
            merged_range[0] = (center_dist + p1.y_range[0] + p2.y_range[0]) / 2
            merged_range[1] = (p1.x_range[0] + p2.x_range[0]) / 2
        elif equal_xy:
            merged_range[0] = (center_dist + p1.x_range[0] + p2.y_range[0]) / 2
            merged_range[1] = (p1.y_range[0] + p2.x_range[0]) / 2
        elif equal_yx:
            merged_range[0] = (center_dist + p1.y_range[0] + p2.x_range[0]) / 2
            merged_range[1] = (p1.x_range[0] + p2.y_range[0]) / 2
            
        equal_norm = cross_norm(p1.n, p2.n) < ang_thresh # angle of two normals < 15 degrees
        axis_same_plame = (equal_x or equal_y or equal_xy or equal_yx) and equal_norm
        centers_same_cuboid = abs(np.dot(center_vector, mean_norm)) < ang_thresh # centers on the same cuboid < 15 degrees
        
        # merge p1 and p2
        if axis_same_plame and centers_same_cuboid:
            center = (p1.center + p2.center) / 2
            n = mean_norm
            x_axis = center_vector
            y_axis = l2_norm(np.cross(n, x_axis))
            x_range = [merged_range[0], merged_range[0]]
            y_range = [merged_range[1], merged_range[1]]
            obj_conf = (p1.obj_conf + p2.obj_conf) / 2

            cuboid = Cuboid(n=n, center=center, #, c[k], 
                            x_axis=x_axis, y_axis=y_axis, 
                            x_range=x_range, y_range=y_range,
                            obj_conf=obj_conf)
            
            # TODO: add point cloud plot and compare 
            # matplotlib.use( 'tkagg' )
            # fig = plt.figure(figsize=(20, 20))
            # ax = fig.add_subplot(121, projection='3d')
            # ax.view_init(azim=60, elev=0)
            # p1.plot(ax)
            # p2.plot(ax)

            # ax = fig.add_subplot(122, projection='3d')
            # ax.view_init(azim=60, elev=0)
            # cuboid.plot(ax)

            # set_axes_equal(ax)
            # plt.show()

            # cuboid = Cuboid()
            # cuboid.center = (p1.center + p2.center) / 2
            # cuboid.n = l2_norm(p1.n + p2.n)
            # cuboid.x_axis = center_vector
            # cuboid.y_axis = l2_norm(np.cross(cuboid.n, cuboid.x_axis))
            # cuboid.x_range = [merged_range[0], merged_range[0]]
            # cuboid.y_range = [merged_range[1], merged_range[1]]
            return 1, cuboid
        else:
            return 0, None

    def merge(p1:Cuboid, p2:Cuboid):
        # TODO: merge p1 and p2
        cuboid = Cuboid()
        cuboid.center = (p1.center + p2.center) / 2
        cuboid.n = l2_norm(p1.n + p2.n)
        # cuboid.x_axis = 
        # cuboid.y_axis = 
        pass

    def merge_cuboids(cuboid_list, ax, plot=False):
        # cuboid_list = parameters['cuboids']
        cuboids_cur = [cuboid_list[0]] 

        for k in np.arange(1, len(cuboid_list)):
            ground_truth = cuboid_list[k]
            flag = 1
            # print("Testing instance" + str(i))
            for j in range(len(cuboids_cur)):
                # TODO: find best merging option, then merge
                predicted = cuboids_cur[j]
                is_same_cuboid, merged_cuboid = CuboidFitter.same_cuboid(predicted, ground_truth) 
                
                if is_same_cuboid: # if can be merged
                    flag = -1 # merge and add to current cuboid list
                    # merged = merge(predicted, ground_truth)
                    cuboids_cur[j] = merged_cuboid
                    break
            if flag == 1:
                cuboids_cur.append(ground_truth) # if not mergeable with any current cuboids, add to current cuboid list

        for k in range(len(cuboids_cur)):
            if plot:
                cuboids_cur[k].plot(ax)
        print("Cuboids after merging", len(cuboids_cur))
        # parameters['cuboids'] = cuboids_cur
        return cuboids_cur


    def compute_parameters(feed_dict, parameters, ax, n_instances, plot=False):
        matplotlib.use( 'tkagg' )
        P = feed_dict['P'].squeeze(0).cpu().numpy()  # BxNx3
        W = feed_dict['W']  # BxNxK
        gmm = feed_dict['GMM']
        s = feed_dict['SVD']['s']
        V = feed_dict['SVD']['V']
        # normalize V
        V = l2_norm(V, axis=2)
        axis_range = np.sqrt(5.99*s)

        alpha = gmm.weights_ / gmm.weights_.max()
        boxs = []
        for k in range(n_instances):
            if not gmm.weights_[k]: # if no points in instance k
                continue

            # p_seg = P[W[0,:,k]] - center
            # dist_x = np.dot(p_seg, x_axis) / np.linalg.norm(x_axis)
            # dist_y = np.dot(p_seg, y_axis) / np.linalg.norm(y_axis)
            # x_len_1 = np.max(dist_x)
            # x_len_2 = abs(np.min(dist_x))
            # y_len_1 = np.max(dist_y)
            # y_len_2 = abs(np.min(dist_y))

            box_model = Cuboid(center=gmm.means_[k], 
                            x_axis=V[k,0,:], y_axis=V[k,1,:], z_axis=V[k,2,:], 
                            x_range=[axis_range[k,0]/2, axis_range[k,0]/2], 
                            y_range=[axis_range[k,1]/2, axis_range[k,1]/2], 
                            z_range=[axis_range[k,2]/2, axis_range[k,2]/2], 
                            obj_conf=alpha[k])

            box_k = {
                'type': 'cuboid',
                'center': box_model.center,
                'x_range': box_model.x_range,
                'y_range': box_model.y_range,
                'z_range': box_model.z_range,
                'x_axis': box_model.x_axis,
                'y_axis': box_model.y_axis,
                'z_axis': box_model.z_axis,
                'obj_conf': box_model.obj_conf
            }

            boxs.append(box_k)
            if plot:
                # matplotlib.use( 'tkagg' )
                # fig = plt.figure(figsize=(20, 20))
                # ax = fig.add_subplot(111, projection='3d')
                # ax.view_init(azim=60, elev=0)
                box_model.plot(ax)
                # set_axes_equal(ax)
                # plt.show()
                
        parameters['boxs'] = boxs   
        return boxs

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return PlaneFitter.compute_residue_single(
            *adaptor_matching([parameters['plane_n'], parameters['plane_c']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return PlaneFitter.compute_residue_single(
            *adaptor_pairwise([parameters['plane_n'], parameters['plane_c']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(n, c, p):
        # n: ...x3, c: ..., p: ...x3
        return tf.square(tf.reduce_sum(p * n, axis=-1) - c)

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        # n - BxKx3
        n = batched_gather(parameters_pred['plane_n'], matching_indices, axis=1)
        dot_abs = tf.abs(tf.reduce_sum(n * parameters_gt['plane_n'], axis=2))
        if angle_diff:
            return acos_safe(dot_abs) # BxK
        else:
            return 1.0 - dot_abs # BxK

    def extract_predicted_parameters_as_json(fetched, k):
        # This is only for a single plane
        plane = Plane(fetched['plane_n'][k], fetched['plane_c'][k])
        
        json_info = {
            'type': 'plane',
            'center_x': plane.center[0],
            'center_y': plane.center[1],
            'center_z': plane.center[2], 
            'normal_x': plane.n[0],
            'normal_y': plane.n[1],
            'normal_z': plane.n[2],
            'x_size': plane.x_range[1] - plane.x_range[0],
            'y_size': plane.y_range[1] - plane.y_range[0],
            'x_axis_x': plane.x_axis[0],
            'x_axis_y': plane.x_axis[1],
            'x_axis_z': plane.x_axis[2],
            'y_axis_x': plane.y_axis[0],
            'y_axis_y': plane.y_axis[1],
            'y_axis_z': plane.y_axis[2],
        }

        return json_info

    def extract_parameter_data_as_dict(primitives, n_max_instances):
        n = np.zeros(dtype=float, shape=[n_max_instances, 3])
        for i, primitive in enumerate(primitives):
            if isinstance(primitive, Plane):
                n[i] = primitive.n
        return {
            'plane_n_gt': n
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'plane'
        location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
        axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
        return Plane(n=axis, c=np.dot(location, axis))

