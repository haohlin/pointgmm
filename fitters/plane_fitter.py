import os, sys
# BASE_DIR = os.path.dirname(__file__)
# sys.path.append(os.path.join(BASE_DIR, '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

# from utils.tf_wrapper import batched_gather
# from utils.geometry_utils import weighted_plane_fitting
# from utils.tf_numerical_safe import acos_safe
# from fitters.adaptors import *
from models.primitives import Plane
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

class PlaneFitter:
    def primitive_name():
        return 'plane'
        
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

    def same_plane(p1:Plane, p2:Plane):
        dist_thresh = 0.3#0.2
        ang_thresh = 0.15#0.258
        
        center_vector = p1.center-p2.center
        center_dist = np.linalg.norm(center_vector)
        center_vector = l2_norm(center_vector)
        
        if np.dot(p1.n, p2.n) < 0:
            mean_norm = l2_norm(p1.n - p2.n)
        else: 
            mean_norm = l2_norm(p1.n + p2.n)

        merged_range = np.zeros(2)
        equal_x = cross_norm(p1.x_axis, p2.x_axis) < ang_thresh \
                    and cross_norm(p1.x_axis, center_vector) < ang_thresh \
                    and cross_norm(p2.x_axis, center_vector) < ang_thresh \
                    and (center_dist - (p1.x_range[0] + p2.x_range[0])) / center_dist < dist_thresh # Boundaries of two planes are close
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
        centers_same_plane = abs(np.dot(center_vector, mean_norm)) < ang_thresh # centers on the same plane < 15 degrees
        
        # merge p1 and p2
        if axis_same_plame and centers_same_plane:
            center = (p1.center + p2.center) / 2
            n = mean_norm
            x_axis = center_vector
            y_axis = l2_norm(np.cross(n, x_axis))
            x_range = [merged_range[0], merged_range[0]]
            y_range = [merged_range[1], merged_range[1]]
            obj_conf = (p1.obj_conf + p2.obj_conf) / 2

            plane = Plane(n=n, center=center, #, c[k], 
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
            # plane.plot(ax)

            # set_axes_equal(ax)
            # plt.show()

            # plane = Plane()
            # plane.center = (p1.center + p2.center) / 2
            # plane.n = l2_norm(p1.n + p2.n)
            # plane.x_axis = center_vector
            # plane.y_axis = l2_norm(np.cross(plane.n, plane.x_axis))
            # plane.x_range = [merged_range[0], merged_range[0]]
            # plane.y_range = [merged_range[1], merged_range[1]]
            return 1, plane
        else:
            return 0, None

    def merge(p1:Plane, p2:Plane):
        # TODO: merge p1 and p2
        plane = Plane()
        plane.center = (p1.center + p2.center) / 2
        plane.n = l2_norm(p1.n + p2.n)
        # plane.x_axis = 
        # plane.y_axis = 
        pass

    def merge_planes(plane_list, ax, plot=False):
        # plane_list = parameters['planes']
        planes_cur = [plane_list[0]] 

        for k in np.arange(1, len(plane_list)):
            ground_truth = plane_list[k]
            flag = 1
            # print("Testing instance" + str(i))
            for j in range(len(planes_cur)):
                # TODO: find best merging option, then merge
                predicted = planes_cur[j]
                is_same_plane, merged_plane = PlaneFitter.same_plane(predicted, ground_truth) 
                
                if is_same_plane: # if can be merged
                    flag = -1 # merge and add to current plane list
                    # merged = merge(predicted, ground_truth)
                    planes_cur[j] = merged_plane
                    break
            if flag == 1:
                planes_cur.append(ground_truth) # if not mergeable with any current planes, add to current plane list

        for k in range(len(planes_cur)):
            if plot:
                planes_cur[k].plot(ax)
        print("Planes after merging", len(planes_cur))
        # parameters['planes'] = planes_cur
        return planes_cur


    def compute_parameters(feed_dict, parameters, ax, n_instances, plot=False):
        matplotlib.use( 'tkagg' )
        P = feed_dict['P'].squeeze(0).cpu().numpy()  # BxNx3
        W = feed_dict['W']  # BxNxK
        gmm = feed_dict['GMM']
        s = feed_dict['SVD']['s']
        V = feed_dict['SVD']['V']
        # normalize V
        V = l2_norm(V, axis=2)
        axis_range = 1.2*np.sqrt(5.99*s)

        # calculate the correct normal of x/y plane
        n = np.cross(V[:, 0, :], V[:, 1, :]) # K
        n = l2_norm(n, axis=1)
        # n = l2_norm(V[:, :, 2], axis=1)
        # n = V[:, :, -1]
        # P_mean = np.mean(P, axis=0, keepdims=True)     # Bx3
        # c_ = n * P_mean
        # c = np.sum(c_, axis=1)          
        # parameters['plane_n'] = n       # BxKx3
        # parameters['plane_c'] = c       # BxK

        alpha = gmm.weights_ / gmm.weights_.max()
        planes = []
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

            plane_k = Plane(n=n[k], center=gmm.means_[k], #, c[k], 
                            x_axis=V[k,0,:], y_axis=V[k,1,:], 
                            x_range=[axis_range[k,0]/2, axis_range[k,0]/2], 
                            y_range=[axis_range[k,1]/2, axis_range[k,1]/2], 
                            obj_conf=alpha[k])
            planes.append(plane_k)
            if plot:
                # matplotlib.use( 'tkagg' )
                # fig = plt.figure(figsize=(20, 20))
                # ax = fig.add_subplot(111, projection='3d')
                # ax.view_init(azim=60, elev=0)
                plane_k.plot(ax)
                # set_axes_equal(ax)
                # plt.show()
                
        parameters['planes'] = planes     

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

