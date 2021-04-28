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

    def compute_parameters(feed_dict, parameters, ax, plot=False):
        matplotlib.use( 'tkagg' )
        P = feed_dict['P'].squeeze(0).cpu().numpy()  # BxNx3
        W = feed_dict['W']  # BxNxK
        gmm = feed_dict['GMM']
        s = feed_dict['SVD']['s']
        V = feed_dict['SVD']['V']
        axis_range = 1.2*np.sqrt(5.99*s)
        n = V[:, :, -1]
        P_mean = np.mean(P, axis=0, keepdims=True)     # Bx3
        c_ = n * P_mean
        c = np.sum(c_, axis=1)          
        parameters['plane_n'] = n       # BxKx3
        parameters['plane_c'] = c       # BxK

        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.view_init(azim=60, elev=0)
        alpha = gmm.weights_ / gmm.weights_.max()
        planes = []
        for k in range(s.shape[0]):
            # X, Y, Z = surface_data([0,0,0], [1,1,0])
            # XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
            # x, y, z = V[k].T @ (1.2*np.sqrt(5.99*s[k])[:, None] * XYZ) + gmm.means_[k][:, None]
            # x = x.reshape(2,2)
            # y = y.reshape(2,2)
            # z = z.reshape(2,2)
            # ax1.plot_surface(x,y,z, rstride=1, cstride=1, alpha=alpha[k])

            x_len = axis_range[k,0]/2
            y_len = axis_range[k,1]/2
            plane_k = Plane(n[k], c[k], center=gmm.means_[k], 
                            x_axis=V[k,:,0], y_axis=V[k,:,1], 
                            x_range=[-x_len, x_len], 
                            y_range=[-y_len, y_len])
            planes.append(plane_k)
            if plot:
                plane_k.plot(alpha[k], ax)
                
        # if plot:
        #     set_axes_equal(ax)
        #     plt.show()
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

