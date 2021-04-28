import os, sys
# BASE_DIR = os.path.dirname(__file__)
# sys.path.append(os.path.join(BASE_DIR, '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..'))
# sys.path.append(os.path.join(BASE_DIR, '..', '..', '..'))

# from utils.tf_wrapper import batched_gather
# from utils.geometry_utils import weighted_sphere_fitting
# from utils.tf_numerical_safe import sqrt_safe
# from fitters.adaptors import *
# from primitives import Sphere

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from show.view_utils import set_axes_equal
import tensorflow as tf
import numpy as np
import matplotlib

DIVISION_EPS = 1e-10
SQRT_EPS = 1e-10
LS_L2_REGULARIZER = 1e-8

'''
    L = \sum_{i=1}^m w_i [(p_i - x)^2 - r^2]^2
    dL/dr = 0 => r^2 = \frac{1}{\sum_i w_i} \sum_j w_j (p_j - x)^2
    => L = \sum_{i=1}^m w_i[p_i^2 - \frac{\sum_j w_j p_j^2}{\sum_j w_j} + 2x \cdot (-p_i + \frac{\sum_j w_j p_j}{\sum_j w_j})]^2
    So
    A_i = 2\sqrt{w_i} (\frac{w_j p_j}{\sum_j w_j} - p_i)
    b_i = \sqrt{w_i}[\frac{\sum_j w_j p_j^2}{\sum_j w_j} - p_i^2]
    So \argmin_x ||Ax-b||^2 gives the best center of the sphere
'''
def weighted_sphere_fitting(P, W):
    # P - BxNxD
    # W - BxN
    W_sum = tf.reduce_sum(W, axis=1) # B
    WP_sqr_sum = tf.reduce_sum(W * tf.reduce_sum(tf.square(P), axis=2), axis=1) # B
    P_sqr = tf.reduce_sum(tf.square(P), axis=2) # BxN
    b = tf.expand_dims(tf.expand_dims(WP_sqr_sum / tf.maximum(W_sum, DIVISION_EPS), axis=1) - P_sqr, axis=2) # BxNx1
    WP_sum = tf.reduce_sum(tf.expand_dims(W, axis=2) * P, axis=1) # BxD
    A = 2 * (tf.expand_dims(WP_sum / tf.expand_dims(tf.maximum(W_sum, DIVISION_EPS), axis=1), axis=1) - P) # BxNxD

    # Seek least norm solution to the least square
    center = guarded_matrix_solve_ls(A, b, W) # BxD
    W_P_minus_C_sqr_sum = P - tf.expand_dims(center, axis=1) # BxNxD
    W_P_minus_C_sqr_sum = W * tf.reduce_sum(tf.square(W_P_minus_C_sqr_sum), axis=2) # BxN
    r_sqr = tf.reduce_sum(W_P_minus_C_sqr_sum, axis=1) / tf.maximum(W_sum, DIVISION_EPS) # B

    return {'center': center, 'radius_squared': r_sqr}

def guarded_matrix_solve_ls(A, b, W, condition_number_cap=1e5):
    # Solve weighted least square ||\sqrt(W)(Ax-b)||^2
    # A - BxNxD
    # b - BxNx1
    # W - BxN
    sqrt_W = tf.sqrt(tf.maximum(W, SQRT_EPS)) # BxN
    A *= tf.expand_dims(sqrt_W, axis=2) # BxNxD
    b *= tf.expand_dims(sqrt_W, axis=2) # BxNx1
    # Compute singular value, trivializing the problem when condition number is too large
    AtA = tf.matmul(a=A, b=A, transpose_a=True)
    s, _, _ = [tf.stop_gradient(u) for u in tf.linalg.svd(AtA)] # s will be BxD
    mask = tf.less(s[:, 0] / s[:, -1], condition_number_cap) # B
    A *= tf.cast(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=2), dtype=tf.float32) # zero out badly conditioned data
    x = tf.linalg.lstsq(A, b, l2_regularizer=LS_L2_REGULARIZER, fast=True) # BxDx1 
    return tf.squeeze(x, axis=2) # BxD

class SphereFitter:
    def primitive_name():
        return 'sphere'

    def insert_prediction_placeholders(pred_ph, n_max_instances):
        pred_ph['sphere_center'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances, 3])
        pred_ph['sphere_radius_squared'] = tf.placeholder(dtype=tf.float32, shape=[None, n_max_instances])

    def normalize_parameters(parameters):
        parameters['sphere_radius_squared'] = tf.clip_by_value(parameters['sphere_radius_squared'], 1e-4, 1e6)

    def insert_gt_placeholders(parameters_gt, n_max_instances):
        pass

    def fill_gt_placeholders(feed_dict, parameters_gt, batch):
        pass

    def compute_parameters(feed_dict, parameters, ax, plot=False):
        gmm = feed_dict['GMM']
        k = gmm.weights_.shape[0]
        P = feed_dict['P']
        W = feed_dict['W']
        batch_size = tf.shape(P)[0]
        n_points = tf.shape(P)[1]
        # W = np.ones((batch_size, n_points, k), dtype=np.float32) / k
        n_max_primitives = tf.shape(W)[2]
        P = tf.tile(tf.expand_dims(P, axis=1), [1, n_max_primitives, 1, 1]) # BxKxNx3
        W = tf.transpose(W, perm=[0, 2, 1]) # BxKxN
        P = tf.reshape(P, [batch_size * n_max_primitives, n_points, 3]) # BKxNx3
        W = tf.reshape(W, [batch_size * n_max_primitives, n_points]) # BKxN
        fitting_result = weighted_sphere_fitting(P, W)

        parameters['sphere_center'] = tf.reshape(fitting_result['center'], [batch_size, n_max_primitives, 3])
        parameters['sphere_radius_squared'] = tf.reshape(fitting_result['radius_squared'], [batch_size, n_max_primitives])

        if plot:
            matplotlib.use( 'tkagg' )
            # fig = plt.figure(figsize=(20, 20))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.view_init(azim=60, elev=0)
            alpha = gmm.weights_ / gmm.weights_.max()

            numWires = 6
            u = np.linspace(0.0, 2.0 * np.pi, numWires)
            v = np.linspace(0.0, np.pi, numWires)
            X = np.outer(np.cos(u), np.sin(v))
            Y = np.outer(np.sin(u), np.sin(v))
            Z = np.outer(np.ones_like(u), np.cos(v)) 
            XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])

            for idx, (c, r) in enumerate(zip(parameters['sphere_center'][0], parameters['sphere_radius_squared'][0])):
                # x, y, z = np.sqrt(r) * XYZ + c[:, None]
                x, y, z = 0.1 * XYZ + gmm.means_[idx][:, None]
                x = x.reshape(numWires, numWires)#.numpy()
                y = y.reshape(numWires, numWires)
                z = z.reshape(numWires, numWires)
                ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=alpha[idx])
            set_axes_equal(ax)
            plt.show()

    def compute_residue_loss(parameters, P_gt, matching_indices):
        return SphereFitter.compute_residue_single(
            *adaptor_matching([parameters['sphere_center'], parameters['sphere_radius_squared']], matching_indices), 
            P_gt
        )

    def compute_residue_loss_pairwise(parameters, P_gt):
        return SphereFitter.compute_residue_single(
            *adaptor_pairwise([parameters['sphere_center'], parameters['sphere_radius_squared']]), 
            adaptor_P_gt_pairwise(P_gt)
        )

    def compute_residue_single(center, radius_squared, p):
        return tf.square(sqrt_safe(tf.reduce_sum(tf.square(p - center), axis=-1)) - sqrt_safe(radius_squared))

    def compute_parameter_loss(parameters_pred, parameters_gt, matching_indices, angle_diff):
        return None

    def extract_parameter_data_as_dict(primitives, n_max_primitives):
        return {}

    def extract_predicted_parameters_as_json(fetched, k):
        sphere = Sphere(fetched['sphere_center'][k], np.sqrt(fetched['sphere_radius_squared'][k]))

        return {
            'type': 'sphere',
            'center_x': sphere.center[0],
            'center_y': sphere.center[1],
            'center_z': sphere.center[2],
            'radius': sphere.radius,
        }

    def create_primitive_from_dict(d):
        assert d['type'] == 'sphere'
        location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
        radius = float(d['radius'])
        return Sphere(center=location, radius=radius)
