import numpy as np
import math
import random
import torch
import matplotlib.pyplot as plt

def normalized(v):
    return v / np.linalg.norm(v)

def make_rand_unit_vector(dims=3):
    vec = np.array([random.gauss(0, 1) for i in range(dims)])
    return normalized(vec)

class Plane: # A finite plane patch spanned by x_axis and y_axis
    @staticmethod
    def get_primitive_name():
        return 'plane'

    def __init__(self, n=None, c=None, center=None, x_axis=None, y_axis=None, x_range=[-1, 1],  y_range=[-1, 1], obj_conf=1):
        if type(n) is not np.ndarray:
            print('Normal {} needs to be a numpy array!'.format(n))
            raise
        # Plane is defined by {p: n^T p = c}, where the bound is determined by xy_range w.r.t. center
        if center is None:
            center = n * c
        self.n = n
        self.c = c
        self.center = center
        self.x_range = x_range
        self.y_range = y_range
        self.obj_conf = obj_conf
        self.cls_conf = 1

        # parameterize the plane by picking axes
        if x_axis is None or y_axis is None:
            ax_tmp = make_rand_unit_vector()
            self.x_axis = normalized(np.cross(ax_tmp, self.n))
            self.y_axis = normalized(np.cross(self.n, self.x_axis))
        else:
            self.x_axis = x_axis
            self.y_axis = y_axis

    def get_corners(self):
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        orign = [0,0,0]
        size = [1,1,0]
        o = [a - b / 2 for a, b in zip(orign, size)]   # (-1/2, -1/2, 0)
        # get the length, width, and height
        l, w, h = size                                  # (1, 1, 0)
        x = np.array([[o[0], o[0] + l], 
                     [o[0], o[0] + l]])                # x coordinate of points in bottom surface

        y = np.array([[o[1], o[1]], 
                     [o[1] + w, o[1] + w]])                # y coordinate of points in bottom surface

        z = np.array([[o[2], o[2]], 
                     [o[2], o[2]]])                        # z coordinate of points in bottom surface

        XYZ = np.stack([x.flatten(), y.flatten(), z.flatten()])
        V = np.vstack((self.x_axis, self.y_axis, self.n))
        s = np.hstack((sum(self.x_range), sum(self.y_range), 0))
        # s = np.hstack((2 * self.x_range[1], 2 * self.y_range[1], 0))
        x, y, z = V.T @ (s[:, None] * XYZ) + self.center[:, None]
        x = x.reshape(2,2)
        y = y.reshape(2,2)
        z = z.reshape(2,2)
        return x,y,z

    def get_area(self):
        return (self.x_range[1]-self.x_range[0])*(self.y_range[1]-self.y_range[0])*np.linalg.norm(np.cross(self.x_axis, self.y_axis))

    def distance_to(self, p): # p should be point as a numpy array
        return abs(np.dot(self.n, p) - self.c)

    def sample_single_point(self, noise_radius=0.0):
        origin = self.center
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        p = origin + x * self.x_axis + y * self.y_axis
        if noise_radius > 0:
            p += random.uniform(0, noise_radius) * make_rand_unit_vector()
        return (p, self.n)

    def plot(self, ax):
        x, y, z = self.get_corners()
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(121, projection='3d')
        # ax.view_init(azim=60, elev=0)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=self.obj_conf)
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.n[0], self.n[1], self.n[2], color="r")
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.x_axis[0], self.x_axis[1], self.x_axis[2], color="r")
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.y_axis[0], self.y_axis[1], self.y_axis[2], color="r")
        # plt.show()
    
    def extract_predicted_parameters_as_json(self):
        json_info = {
            'type': 'plane',
            'center_x': self.center[0],
            'center_y': self.center[1],
            'center_z': self.center[2], 
            'normal_x': self.n[0],
            'normal_y': self.n[1],
            'normal_z': self.n[2],
            'x_size': self.x_range[1] - self.x_range[0],
            'y_size': self.y_range[1] - self.y_range[0],
            'x_axis_x': self.x_axis[0],
            'x_axis_y': self.x_axis[1],
            'x_axis_z': self.x_axis[2],
            'y_axis_x': self.y_axis[0],
            'y_axis_y': self.y_axis[1],
            'y_axis_z': self.y_axis[2],
        }
        return json_info

    @classmethod
    def create_random(cls, intercept_range=[-1, 1]):
        return cls(make_rand_unit_vector(), random.uniform(*intercept_range))

class Cuboid: # A finite plane patch spanned by x_axis and y_axis
    @staticmethod
    def get_primitive_name():
        return 'cuboid'

    def __init__(self, center=None, x_axis=None, y_axis=None, z_axis=None, x_range=[-1, 1],  y_range=[-1, 1], z_range=[-1, 1], obj_conf=1):
        # if type(n) is not np.ndarray:
        #     print('Normal {} needs to be a numpy array!'.format(n))
        #     raise
        # Plane is defined by {p: n^T p = c}, where the bound is determined by xy_range w.r.t. center
        self.center = center
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.obj_conf = obj_conf
        self.cls_conf = 1

        # parameterize the plane by picking axes
        if x_axis is None or y_axis is None:
            ax_tmp = make_rand_unit_vector()
            self.x_axis = normalized(np.cross(ax_tmp, self.n))
            self.y_axis = normalized(np.cross(self.n, self.x_axis))
        else:
            self.x_axis = x_axis
            self.y_axis = y_axis
            self.z_axis = z_axis

    def get_volume(self):
        return self.x_range[0] * self.y_range[0] * self.z_range[0]

    def get_cuboid_vertices(self):
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        orign = [0,0,0]
        size = [1,1,1]
        o = [a - b / 2 for a, b in zip(orign, size)]   # (-1/2, -1/2, 0)
        # get the length, width, and height
        l, w, h = size                                  # (1, 1, 0)
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
    
        XYZ = np.stack([np.array(x).flatten(), np.array(y).flatten(), np.array(z).flatten()])
        V = np.vstack((self.x_axis, self.y_axis, self.z_axis))
        s = np.hstack((sum(self.x_range), sum(self.y_range), sum(self.z_range)))
        # s = np.hstack((2 * self.x_range[1], 2 * self.y_range[1], 0))
        x, y, z = V.T @ (s[:, None] * XYZ) + self.center[:, None]
        return x,y,z

    def get_box_corners(self):
        x,y,z = self.get_cuboid_vertices()
        x_corners = x[np.r_[0:4,5:9]]
        y_corners = y[np.r_[0:4,5:9]]
        z_corners = z[np.r_[0:4,5:9]]
        corners_3d = np.vstack([x_corners,y_corners,z_corners]).T
        return torch.tensor(corners_3d)

    def cuboid_faces(self, k):
        bottom = [[0,1,2], [0,2,3]]
        upper = [[4,5,6], [4,6,7]]
        front = [[0,1,5], [0,5,4]]
        back = [[3,2,6], [3,6,7]]
        left = [[0,4,7], [0,7,3]]
        right = [[1,2,6], [1,6,5]]
        return torch.tensor(bottom + upper + front + back + left + right) + 8 * k 

    def get_mesh(self, k):
        corners = self.get_box_corners()
        box_faces = self.cuboid_faces(k)
        return corners, box_faces

    def get_area(self):
        return (self.x_range[1]-self.x_range[0])*(self.y_range[1]-self.y_range[0])*np.linalg.norm(np.cross(self.x_axis, self.y_axis))

    def distance_to(self, p): # p should be point as a numpy array
        return abs(np.dot(self.n, p) - self.c)

    def sample_single_point(self, noise_radius=0.0):
        origin = self.center
        x = random.uniform(*self.x_range)
        y = random.uniform(*self.y_range)
        p = origin + x * self.x_axis + y * self.y_axis
        if noise_radius > 0:
            p += random.uniform(0, noise_radius) * make_rand_unit_vector()
        return (p, self.n)

    def plot(self, ax):
        x, y, z = self.get_cuboid_vertices()
        x = x.reshape(4, 5)
        y = y.reshape(4, 5)
        z = z.reshape(4, 5)
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(121, projection='3d')
        # ax.view_init(azim=60, elev=0)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=self.obj_conf)
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.n[0], self.n[1], self.n[2], color="r")
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.x_axis[0], self.x_axis[1], self.x_axis[2], color="r")
        # ax.quiver(self.center[0], self.center[1], self.center[2], self.y_axis[0], self.y_axis[1], self.y_axis[2], color="r")
        # plt.show()
    
    def extract_predicted_parameters_as_json(self):
        json_info = {
            'type': 'plane',
            'center_x': self.center[0],
            'center_y': self.center[1],
            'center_z': self.center[2], 
            'x_size': self.x_range[1] - self.x_range[0],
            'y_size': self.y_range[1] - self.y_range[0],
            'x_axis_x': self.x_axis[0],
            'x_axis_y': self.x_axis[1],
            'x_axis_z': self.x_axis[2],
            'y_axis_x': self.y_axis[0],
            'y_axis_y': self.y_axis[1],
            'y_axis_z': self.y_axis[2],
        }
        return json_info

    @classmethod
    def create_random(cls, intercept_range=[-1, 1]):
        return cls(make_rand_unit_vector(), random.uniform(*intercept_range))

class Sphere:
    @staticmethod
    def get_primitive_name():
        return 'sphere'

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def get_area(self):
       return 4 * np.pi * self.radius * self.radius

    def sample_single_point(self):
        n = make_rand_unit_vector()
        p = self.center + self.radius * n
        return (p, n)

class Cylinder:
    @staticmethod
    def get_primitive_name():
        return 'cylinder'

    def __init__(self, center, radius, axis, height=10.0):
        self.center = center
        self.radius = radius
        self.axis = axis
        self.height = height

        tmp_axis = make_rand_unit_vector()
        self.x_axis = normalized(np.cross(tmp_axis, self.axis))
        self.y_axis = normalized(np.cross(self.axis, self.x_axis))

    def get_area(self):
        return 2 * np.pi * self.radius * self.height
    
    def sample_single_point(self):
        kx, ky = make_rand_unit_vector(dims=2)
        n = kx * self.x_axis + ky * self.y_axis
        p = random.uniform(-self.height/2, self.height/2) * self.axis + self.radius * n + self.center
        return (p, n)

class Cone:
    @staticmethod
    def get_primitive_name():
        return 'cone'

    def __init__(self, apex, axis, half_angle, z_min=0.0, z_max=10.0):
        self.apex = apex
        self.axis = axis
        self.half_angle = half_angle
        self.z_min = z_min
        self.z_max = z_max
    
class Box:
    def __init__(self, center, axes, halflengths):
        # axes is 3x3, representing an orthogonal frame
        # sidelength is length-3 array
        self.center = center
        self.axes = axes
        self.halflengths = halflengths

    def get_six_planes(self):
        result = []
        for i, axis in enumerate(self.axes):
            for sgn in range(-1, 2, 2):
                n = sgn * axis
                center = self.center + self.halflengths[i] * n
                c = np.dot(n, center)
                j = (i + 1) % 3
                k = (j + 1) % 3
                x_range = [-self.halflengths[j], self.halflengths[j]]
                y_range = [-self.halflengths[k], self.halflengths[k]]
                plane = Plane(n, c, center=center, x_axis=self.axes[j], y_axis=self.axes[k], x_range=x_range, y_range=y_range)
                result.append(plane)

        return result

    @classmethod
    def create_random(cls, center_range=[-1, 1], halflength_range=[0.5,2]):
        center = np.array([random.uniform(*center_range) for _ in range(3)])
        x_axis = make_rand_unit_vector()
        ax_tmp = make_rand_unit_vector()
        y_axis = normalized(np.cross(ax_tmp, x_axis))
        z_axis = normalized(np.cross(x_axis, y_axis))
        axes = [x_axis, y_axis, z_axis]
        halflengths = [random.uniform(*halflength_range) for _ in range(3)]
        return Box(center, axes, halflengths)

