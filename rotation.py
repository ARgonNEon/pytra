#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .quaternion import Quaternion
from .trafo_utils import *
import numpy as np
from scipy import linalg as la
from .add_pythonconvention_method import *
from .deprecated import *


class Rotation(object):
    '''Class to handle different Representation of orientation and rotation

        Rotation representations are available in:
            - Rotation matrix
            - Axis Angle
            - Euler Angles (X-Y'-Z'')
            - Quaternion
            - Gnonomic Projection (Gibb's Vector) (not as input)
            - Skew Symmetric so(3)-Matrix (not as input)'''

    verbose = False  # True
    logm_repair_epsilon = 1e-6

    def __init__(self, input=None, useDegrees=False):
        self.__rotation_matrix = None
        self.__quaternion = None
        self.__axis_angle = None
        self.__euler = None
        self.__quaternion = None
        self.__gibbs = None
        self.__axis_angle_vector = None
        self.__skew = None
        if input is None:
            self.__rotation_matrix = np.eye(3)
            return
        elif type(input) is np.ndarray:
            if input.shape == (3, 3):
                if Rotation.verbose:
                    print('input is rotation matrix')
                self.__rotation_matrix = input.copy()
                return
            if input.shape == (3,):
                if Rotation.verbose:
                    print('input are euler angles')
                func = kuka_to_trafo_deg if useDegrees else kuka_to_trafo_rad
                self.__rotation_matrix = np.array(func(0, 0, 0, input[0], input[1], input[2])[:3, :3])
                func = np.deg2rad if useDegrees else lambda x: x
                self.__euler = func(input)
                return
        elif type(input) is tuple:
            if Rotation.verbose:
                print('input is axis-angle')
            self.__rotation_matrix = np.array(axis_angle_to_rotation_matrix(np.matrix(input[0]), input[
                1] if not useDegrees else np.deg2rad(input[1])))
            self.__axis_angle = (input[0], input[1] if not useDegrees else np.deg2rad(input[1]))
            return
        elif type(input) is Quaternion:
            if Rotation.verbose:
                print('input is Quaternion')
            self.__rotation_matrix = axis_angle_to_rotation_matrix(input.get_axis(), input.get_angle())
            self.__quaternion = input
            return

        raise TypeError('Rotation representation not understood')

    @deprecated
    def getRotationMatrix(self):
        return self.__rotation_matrix.copy()

    @add_pythonconvention_method(getRotationMatrix)
    def get_rotation_matrix(self):
        pass

    @deprecated
    def getEulerAngles(self, useDegrees=False):
        if self.__euler is None:
            self.__euler = rotation_matrix_to_euler_angles_rad(np.matrix(self.get_rotation_matrix()))
        func = np.rad2deg if useDegrees else lambda x: x
        return func(self.__euler.copy())

    @add_pythonconvention_method(getEulerAngles)
    def get_euler_angles(self):
        pass

    @deprecated
    def getAxisAngle(self):
        if self.__axis_angle is None:
            self.__axis_angle = rotation_matrix_to_axis_angle(self.get_rotation_matrix())
        return self.__axis_angle

    @add_pythonconvention_method(getAxisAngle)
    def get_axis_angle(self):
        pass

    @deprecated
    def getQuaternion(self):
        if self.__quaternion == None:
            self.__quaternion = Quaternion.create_from_rotation_matrix(self.get_rotation_matrix())
        return self.__quaternion

    @add_pythonconvention_method(getQuaternion)
    def get_quaternion(self):
        pass

    @deprecated
    def getSkew(self):
        if self.__skew is None:
            axis, angle = self.get_axis_angle()[0], self.get_axis_angle()[1]
            axis = np.array(axis)
            sk = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            self.__skew = sk * angle
        return self.__skew

    @add_pythonconvention_method(getSkew)
    def get_skew(self):
        pass

    @deprecated
    def getAxisAngleVector(self):
        if self.__axis_angle_vector is None:
            axis, angle = self.get_axis_angle()[0], self.get_axis_angle()[1]
            self.__axis_angle_vector = angle * axis
        return self.__axis_angle_vector

    @add_pythonconvention_method(getAxisAngleVector)
    def get_axis_angle_vector(self):
        pass

    @deprecated
    def getGibbsVector(self):
        if self.__gibbs is None:
            theta = self.get_quaternion().get_angle()
            axis = self.get_quaternion().get_axis()
            axis = np.array(axis)
            self.__gibbs = axis * np.tan(theta / 2.0)
        return self.__gibbs

    @add_pythonconvention_method(getGibbsVector)
    def get_gibbs_vector(self):
        pass

    def rotate(self, other):
        assert type(other) is Rotation
        return Rotation(self.get_rotation_matrix().dot(other.get_rotation_matrix()))

    def invert(self):
        mat = self.get_rotation_matrix()
        return Rotation(la.inv(mat))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        result = ''
        representations = [getattr(self, method) for method in dir(Rotation) if
                           callable(getattr(self, method)) and method.startswith('get') and not method[3] == '_']
        for representation in representations:
            result += representation.func_name[3:] + '\n'
            result += str(representation())
            result += '\n'
        return result

    @deprecated
    def toLnString(self):
        a = self.get_rotation_matrix()
        return str(a)

    @add_pythonconvention_method(toLnString)
    def to_ln_string(self):
        pass

    def d_chord(self, other):
        return la.norm(self.get_rotation_matrix() - other.get_rotation_matrix())

    def d_geod(self, other):
        try:
            ln = la.logm(self.get_rotation_matrix().transpose().dot(other.get_rotation_matrix()))
        except ValueError:
            ln = la.logm(self.get_rotation_matrix().transpose().dot(
                other.get_rotation_matrix()) + Rotation.logm_repair_epsilon * np.ones((3, 3)))
        return (1 / np.sqrt(2)) * la.norm(ln)

    def d_hyp(self, other):
        try:
            ln1 = la.logm(self.get_rotation_matrix())
        except ValueError:
            ln1 = la.logm(self.get_rotation_matrix() + Rotation.logm_repair_epsilon * np.ones((3, 3)))
        try:
            ln2 = la.logm(other.get_rotation_matrix())
        except ValueError:
            ln2 = la.logm(other.get_rotation_matrix() + Rotation.logm_repair_epsilon * np.ones((3, 3)))
        return la.norm(ln1 - ln2)

    def d_quat(self, other):
        return np.array([la.norm(self.get_quaternion().to_mumpy_array() - other.get_quaternion().to_numpy_array()),
                         la.norm(self.get_quaternion().to_numpy_array() + other.get_quaternion().to_numpy_array())]).min()

    def d_dotquat(self, other):
        return np.arccos(self.get_quaternion().dot(other.get_quaternion()))

    def to_pretty_string(self):
        s = ''

        def f(n):
            return '{:.3f}'.format(n)

        mat = self.get_rotation_matrix()
        for line in mat:
            for col in line:
                s += f(col) + '\t'
            s = s[:-1]
            s += '\n'
        return s
