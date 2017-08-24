# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .trafo_utils import *
from .add_pythonconvention_method import *
from .deprecated import *


class Quaternion(object):
    '''Quaternion class to handle Quaternion representation of orientation, operators + - and * are overwritten,
    Constructor: Quaternion(a,b,c,d,normelize=True) unset normalize if Quaternion shall not be an union quaternion.
    Quaternion class contains a factory to create quaterion from rotation_matrix, euler angles or axis-angle representation.'''

    # Factory:
    @staticmethod
    def create_from_axis_angle(K, theta):
        a = np.cos(theta / 2)
        b = np.sin(theta / 2) * K[0, 0]
        c = np.sin(theta / 2) * K[1, 0]
        d = np.sin(theta / 2) * K[2, 0]
        return Quaternion(a, b, c, d)

    @staticmethod
    def of_axis_angle(K, theta):
        return Quaternion.create_from_axis_angle(K, theta)

    @staticmethod
    def create_from_rotation_matrix(rotmat):
        axis, angle = rotation_matrix_to_axis_angle(rotmat)
        return Quaternion.create_from_axis_angle(axis, angle)

    @staticmethod
    def of_rotation_matrix(rotmat):
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def create_from_euler_angles_rad(euler):
        rotmat = kuka_to_trafo_rad(0, 0, 0, euler[0], euler[1], euler[2])[:3, :3]
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def of_euler_angles_rad(euler):
        return Quaternion.create_from_euler_angles_rad(euler)

    @staticmethod
    def create_from_euler_angles_deg(euler):
        rotmat = kuka_to_trafo_deg(0, 0, 0, euler[0], euler[1], euler[2])[:3, :3]
        return Quaternion.create_from_rotation_matrix(rotmat)

    @staticmethod
    def of_euler_angles_deg(euler):
        return Quaternion.create_from_euler_angles_deg(euler)

    @staticmethod
    def create_from_quaternion_array(array):
        return Quaternion(array[0], array[1], array[2], array[3])

    @staticmethod
    def of_quaternion_array(qarray):
        return Quaternion.create_from_quaternion_array(qarray)

    @staticmethod
    def create_neutral_quaternion():
        return Quaternion(0.0, 0.0, 0.0, 1.0)

    # EndFactory

    def __init__(self, a, b, c, d, normalize=True):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if normalize:
            self.normalize()

    def normalize(self):
        length = self.getLength()
        self.a /= length
        self.b /= length
        self.c /= length
        self.d /= length

        if self.a < 0:
            self.a *= -1.0
            self.b *= -1.0
            self.c *= -1.0
            self.d *= -1.0
        return self

    @deprecated
    def getLength(self):
        return np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2)

    @add_pythonconvention_method(getLength)
    def get_length(self):
        pass

    @deprecated
    def getA(self):
        return self.a

    @add_pythonconvention_method(getA)
    def get_a(self):
        pass

    @deprecated
    def getB(self):
        return self.b

    @add_pythonconvention_method(getB)
    def get_b(self):
        pass

    @deprecated
    def getC(self):
        return self.c

    @add_pythonconvention_method(getC)
    def get_c(self):
        pass

    @deprecated
    def getD(self):
        return self.d

    @add_pythonconvention_method(getD)
    def get_d(self):
        pass

    @deprecated
    def getScalar(self):
        return self.getA()

    @add_pythonconvention_method(getScalar)
    def get_scalar(self):
        pass

    @deprecated
    def getReal(self):
        return self.getA()

    @add_pythonconvention_method(getReal)
    def get_real(self):
        pass

    @deprecated
    def getVector(self):
        return np.array([self.b, self.c, self.d])

    @add_pythonconvention_method(getVector)
    def get_vector(self):
        pass

    @deprecated
    def getImag(self):
        return self.getVector()

    @add_pythonconvention_method(getImag)
    def get_imag(self):
        pass

    @deprecated
    def getAngle(self):
        return 2 * np.arccos(self.getA())

    @add_pythonconvention_method(getAngle)
    def get_angle(self):
        pass

    @deprecated
    def getAxis(self):
        if self.get_angle() == 0.0:
            return np.matrix([[1], [0], [0]])
        return 1 / (np.sin(self.get_angle() / 2)) * np.matrix([[self.get_b()], [self.get_c()], [self.get_d()]])

    @add_pythonconvention_method(getAxis)
    def get_axis(self):
        pass

    @deprecated
    def toNumpyArray(self):
        return np.array([self.a, self.b, self.c, self.d])

    @add_pythonconvention_method(toNumpyArray)
    def to_numpy_array(self):
        pass

    def add(self, quat):
        a = self.a + quat.get_a()
        b = self.b + quat.get_b()
        c = self.c + quat.get_c()
        d = self.d + quat.get_d()
        return Quaternion(a, b, c, d, False)

    def sub(self, quat):
        a = self.a - quat.get_a()
        b = self.b - quat.get_b()
        c = self.c - quat.get_c()
        d = self.d - quat.get_d()
        return Quaternion(a, b, c, d, False)

    def dot(self, quat):
        return self.a * quat.get_a() + self.b * quat.get_b() + self.c * quat.get_c() + self.d * quat.get_d()

    def outer(self, quat):
        a = np.array([self.a, self.b, self.c, self.d])
        b = np.array([quat.get_a(), quat.get_b(), quat.get_c(), quat.get_d()])
        return np.outer(a, b)

    def mul(self, quat, normalize=True):
        a = self.a * quat.get_a() - self.b * quat.get_b() - self.c * quat.get_c() - self.d * quat.get_d()
        b = self.a * quat.get_b() + self.b * quat.get_a() + self.c * quat.get_d() - self.d * quat.get_c()
        c = self.a * quat.get_c() - self.b * quat.get_d() + self.c * quat.get_a() + self.d * quat.get_b()
        d = self.a * quat.get_d() + self.b * quat.get_c() - self.c * quat.get_b() + self.d * quat.get_a()
        return Quaternion(a, b, c, d, normalize)

    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    @deprecated
    def getAngleToOther(self, other):
        a = self.get_axis()
        b = other.get_axis()
        cosa = (a.T.dot(b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        return np.arccos(cosa)

    @add_pythonconvention_method(getAngleToOther)
    def get_angle_to_other(self):
        pass

    @deprecated
    def getErrorQuaternion(self, meanquat):
        return self.conjugate() * meanquat

    @add_pythonconvention_method(getErrorQuaternion)
    def get_error_quaternion(self):
        pass

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        if type(other) is int or type(other) is float:
            return Quaternion(self.a * float(other), self.b * float(other), self.c * float(other),
                              self.d * float(other), False)
        elif type(other) is Quaternion:
            return self.mul(other)
        else:
            raise Exception('unexpected type for quaternion multiplication')

    def __imul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return 'Quaternion: [{0} {1} {2} {3}]'.format(self.a, self.b, self.c, self.d)

    def __repr__(self):
        return self.__str__()
