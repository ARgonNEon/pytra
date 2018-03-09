#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg as la

from .add_pythonconvention_method import *
from .deprecated import *
from .rotation import Rotation
from .translation import Translation


class Transformation(Rotation, Translation):
    euler_convention = {'x': 0, 'y': 1, 'z': 2, 'a': 3, 'b': 4, 'c': 5}
    euler_translational_parts = ['x', 'y', 'z']
    euler_rotational_parts = ['a', 'b', 'c']

    @staticmethod
    def of_xyzABC_rad(x, y, z, A, B, C):
        return Transformation(np.array([x, y, z, A, B, C], dtype=float), False)

    @staticmethod
    def of_xyzABC_deg(x, y, z, A, B, C):
        return Transformation(np.array([x, y, z, A, B, C], dtype=float), True)

    @staticmethod
    def of_3_by_4_matrix(r11, r12, r13, x, r21, r22, r23, y, r31, r32, r33, z):
        matrix = np.array([[r11, r12, r13, x],
                           [r21, r22, r23, y],
                           [r31, r32, r33, z],
                           [0, 0, 0, 1]])
        return Transformation(matrix)

    def __init__(self, input=None, useDegrees=False):
        if input is None:
            Rotation.__init__(self)
            Translation.__init__(self)
            return
        elif type(input) is np.ndarray:
            if input.shape == (4, 4):
                rotmat = input[:3, :3]
                Rotation.__init__(self, rotmat, useDegrees=useDegrees)
                Translation.__init__(self, np.array([input[0, 3], input[1, 3], input[2, 3]]))
                return
            elif input.shape == (6,):
                euler_angles = input[3:]
                Rotation.__init__(self, euler_angles, useDegrees=useDegrees)
                Translation.__init__(self, input[:3])
                return
        elif type(input) is tuple:
            if type(input[0]) is Translation and type(input[1]) is Rotation:
                Rotation.__init__(self, input[1].get_rotation_matrix())
                Translation.__init__(self, input[0].get_translation_vector())
                return

        raise TypeError('Transformation representation not understood')

    @deprecated
    def getEulerArray(self, useDegrees=False):
        trans = self.get_translation_vector()
        rot = self.get_euler_angles(useDegrees=useDegrees)
        return np.array([trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]])

    @add_pythonconvention_method(getEulerArray)
    def get_euler_array(self):
        pass

    def __getitem__(self, index):
        return self.get_euler_array()[index]

    @deprecated
    def getTransformationMatrix(self):
        res = np.hstack([self.get_rotation_matrix(), self.get_translation_vector().reshape(3, 1)])
        return np.vstack([res, np.array([0, 0, 0, 1])])

    @add_pythonconvention_method(getTransformationMatrix)
    def get_transformation_matrix(self):
        pass

    @add_pythonconvention_method(getTransformationMatrix)
    def get_matrix(self):
        pass

    @deprecated
    def getRotation(self):
        return Rotation(self.get_rotation_matrix())

    @add_pythonconvention_method(getRotation)
    def get_rotation(self):
        pass

    @deprecated
    def getTranslation(self):
        return Translation(self.get_translation_vector())

    @add_pythonconvention_method(getTranslation)
    def get_translation(self):
        pass

    def transform(self, other):
        assert type(other) is Transformation
        return Transformation(self.get_transformation_matrix().dot(other.get_transformation_matrix()))

    def invert(self):
        hom = self.get_transformation_matrix()
        return Transformation(la.inv(hom))

    def d(self, other):
        return self.get_rotation().d_geod(other.get_rotation()) + self.get_translation().d_transl(
            other.get_translation())

    def __repr__(self):
        return str(self.get_transformation_matrix())

    def __str__(self):
        return self.__repr__()

    def __mul__(self, other):
        return self.transform(other)

    def __imul__(self, other):
        return self.__mul__(other)

    @deprecated
    def toLnString(self, useDegrees=False):
        s = ''
        for dof in self.get_euler_array(useDegrees):
            s += str(dof) + ' '
        return s[:-1]

    @add_pythonconvention_method(toLnString)
    def to_ln_string(self):
        pass

    def to_pretty_string(self):
        s = ''

        def f(n):
            return '{:.3f}'.format(n)

        mat = self.get_transformation_matrix()
        for line in mat:
            for col in line:
                s += f(col) + '\t'
            s = s[:-1]
            s += '\n'
        return s
