#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .add_pythonconvention_method import *
from .deprecated import *


class Translation(object):
    def __init__(self, input=None):
        if input is None:
            self.__translation_vector = np.array([0, 0, 0])
        elif type(input) is np.ndarray:
            self.__translation_vector = input.copy()

        assert self.__translation_vector is not None

    @deprecated
    def getTranslationVector(self):
        return self.__translation_vector.copy()

    @add_pythonconvention_method(getTranslationVector)
    def get_translation_vector(self):
        pass

    @deprecated
    def getLength(self):
        return np.sqrt(np.sum(self.__translation_vector ** 2))

    @add_pythonconvention_method(getLength)
    def get_length(self):
        pass

    def d_transl(self, other):
        t = Translation(self.get_translation_vector() - other.get_translation_vector())
        return t.get_length()

    def __repr__(self):
        return 'Translation: ' + str(self.__translation_vector)

    def __str__(self):
        return self.__repr__()

    @deprecated
    def toLnString(self):
        return str(self.get_translation_vector())

    @add_pythonconvention_method(toLnString)
    def to_ln_string(self):
        pass

    def __add__(self, other):
        return Translation(self.get_translation_vector() + other.get_translation_vector())

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Translation(self.get_translation_vector() - other.get_translation_vector())

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        assert type(other) is float or type(other) is int
        return Translation(self.get_translation_vector() * float(other))

    def __imul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, index):
        return self.get_translation_vector()[index]

    def to_pretty_string(self):

        def f(n):
            return '{:.3f}'.format(n)

        s = ''
        for line in self.get_translation_vector():
            s += f(line) + '\n'
        return s[:-1]
