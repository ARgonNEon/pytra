#!/usr/bin/env python


class add_pythonconvention_method(object):
    '''This class used as decorator calls the given original function instead of wrapped function.
    
    Use like this:
    
    class FancyClass(FancyBaseClass):
    
        def getFancyData(self, arg1, arg2):
            do_fancy_stuff()
            a = calculate_fancy_data()
            return a
        
        @add_pythonconvention_method(getFancyData)               #<<<<<< here it is used.
        def get_fancy_data(self):
            pass
            
    '''

    def __init__(self, original):
        self.original = original  # original function to call instead of wrapped function

    def __call__(self, f):
        def wrapped_f(innerself, *args, **kwargs):
            return self.original(innerself, *args, **kwargs)

        return wrapped_f
