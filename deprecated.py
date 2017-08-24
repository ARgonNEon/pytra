import warnings
from functools import wraps


def deprecated(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        print("Warning: call to deprected function: ".format(func.__name__))
#        warnings.warn_explicit(
#            "Call to deprecated function {}.".format(func.__name__),
#            category=DeprecationWarning,
#            filename=func.func_code.co_filename,
#            lineno=func.func_code.co_firstlineno + 1
#        )
        return func(*args, **kwargs)
    return new_func
