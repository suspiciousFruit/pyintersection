# The script that will build the module
# from distutils.core import setup, Extension
from setuptools import setup, Extension
import sysconfig
import sys
import os.path as path
import site

# Find path to numpy include folder
def get_numpy_include():
    numpy_postfix = path.normpath('numpy/core/include/numpy')
    user_sp = site.getusersitepackages()
    sys_sp = site.getsitepackages()
    for sp in [user_sp] + sys_sp:
        numpy_include = path.join(sp, numpy_postfix)
        if path.exists(numpy_include):
            return numpy_include
    raise Exception('Numpy include directory not found.')

# Find paths to Python.h dir and numpy include dir
def get_include_dirs():
    paths = sysconfig.get_paths()
    python_include = paths['include']
    numpy_include = get_numpy_include()
    return python_include, numpy_include

def get_extra_compile_args():
    return ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']


PYTHON_INCLUDE, NUMPY_INCLUDE = get_include_dirs()
EXTRA_COMPILE_ARGS = get_extra_compile_args()

# Add extension to module
module = Extension('pyintersection',
                    sources = ['src/pymodule/main.cpp'],
                    include_dirs = [PYTHON_INCLUDE, NUMPY_INCLUDE],
                    extra_compile_args = EXTRA_COMPILE_ARGS)

# Make setup
setup(name = 'pyintersection',
        version = '1.0',
        description = 'Pyintersection package',
        ext_modules = [module])
