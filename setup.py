# The script that will build the module
# from distutils.core import setup, Extension

from setuptools import setup, Extension
import sysconfig
import sys
from pathlib import Path

def get_include_dirs():
    paths = sysconfig.get_paths()
    python_include = paths['include']
    numpy_include = paths['purelib'] + str(Path('/numpy/core/include/numpy'))
    return python_include, numpy_include

def get_extra_compile_args():
    return ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17']

# def get_platform_postfix():
#     if sys.platform == 'win32':
#         return '\\numpy\\core\\include\\numpy'
#     else:
#         return '/numpy/core/include/numpy'

PYTHON_INCLUDE, NUMPY_INCLUDE = get_include_dirs()
EXTRA_COMPILE_ARGS = get_extra_compile_args()


module = Extension('pyintersection',
                    sources = ['src/pymodule/main.cpp'],
                    include_dirs = [PYTHON_INCLUDE, NUMPY_INCLUDE],
                    extra_compile_args = EXTRA_COMPILE_ARGS)

setup(name = 'pyintersection',
        version = '1.0',
        description = 'Pyintersection package',
        ext_modules = [module])