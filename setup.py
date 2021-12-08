# The script that will build the module

from setuptools import setup, Extension
# from distutils.core import setup, Extension
import ospath

module = Extension('pyintersection',
                    sources = ['src/pymodule/main.cpp'],
                    include_dirs = [ospath.PYTHON_INCLUDE, ospath.NUMPY_INCLUDE],
                    extra_compile_args = ["/std:c++17"])

setup(name = 'pyintersection',
        version = '1.0',
        description = 'This is a demo package',
        ext_modules = [module])